import pickle
import pandas as pd

# REQUIRED INPUT FEATURES (EXACT MODEL ORDER) WITH MEANING, TYPE, AND VALID VALUES.
# 1. Toilet_Facility: Household has toilet facility. Type=int, values: 0 or 1.
# 2. Child_under5: Number of children under age 5. Type=int, range: 0-20.
# 3. Tot_child_born: Total children born to respondent. Type=int, range: 0-30.
# 4. Sons_died: Number of sons who died. Type=int, range: 0-20.
# 5. Daughters_died: Number of daughters who died. Type=int, range: 0-20.
# 6. Curr_Preg: Respondent currently pregnant. Type=int, values: 0 or 1.
# 7. Curr_BrstFeed: Respondent currently breastfeeding. Type=int, values: 0 or 1.
# 8. ChildFood_bottle: Child fed with bottle. Type=int, values: 0 or 1.
# 9. Resp_height: Respondent height in cm. Type=float, range: 30.0-250.0.
# 10. HealthInsurance: Respondent has health insurance. Type=int, values: 0 or 1.
# 11. B_ChildTwin: Birth was a twin birth. Type=int, values: 0 or 1.
# 12. First3Day_fruitJuice: Child received fruit juice in first 3 days. Type=int, values: 0 or 1.
# 13. HepatitisB_atBirth: Hepatitis B dose at birth given. Type=int, values: 0 or 1.
# 14. ShortBreaths: Child has shortness of breath symptom. Type=int, values: 0 or 1.
# 15. VitaminA: Child received Vitamin A supplementation. Type=int, values: 0 or 1.
# 16. IronPill: Child/household received iron pills. Type=int, values: 0 or 1.
# 17. IntestinalDrug: Child received deworming/intestinal drug. Type=int, values: 0 or 1.
# 18. ultrasound: Ultrasound done during pregnancy. Type=int, values: 0 or 1.
# 19. MMR: Child received MMR vaccine indicator. Type=int, values: 0 or 1.
# 20. DeliveryPlace_Private: Delivery in private facility. Type=int, values: 0 or 1.
# 21. Water_Source_Other: Water source category is "Other". Type=int, values: 0 or 1.
# 22. DPT_full: Child completed full DPT schedule. Type=int, values: 0 or 1.
# 23. MEASLES_full: Child completed measles dose(s). Type=int, values: 0 or 1.
# 24. State_Bihar: One-hot state flag for Bihar. Type=int, values: 0 or 1.
# 25. State_Jharkhand: One-hot state flag for Jharkhand. Type=int, values: 0 or 1.
# 26. State_Meghalaya: One-hot state flag for Meghalaya. Type=int, values: 0 or 1.
# 27. State_Mizoram: One-hot state flag for Mizoram. Type=int, values: 0 or 1.
# 28. State_Sikkim: One-hot state flag for Sikkim. Type=int, values: 0 or 1.
# 29. State_Uttar Pradesh: One-hot state flag for Uttar Pradesh. Type=int, values: 0 or 1.
# 30. State_Uttarakhand: One-hot state flag for Uttarakhand. Type=int, values: 0 or 1.
#
# PREPROCESSING HANDLED IN CODE:
# - Categorical aliases accepted and converted to one-hot model columns:
#   delivery_place -> DeliveryPlace_Private, water_source -> Water_Source_Other,
#   state -> one of the State_* columns (all 0 if not in modeled states).
# - Direct model-column input is also accepted.

FEATURE_ORDER = [
    "Toilet_Facility",
    "Child_under5",
    "Tot_child_born",
    "Sons_died",
    "Daughters_died",
    "Curr_Preg",
    "Curr_BrstFeed",
    "ChildFood_bottle",
    "Resp_height",
    "HealthInsurance",
    "B_ChildTwin",
    "First3Day_fruitJuice",
    "HepatitisB_atBirth",
    "ShortBreaths",
    "VitaminA",
    "IronPill",
    "IntestinalDrug",
    "ultrasound",
    "MMR",
    "DeliveryPlace_Private",
    "Water_Source_Other",
    "DPT_full",
    "MEASLES_full",
    "State_Bihar",
    "State_Jharkhand",
    "State_Meghalaya",
    "State_Mizoram",
    "State_Sikkim",
    "State_Uttar Pradesh",
    "State_Uttarakhand",
]

BINARY_FEATURES = {
    "Toilet_Facility",
    "Curr_Preg",
    "Curr_BrstFeed",
    "ChildFood_bottle",
    "HealthInsurance",
    "B_ChildTwin",
    "First3Day_fruitJuice",
    "HepatitisB_atBirth",
    "ShortBreaths",
    "VitaminA",
    "IronPill",
    "IntestinalDrug",
    "ultrasound",
    "MMR",
    "DeliveryPlace_Private",
    "Water_Source_Other",
    "DPT_full",
    "MEASLES_full",
    "State_Bihar",
    "State_Jharkhand",
    "State_Meghalaya",
    "State_Mizoram",
    "State_Sikkim",
    "State_Uttar Pradesh",
    "State_Uttarakhand",
}

COUNT_FEATURES = {
    "Child_under5",
    "Tot_child_born",
    "Sons_died",
    "Daughters_died",
}

STATE_FEATURES = {
    "bihar": "State_Bihar",
    "jharkhand": "State_Jharkhand",
    "meghalaya": "State_Meghalaya",
    "mizoram": "State_Mizoram",
    "sikkim": "State_Sikkim",
    "uttar pradesh": "State_Uttar Pradesh",
    "uttarakhand": "State_Uttarakhand",
}

model = None


def _to_binary(value):
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return 1
        if lowered in {"0", "false", "no", "n"}:
            return 0
    return 1 if bool(value) else 0


def _normalize_feature_value(feature_name, value):
    if feature_name in BINARY_FEATURES:
        return _to_binary(value)
    if feature_name in COUNT_FEATURES:
        return max(0, int(value))
    if feature_name == "Resp_height":
        h = float(value)
        # Convert cm to meters if value is large
        if h > 5.0:
            return h / 100.0
        return h
    return float(value)


def _apply_alias_preprocessing(row, input_data):
    delivery_place = input_data.get("delivery_place", input_data.get("DeliveryPlace"))
    if delivery_place is not None and "DeliveryPlace_Private" not in input_data:
