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
