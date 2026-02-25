"""
NurtureAI - Output Coverage Test
Goal: Verify that every possible output class for each model is reachable.

Models & expected classes:
  Celiac         : 0 (Negative), 1 (Positive)
  Behaviour      : Healthy Home Environment, Moderate Risk Environment, High Risk Environment
  Child Mortality: 0 (low risk), 1 (high risk)
  Nurture        : sii=0 (None/Healthy), 1 (Mild), 2 (Moderate), 3 (Severe)
  Chatbot        : input-validation paths (no live API key needed)
"""

import sys, os, importlib.util, json

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PASS = "[PASS]"; FAIL = "[FAIL]"; INFO = "[INFO]"

def section(t): print(f"\n{'='*60}\n  {t}\n{'='*60}")
def ok(label, got, expected_in):
    hit = got in expected_in
    print(f"  {PASS if hit else FAIL}  {label}  ->  got: {repr(got)}")
    return hit

def _import(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

coverage_results = {}   # module -> {class: bool}

# ===========================================================
# 1. CELIAC  — expected outputs: 0 (Negative)  or  1 (Positive)
# ===========================================================
section("1. CELIAC — output classes: 0=Negative, 1=Positive")

celiac = _import("celiac_model", os.path.join(BASE_DIR, "celiac-disease", "celiac_model.py"))
celiac.load()

CELIAC_CASES = [
    {
        # KEY INSIGHT: model treats Diabetes=Yes+no GI symptoms as non-celiac
        "label": "Diabetic but no GI symptoms -> expect 0 (Negative)",
        "expect": 0,
        "data": {
            "Age": 30, "Gender": "Female",
            "Diabetes": "Yes", "Diabetes Type": "Type 1",
            "Diarrhoea": "No", "Abdominal": "No",
            "Short_Stature": "No", "Sticky_Stool": "No",
            "Weight_loss": "No",
            "IgA": 1.5, "IgG": 5.0, "IgM": 0.5
        }
    },
    {
        "label": "Classic celiac symptoms -> expect 1 (Positive)",
        "expect": 1,
        "data": {
            "Age": 8, "Gender": "Female",
            "Diabetes": "Yes", "Diabetes Type": "Type 1",
            "Diarrhoea": "Yes", "Abdominal": "Yes",
            "Short_Stature": "Yes", "Sticky_Stool": "Yes",
            "Weight_loss": "Yes",
            "IgA": 9.5, "IgG": 23.0, "IgM": 4.5
        }
    },
]

celiac_hit = {0: False, 1: False}
for case in CELIAC_CASES:
    r = celiac.predict(case["data"])
    assert r["code"] == 200, f"predict() error: {r}"
    pred = r["prediction"]
    hit  = pred == case["expect"]
    status = PASS if hit else FAIL
    print(f"  {status}  {case['label']}")
    print(f"         prediction={pred}  (expected {case['expect']})")
    celiac_hit[pred] = True

print()
for cls, found in celiac_hit.items():
    label = "Negative" if cls == 0 else "Positive"
    print(f"  {PASS if found else FAIL}  class {cls} ({label}) was observed")
coverage_results["celiac"] = celiac_hit


# ===========================================================
# 2. BEHAVIOUR — 3 classes
# ===========================================================
section("2. BEHAVIOUR — 3 classes")

beh_pkl = os.path.join(BASE_DIR, "behaviour_analysis", "nurture_model.pkl")
beh = _import("behaviour_model", os.path.join(BASE_DIR, "behaviour_analysis", "nurture_model.py"))
beh.load(pkl_path=beh_pkl)

BEHAVIOUR_CLASSES = [
    "Healthy Home Environment",
    "Moderate Risk Environment",
    "High Risk Environment",
]

BEHAVIOUR_CASES = [
    {
        "label": "Excellent health/lifestyle -> Healthy Home Environment",
        "expect": "Healthy Home Environment",
        "data": {
            "General_Health": 1, "Sleep_Hours": 8, "Exercise_Any": 1,
            "Smoked_100_Cigs": 2, "Income_Level": 8, "Marital_Status": 1,
            "Physical_Health_Days": 0, "Mental_Health_Days": 0,
            "Depression_Diagnosis": 2, "BMI_Indicator": 22.0, "Alcohol_Days_Monthly": 0
        }
    },
    {
        "label": "Mixed lifestyle indicators -> Moderate Risk Environment",
        "expect": "Moderate Risk Environment",
        "data": {
            "General_Health": 3, "Sleep_Hours": 6, "Exercise_Any": 2,
            "Smoked_100_Cigs": 1, "Income_Level": 4, "Marital_Status": 2,
            "Physical_Health_Days": 8, "Mental_Health_Days": 10,
            "Depression_Diagnosis": 1, "BMI_Indicator": 27.5, "Alcohol_Days_Monthly": 12
        }
    },
    {
        "label": "Poor health, high risk -> High Risk Environment",
        "expect": "High Risk Environment",
        "data": {
            "General_Health": 5, "Sleep_Hours": 4, "Exercise_Any": 2,
            "Smoked_100_Cigs": 1, "Income_Level": 1, "Marital_Status": 4,
            "Physical_Health_Days": 28, "Mental_Health_Days": 28,
            "Depression_Diagnosis": 1, "BMI_Indicator": 38.0, "Alcohol_Days_Monthly": 25
        }
    },
]

beh_hit = {c: False for c in BEHAVIOUR_CLASSES}
for case in BEHAVIOUR_CASES:
    r = beh.predict(case["data"])
    assert r["status"] == 200, f"predict() error: {r}"
    pred  = r["prediction"]
    hit   = pred == case["expect"]
    top_conf = sorted(r["confidence"].items(), key=lambda x: -x[1])
    status = PASS if hit else FAIL
    print(f"  {status}  {case['label']}")
    print(f"         got: {pred}  (expected: {case['expect']})")
    print(f"         confidence: { {k: round(v,3) for k,v in top_conf} }")
    if pred in beh_hit:
        beh_hit[pred] = True

print()
for cls, found in beh_hit.items():
    print(f"  {PASS if found else FAIL}  class '{cls}' was observed")
coverage_results["behaviour"] = beh_hit


# ===========================================================
# 3. CHILD MORTALITY — output: 1 (Alive/Low risk) or 0 (Dead/High risk)
# ===========================================================
section("3. CHILD MORTALITY — output classes: 1=Low risk, 0=High risk")

child = _import("child_health_model",
                os.path.join(BASE_DIR, "child_mortality", "child_health_model.py"))
child.load()

def _child_base():
    return {feat: 0 for feat in child.FEATURE_ORDER}

CHILD_CASES = [
    {
        "label": "Ideal conditions, full vaccination -> expect 1 (Alive / low risk)",
        "expect": 1,
        "data": {
            **_child_base(),
            "Toilet_Facility": 1, "Child_under5": 1, "Tot_child_born": 1,
            "Sons_died": 0, "Daughters_died": 0,
            "Curr_Preg": 0, "Curr_BrstFeed": 1, "ChildFood_bottle": 0,
            "Resp_height": 170.0, "HealthInsurance": 1, "B_ChildTwin": 0,
            "First3Day_fruitJuice": 0, "HepatitisB_atBirth": 1,
            "ShortBreaths": 0, "VitaminA": 1, "IronPill": 1,
            "IntestinalDrug": 1, "ultrasound": 1, "MMR": 1,
            "DeliveryPlace_Private": 1, "Water_Source_Other": 0,
            "DPT_full": 1, "MEASLES_full": 1,
        }
    },
    {
        "label": "High-risk child (no vaccines, multiple deaths) -> expect 0 (Dead / high risk)",
        "expect": 0,
        "data": {
            **_child_base(),
            "Toilet_Facility": 0, "Child_under5": 5, "Tot_child_born": 9,
            "Sons_died": 3, "Daughters_died": 3,
            "Curr_Preg": 1, "Curr_BrstFeed": 0, "ChildFood_bottle": 1,
            "Resp_height": 145.0, "HealthInsurance": 0, "B_ChildTwin": 1,
            "First3Day_fruitJuice": 1, "HepatitisB_atBirth": 0,
            "ShortBreaths": 1, "VitaminA": 0, "IronPill": 0,
            "IntestinalDrug": 0, "ultrasound": 0, "MMR": 0,
            "DeliveryPlace_Private": 0, "Water_Source_Other": 1,
            "DPT_full": 0, "MEASLES_full": 0,
            "State_Bihar": 1,
        }
    },
]

child_hit = {0: False, 1: False}

for case in CHILD_CASES:
    r = child.predict(case["data"])
    assert r["code"] == 200, f"predict() error: {r}"
    pred  = r["prediction"]
    prob  = round(r.get("probability_class_1", -1), 3)
    hit   = pred == case["expect"]
    status = PASS if hit else FAIL
    print(f"  {status}  {case['label']}")
    print(f"         prediction={pred}  prob_class_1={prob}  (expected {case['expect']})")
    child_hit[pred] = True

print()
for cls, found in child_hit.items():
    label = "High risk (Dead)" if cls == 0 else "Low risk (Alive)"
    print(f"  {PASS if found else FAIL}  class {cls} ({label}) was observed")
coverage_results["child_mortality"] = child_hit


# ===========================================================
# 4. NURTURE MODEL — 4 classes: sii = 0,1,2,3
# ===========================================================
section("4. NURTURE MODEL — sii classes: 0=None, 1=Mild, 2=Moderate, 3=Severe")
