"""
NurtureAI - Comprehensive Module Test Suite
Tests all 5 model modules individually and checks app.py integration.
"""

import sys
import os
import json
import importlib.util
import traceback

# Force UTF-8 output on Windows to avoid UnicodeEncodeError
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
INFO = "[INFO]"

results = {}

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  {status} {label}" + (f" — {detail}" if detail else ""))
    return condition

def _import(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─────────────────────────────────────────────────────────────
# 1. CELIAC DISEASE MODEL
# ─────────────────────────────────────────────────────────────
section("1. CELIAC DISEASE MODEL (celiac-disease/celiac_model.py)")
try:
    celiac = _import("celiac_model", os.path.join(BASE_DIR, "celiac-disease", "celiac_model.py"))

    load_res = celiac.load()
    loaded = check("load() returns code 200", load_res.get("code") == 200, str(load_res))
    results["celiac_load"] = loaded

    if loaded:
        # Valid positive case
        r1 = celiac.predict({
            "Age": 25, "Gender": "Female", "Diabetes": "Yes",
            "Diabetes Type": "Type 1", "Diarrhoea": "Yes", "Abdominal": "Yes",
            "Short_Stature": "No", "Sticky_Stool": "Yes", "Weight_loss": "Yes",
            "IgA": 5.2, "IgG": 15.1, "IgM": 1.8
        })
        results["celiac_predict"] = check("predict() valid input -> code 200", r1.get("code") == 200, str(r1))
        check("predict() returns 'prediction' key", "prediction" in r1)

        # Invalid input — missing field
        r2 = celiac.predict({"Age": 25, "Gender": "Female"})
        check("predict() missing fields -> code 400", r2.get("code") == 400, str(r2))

        # Invalid encoder value
        r3 = celiac.predict({
            "Age": 25, "Gender": "Unknown", "Diabetes": "Yes",
            "Diabetes Type": "Type 1", "Diarrhoea": "Yes", "Abdominal": "Yes",
            "Short_Stature": "No", "Sticky_Stool": "Yes", "Weight_loss": "Yes",
            "IgA": 5.2, "IgG": 15.1, "IgM": 1.8
        })
        check("predict() invalid enum -> graceful error (code 400/500)", r3.get("code") in (400, 500), str(r3))
except Exception as e:
    print(f"  {FAIL} Module import/load crashed: {e}")
    traceback.print_exc()
    results["celiac_load"] = False
    results["celiac_predict"] = False


# ─────────────────────────────────────────────────────────────
# 2. BEHAVIOUR ANALYSIS MODEL
# ─────────────────────────────────────────────────────────────
section("2. BEHAVIOUR ANALYSIS MODEL (behaviour_analysis/nurture_model.py)")
try:
    beh_pkl = os.path.join(BASE_DIR, "behaviour_analysis", "nurture_model.pkl")
    behaviour = _import("behaviour_model", os.path.join(BASE_DIR, "behaviour_analysis", "nurture_model.py"))

    load_res = behaviour.load(pkl_path=beh_pkl)
    loaded = check("load() returns status 200", load_res.get("status") == 200, str(load_res))
    results["behaviour_load"] = loaded

    if loaded:
        valid_input = {
            "General_Health": 3, "Sleep_Hours": 7, "Exercise_Any": 1,
            "Smoked_100_Cigs": 2, "Income_Level": 6, "Marital_Status": 1,
        }
        r1 = behaviour.predict(valid_input)
        results["behaviour_predict"] = check("predict() valid 6-feat input -> status 200", r1.get("status") == 200, str(r1))
        check("predict() returns 'prediction' key", "prediction" in r1)

        # Extended features
        ext_input = {**valid_input,
            "Physical_Health_Days": 2, "Mental_Health_Days": 3,
            "Depression_Diagnosis": 2, "BMI_Indicator": 24.5, "Alcohol_Days_Monthly": 4
        }
        r2 = behaviour.predict(ext_input)
        check("predict() extended features -> status 200", r2.get("status") == 200, str(r2))
        check("predict() extended features -> has risk_scores", "risk_scores" in r2)

        # Missing required field
        r3 = behaviour.predict({"General_Health": 3})
        check("predict() missing features -> status 400", r3.get("status") == 400, str(r3))

        # Out-of-range value
        r4 = behaviour.predict({**valid_input, "General_Health": 99})
        check("predict() out-of-range -> status 400", r4.get("status") == 400, str(r4))

        # app.py calls load(pkl_path=...) — verify the status code check compatibility
        # app.py checks: result.get("code", result.get("status", 500))
        status_val = load_res.get("code", load_res.get("status", 500))
        check("load() result compatible with app.py status check (200)", status_val == 200,
              f"status_val={status_val}")
except Exception as e:
    print(f"  {FAIL} Module crashed: {e}")
    traceback.print_exc()
    results["behaviour_load"] = False
    results["behaviour_predict"] = False


# ─────────────────────────────────────────────────────────────
# 3. CHILD MORTALITY MODEL (< 1 year)
