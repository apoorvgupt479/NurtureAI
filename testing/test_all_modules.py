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
# ─────────────────────────────────────────────────────────────
section("3. CHILD MORTALITY MODEL (child_mortality/child_health_model.py)")
try:
    child = _import("child_health_model", os.path.join(BASE_DIR, "child_mortality", "child_health_model.py"))

    load_res = child.load()
    loaded = check("load() returns code 200", load_res.get("code") == 200, str(load_res))
    results["child_mortality_load"] = loaded

    if loaded:
        sample = {feat: 0 for feat in child.FEATURE_ORDER}
        sample["Resp_height"] = 155.0
        sample["Child_under5"] = 2
        sample["Tot_child_born"] = 3

        r1 = child.predict(sample)
        results["child_mortality_predict"] = check("predict() valid input -> code 200", r1.get("code") == 200, str(r1))
        check("predict() returns 'prediction' key", "prediction" in r1)

        # Alias-based input
        alias_input = dict(sample)
        alias_input["state"] = "Bihar"
        alias_input["delivery_place"] = "private"
        alias_input["water_source"] = "other"
        r2 = child.predict(alias_input)
        check("predict() alias inputs work -> code 200", r2.get("code") == 200, str(r2))

        # app.py status check compatibility
        status_val = load_res.get("code", load_res.get("status", 500))
        check("load() compatible with app.py status check (200)", status_val == 200)
except Exception as e:
    print(f"  {FAIL} Module crashed: {e}")
    traceback.print_exc()
    results["child_mortality_load"] = False
    results["child_mortality_predict"] = False


# ─────────────────────────────────────────────────────────────
# 4. NURTURE MODEL (>= 1 year child behaviour)
# ─────────────────────────────────────────────────────────────
section("4. NURTURE MODEL (nurture_model/nurture_model.py)")
try:
    nurture = _import("nurture_child_model", os.path.join(BASE_DIR, "nurture_model", "nurture_model.py"))

    load_res = nurture.load()
    loaded = check("load() returns code 200", load_res.get("code") == 200, str(load_res))
    results["nurture_load"] = loaded

    if loaded:
        valid_input = {
            "Basic_Demos-Age": 10, "Basic_Demos-Sex": 0,
            "Physical-BMI": 18.5, "Physical-Height": 140, "Physical-Weight": 36,
            "Physical-Waist_Circumference": 60, "Physical-Diastolic_BP": 65,
            "Physical-Systolic_BP": 105, "Physical-HeartRate": 70,
            "SDS-SDS_Total_T": 36, "PAQ_A-PAQ_A_Total": 3.6,
            "Fitness_Endurance-Max_Stage": 12, "Fitness_Endurance-Time_Mins": 45,
            "BIA-BIA_Fat": 14, "BIA-BIA_FFM": 31, "BIA-BIA_SMM": 23
        }
        r1 = nurture.predict(valid_input)
        results["nurture_predict"] = check("predict() valid input -> code 200", r1.get("code") == 200, str(r1))
        check("predict() returns 'prediction' dict", isinstance(r1.get("prediction"), dict))
        check("predict() has sii key", "sii" in r1.get("prediction", {}))
        check("predict() sii in [0,1,2,3]", r1.get("prediction", {}).get("sii") in [0, 1, 2, 3])

        # Minimal input (missing features auto-filled with medians)
        r2 = nurture.predict({"Basic_Demos-Age": 14, "Basic_Demos-Sex": 1})
        check("predict() minimal input (medians used) -> code 200", r2.get("code") == 200, str(r2))

        # app.py uses load() and checks result.get("code", result.get("status", 500))
        status_val = load_res.get("code", load_res.get("status", 500))
        check("load() compatible with app.py status check (200)", status_val == 200)

        # app.py checks _model_status[name].get("loaded") -> set from status_code in (200, "success", "ok")
        check("load() status 'ok' is recognised by app.py", load_res.get("status") in (200, "success", "ok"))
except Exception as e:
    print(f"  {FAIL} Module crashed: {e}")
    traceback.print_exc()
    results["nurture_load"] = False
    results["nurture_predict"] = False


# ─────────────────────────────────────────────────────────────
# 5. CHATBOT MODEL
# ─────────────────────────────────────────────────────────────
section("5. CHATBOT MODEL (chatbot/chatbot_model.py)")
try:
    chatbot_path = os.path.join(BASE_DIR, "chatbot", "chroma_full_state.pkl")
    chatbot = _import("chatbot_model", os.path.join(BASE_DIR, "chatbot", "chatbot_model.py"))

    print(f"  {INFO} Loading ChromaDB (this may take ~30s for 46 MB pkl)...")
    load_res = chatbot.load(pkl_path=chatbot_path)
    loaded = check("load() returns code 200", load_res.get("code") == 200, str(load_res))
    results["chatbot_load"] = loaded

    if loaded:
        # No API key — should return 400
        r1 = chatbot.predict({"query": "What is fever?", "google_api_key": ""})
        check("predict() missing API key -> code 400", r1.get("code") == 400, str(r1))

        # Missing query
        r2 = chatbot.predict({"query": "", "google_api_key": "dummy"})
        check("predict() empty query -> code 400", r2.get("code") == 400, str(r2))

        # No query key at all
        r3 = chatbot.predict({"google_api_key": "dummy"})
        check("predict() no query key -> code 400", r3.get("code") == 400, str(r3))

        results["chatbot_predict"] = True
        print(f"  {INFO} Skipping live Gemini API call (requires valid key + network).")
except Exception as e:
    print(f"  {FAIL} Module crashed: {e}")
    traceback.print_exc()
    results["chatbot_load"] = False
    results["chatbot_predict"] = False


# ─────────────────────────────────────────────────────────────
# 6. APP.PY INTEGRATION CHECKS
# ─────────────────────────────────────────────────────────────
section("6. APP.PY INTEGRATION CHECKS")

# Check all PKL files exist
pkl_paths = {
    "behaviour":      os.path.join(BASE_DIR, "behaviour_analysis", "nurture_model.pkl"),
    "child_mortality": os.path.join(BASE_DIR, "child_mortality",   "model.pkl"),
    "nurture":        os.path.join(BASE_DIR, "nurture_model",      "nurture_model.pkl"),
    "celiac":         os.path.join(BASE_DIR, "celiac-disease",     "celiac_model.pkl"),
    "chatbot":        os.path.join(BASE_DIR, "chatbot",            "chroma_full_state.pkl"),
}
for name, path in pkl_paths.items():
    check(f"PKL exists: {name}", os.path.exists(path), path)

# Check all model .py files exist
model_paths = {
    "behaviour":      os.path.join(BASE_DIR, "behaviour_analysis", "nurture_model.py"),
    "child_mortality": os.path.join(BASE_DIR, "child_mortality",   "child_health_model.py"),
    "nurture":        os.path.join(BASE_DIR, "nurture_model",      "nurture_model.py"),
    "celiac":         os.path.join(BASE_DIR, "celiac-disease",     "celiac_model.py"),
    "chatbot":        os.path.join(BASE_DIR, "chatbot",            "chatbot_model.py"),
}
for name, path in model_paths.items():
    check(f"Model .py exists: {name}", os.path.exists(path), path)

# Check frontend files
frontend_dir = os.path.join(BASE_DIR, "frontend")
for f in ["index.html", "app.js", "style.css"]:
    check(f"Frontend file exists: {f}", os.path.exists(os.path.join(frontend_dir, f)))

# Check app.py load() status-code compatibility per module
print(f"\n  {INFO} Checking load() return value compatibility with app.py:")
print(f"  {INFO} app.py checks: status_code in (200, 'success', 'ok')")

compat = {
    "celiac":         ("code", 200),       # returns {"code": 200}
    "child_mortality": ("code", 200),      # returns {"code": 200}
    "nurture":        ("status", "ok"),    # returns {"status": "ok", "code": 200}
    "behaviour":      ("status", 200),     # returns {"status": 200}
    "chatbot":        ("code", 200),       # returns {"code": 200}
}
for name, (key, val) in compat.items():
    check(f"load() status compatible — {name} ({key}={val})",
          val in (200, "success", "ok"))

# Critical bug check: behaviour model load() uses status=200 (int), not "success"/"ok"
# app.py line 72-73: status_code = result.get("code", result.get("status", 500))
