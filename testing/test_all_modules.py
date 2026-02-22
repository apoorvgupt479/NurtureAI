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
