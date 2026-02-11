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
