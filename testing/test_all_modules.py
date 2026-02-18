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
