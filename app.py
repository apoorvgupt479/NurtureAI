"""
NurtureAI — Flask Backend
Serves the frontend and exposes API endpoints for all prediction models.
"""

import os
import sys
import json
import importlib.util
import threading
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

# ---------------------------------------------------------------------------
# In-memory settings store
# ---------------------------------------------------------------------------
_settings = {"gemini_api_key": ""}
_api_key_path = os.path.join(BASE_DIR, "gemini_api_key.txt")
if os.path.exists(_api_key_path):
    with open(_api_key_path, "r", encoding="utf-8") as _f:
        _settings["gemini_api_key"] = _f.read().strip()

# ---------------------------------------------------------------------------
# Model Imports (using importlib to avoid name collisions)
# ---------------------------------------------------------------------------
def _import(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# We'll lazy-load these so the server starts fast
_modules = {}
_model_status = {}  # {name: {"loaded": bool, "message": str}}

MODEL_REGISTRY = {
    "behaviour":       os.path.join(BASE_DIR, "behaviour_analysis", "nurture_model.py"),
    "child_mortality":  os.path.join(BASE_DIR, "child_mortality", "child_health_model.py"),
    "nurture":          os.path.join(BASE_DIR, "nurture_model", "nurture_model.py"),
    "celiac":           os.path.join(BASE_DIR, "celiac-disease", "celiac_model.py"),
