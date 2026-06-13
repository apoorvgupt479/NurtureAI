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

# Try to load local .env file manually if it exists (0MB footprint fallback)
_env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(_env_path):
    with open(_env_path, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ[_k.strip()] = _v.strip().strip('"').strip("'")

# ---------------------------------------------------------------------------
# In-memory settings store
# ---------------------------------------------------------------------------
_settings = {
    "gemini_api_key": os.environ.get("GEMINI_API_KEY", os.environ.get("GOOGLE_API_KEY", ""))
}
if not _settings["gemini_api_key"]:
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
    "chatbot":          os.path.join(BASE_DIR, "chatbot", "chatbot_model.py"),
}

PKL_PATHS = {
    "behaviour":       os.path.join(BASE_DIR, "behaviour_analysis", "nurture_model.pkl"),
    "child_mortality":  os.path.join(BASE_DIR, "child_mortality", "model.pkl"),
    "nurture":          os.path.join(BASE_DIR, "nurture_model", "nurture_model.pkl"),
    "celiac":           os.path.join(BASE_DIR, "celiac-disease", "celiac_model.pkl"),
    "chatbot":          os.path.join(BASE_DIR, "chatbot", "chroma_documents.pkl"),
}


def _load_single_model(name):
    """Import and load a single model. Thread-safe."""
    try:
        if name not in _modules:
            _modules[name] = _import(name, MODEL_REGISTRY[name])

        mod = _modules[name]
        if name == "behaviour":
            result = mod.load(pkl_path=PKL_PATHS[name])
        elif name == "chatbot":
            result = mod.load(pkl_path=PKL_PATHS[name])
        else:
            result = mod.load()

        status_code = result.get("code", result.get("status", 500))
        if status_code in (200, "success", "ok"):
            _model_status[name] = {"loaded": True, "message": "OK"}
        else:
            _model_status[name] = {"loaded": False, "message": str(result)}
    except Exception as e:
        _model_status[name] = {"loaded": False, "message": str(e)}


def _load_all_models_background():
    """Load all models in background threads."""
    for name in MODEL_REGISTRY:
        t = threading.Thread(target=_load_single_model, args=(name,), daemon=True)
        t.start()


# ---------------------------------------------------------------------------
# Static File Serving
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(FRONTEND_DIR, path)


# ---------------------------------------------------------------------------
# API: Model Management
# ---------------------------------------------------------------------------
@app.route("/api/load-models", methods=["POST"])
def api_load_models():
    _load_all_models_background()
    return jsonify({"status": "loading", "message": "Models are loading in background"})

@app.route("/api/model-status", methods=["GET"])
def api_model_status():
    return jsonify(_model_status)


# ---------------------------------------------------------------------------
# API: Settings (Gemini API Key)
# ---------------------------------------------------------------------------
@app.route("/api/save-api-key", methods=["POST"])
def api_save_api_key():
    data = request.get_json()
    if not data or "api_key" not in data:
        return jsonify({"status": "error", "message": "No api_key provided"}), 400
    _settings["gemini_api_key"] = data["api_key"]
    return jsonify({"status": "ok", "message": "API key saved"})

@app.route("/api/get-api-key", methods=["GET"])
def api_get_api_key():
    key = _settings.get("gemini_api_key", "")
    # Mask the key for security — only show last 4 chars
    masked = ("•" * max(0, len(key) - 4)) + key[-4:] if len(key) > 4 else key
    return jsonify({"has_key": bool(key), "masked_key": masked})


# ---------------------------------------------------------------------------
# API: Parent Assessment (behaviour_analysis model)
# ---------------------------------------------------------------------------
@app.route("/api/parent-assessment", methods=["POST"])
def api_parent_assessment():
    data = request.get_json()
    if not data:
        return jsonify({"status": 400, "error": "No data provided"}), 400

    if "behaviour" not in _modules or not _model_status.get("behaviour", {}).get("loaded"):
        _load_single_model("behaviour")
        if not _model_status.get("behaviour", {}).get("loaded"):
            return jsonify({"status": 503, "error": "Behaviour model not available"}), 503

    result = _modules["behaviour"].predict(data)
    return jsonify(result)


# ---------------------------------------------------------------------------
# API: Child Health — Infant (< 1 year, child_mortality model)
# ---------------------------------------------------------------------------
@app.route("/api/child-infant", methods=["POST"])
def api_child_infant():
    data = request.get_json()
    if not data:
        return jsonify({"status": 400, "error": "No data provided"}), 400

    if "child_mortality" not in _modules or not _model_status.get("child_mortality", {}).get("loaded"):
        _load_single_model("child_mortality")
        if not _model_status.get("child_mortality", {}).get("loaded"):
            return jsonify({"status": 503, "error": "Child mortality model not available"}), 503

    result = _modules["child_mortality"].predict(data)
    return jsonify(result)


# ---------------------------------------------------------------------------
# API: Child Health — Older child (>= 1 year, nurture model)
# ---------------------------------------------------------------------------
@app.route("/api/child-health", methods=["POST"])
def api_child_health():
    data = request.get_json()
    if not data:
        return jsonify({"status": 400, "error": "No data provided"}), 400

    if "nurture" not in _modules or not _model_status.get("nurture", {}).get("loaded"):
        _load_single_model("nurture")
        if not _model_status.get("nurture", {}).get("loaded"):
            return jsonify({"status": 503, "error": "Nurture model not available"}), 503

    result = _modules["nurture"].predict(data)
    return jsonify(result)


# ---------------------------------------------------------------------------
# API: Celiac Disease Check
# ---------------------------------------------------------------------------
@app.route("/api/celiac-check", methods=["POST"])
def api_celiac():
    data = request.get_json()
    if not data:
        return jsonify({"status": 400, "error": "No data provided"}), 400

    if "celiac" not in _modules or not _model_status.get("celiac", {}).get("loaded"):
        _load_single_model("celiac")
        if not _model_status.get("celiac", {}).get("loaded"):
            return jsonify({"status": 503, "error": "Celiac model not available"}), 503

    result = _modules["celiac"].predict(data)
    return jsonify(result)


# ---------------------------------------------------------------------------
# API: Chatbot
# ---------------------------------------------------------------------------
@app.route("/api/chatbot", methods=["POST"])
def api_chatbot():
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "code": 400, "message": "No data provided"}), 400

    if "chatbot" not in _modules or not _model_status.get("chatbot", {}).get("loaded"):
        _load_single_model("chatbot")
        if not _model_status.get("chatbot", {}).get("loaded"):
            return jsonify({
                "status": "error",
                "code": 503,
                "message": "Chatbot model not available. It may still be loading — please try again in a moment."
            }), 503

    # Inject the server-side API key if client didn't provide one
    if not data.get("google_api_key"):
        data["google_api_key"] = _settings.get("gemini_api_key", "")

    result = _modules["chatbot"].predict(data)
    return jsonify(result)


# ---------------------------------------------------------------------------
# Pre-load Models
# ---------------------------------------------------------------------------
# Start loading all models in the background immediately when the module loads
_load_all_models_background()

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n[NurtureAI] Server starting...")
    print(f"   Frontend: {FRONTEND_DIR}")
    print(f"   Models:   {list(MODEL_REGISTRY.keys())}")
    print(f"   URL:      http://localhost:5000 (and http://<your-local-ip>:5000)\n")
    app.run(host="0.0.0.0", debug=True, port=5000)
