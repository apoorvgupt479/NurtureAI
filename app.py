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
