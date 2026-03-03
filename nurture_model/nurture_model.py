# =============================================================================
#  NurtureAI — Child Behaviour & Mental Health Risk Predictor
#  Model: Random Forest Classifier
#  Dataset: Child Mind Institute — Problematic Internet Use (CMI-PIU)
#  Target: sii (Severity Impairment Index)
# =============================================================================
#
#  📋 INPUT FEATURES — what data you must provide to get a prediction
#  ─────────────────────────────────────────────────────────────────────────────
#  Feature Name                    | Type  | Valid Range / Values   | Meaning
#  ─────────────────────────────────────────────────────────────────────────────
#  Basic_Demos-Age                 | int   | 5 – 22                 | Child's age in years
#  Basic_Demos-Sex                 | int   | 0 or 1                 | 0 = Female, 1 = Male
#  Physical-BMI                    | float | 13.0 – 40.0            | Body Mass Index (weight/height²)
#  Physical-Height                 | float | 90.0 – 200.0 (cm)      | Height in centimetres
#  Physical-Weight                 | float | 15.0 – 120.0 (kg)      | Weight in kilograms
#  Physical-Waist_Circumference    | float | 40.0 – 120.0 (cm)      | Waist measurement in cm
#  Physical-Diastolic_BP           | float | 50.0 – 100.0 (mmHg)    | Lower blood pressure reading
#  Physical-Systolic_BP            | float | 80.0 – 160.0 (mmHg)    | Upper blood pressure reading
#  Physical-HeartRate              | float | 50.0 – 120.0 (bpm)     | Resting heart rate (beats/min)
#  SDS-SDS_Total_T                 | float | 20.0 – 80.0            | Sleep disturbance score (low=good)
#  Fitness_Endurance-Max_Stage     | float | 1.0 – 14.0             | Max stage reached in fitness test
#  Fitness_Endurance-Time_Mins     | float | 5.0 – 60.0 (minutes)   | Total endurance test time
#  PAQ_A-PAQ_A_Total               | float | 1.0 – 4.0              | Physical Activity score (for teens)
#  PAQ_C-PAQ_C_Total               | float | 1.0 – 4.0              | Physical Activity score (for kids)
#  BIA-BIA_Fat                     | float | 5.0 – 50.0 (%)         | Body fat percentage
#  BIA-BIA_FFM                     | float | 15.0 – 80.0 (kg)       | Fat-free mass in kg
#  BIA-BIA_SMM                     | float | 10.0 – 60.0 (kg)       | Skeletal muscle mass in kg
#  ─────────────────────────────────────────────────────────────────────────────
#
#  📤 OUTPUT — what the model returns
#  ─────────────────────────────────────────────────────────────────────────────
#  sii (Severity Impairment Index):
#    0 = None  (Healthy)
#    1 = Mild
#    2 = Moderate
#    3 = Severe
#
#  behavior_score: A 0–100 composite lifestyle score (higher = healthier)
#  ─────────────────────────────────────────────────────────────────────────────
#
#  📦 HOW TO USE:
#    from nurture_model import load, predict
#    response = load()                    # Step 1: Load the model
#    result   = predict(your_input_dict) # Step 2: Pass the child's data
# =============================================================================


import os          # Used to check if the .pkl file exists on disk
import pickle      # Used to load the saved model from the .pkl file
import numpy as np # For math operations (used in behavior score calculation)
import pandas as pd  # For creating a structured data table (DataFrame)

# ── Path to the saved model file ──────────────────────────────────────────────
# This assumes nurture_model.pkl is in the SAME folder as this script.
# If it's somewhere else, update this path.
MODEL_PATH = os.path.join(os.path.dirname(__file__), "nurture_model.pkl")

# ── A global variable to hold the loaded model in memory ─────────────────────
# We keep this here so we don't reload from disk every single time predict() runs.
_model_bundle = None   # None means "not loaded yet"


# =============================================================================
#  FUNCTION 1: load()
#  Purpose : Load the trained model from the .pkl file into memory.
#  Returns : {"status": "ok", "code": 200}  if successful
#            {"status": "error", "message": "...", "code": 500}  if something fails
# =============================================================================
def load():
    """
    Loads the model from the nurture_model.pkl file into memory.
    Call this ONCE before calling predict().
    """
    global _model_bundle   # We want to update the global variable, not create a new local one

    try:
        # Check if the .pkl file actually exists before trying to open it
        if not os.path.exists(MODEL_PATH):
            return {
                "status": "error",
                "message": f"Model file not found at: {MODEL_PATH}. "
                           "Make sure nurture_model.pkl is in the same folder as this script.",
                "code": 404
            }

        # Open the .pkl file in binary-read mode ("rb") and load it using pickle
        with open(MODEL_PATH, "rb") as f:
            _model_bundle = pickle.load(f)   # This gives us a dict with {"model", "feature_names", "imputer"}

        # Quick sanity check — make sure all expected keys are present inside the loaded file
        required_keys = {"model", "feature_names", "raw_medians", "raw_feat_cols"}
        if not required_keys.issubset(_model_bundle.keys()):
            missing = required_keys - _model_bundle.keys()
            return {
                "status": "error",
                "message": f"Corrupted .pkl file. Missing keys: {missing}",
                "code": 500
            }

        # If everything worked, return success
        return {"status": "ok", "code": 200}

    except Exception as e:
        # Catch any unexpected error (file corrupt, wrong format, etc.)
        return {"status": "error", "message": str(e), "code": 500}


# =============================================================================
#  HELPER FUNCTION: _compute_behavior_score()
#  Purpose : Calculate a 0–100 composite lifestyle score from raw input values.
#            This is NOT a model prediction — it's a human-readable health indicator.
#  Weights : 35% sleep + 35% activity + 20% fitness + 10% BMI deviation
# =============================================================================
def _compute_behavior_score(data: dict) -> float:
    """
    Calculates a lifestyle Behavior Score (0–100) from the input dictionary.
    Higher score = healthier lifestyle.
    """
    # ── Sleep component (35 points) ────────────────────────────────────────
    # SDS score: low means GOOD sleep. So we INVERT it (80 - sds) to make
    # high value = good sleep = more points.
    sds = data.get("SDS-SDS_Total_T", 50)           # default 50 if not provided
    sleep_score = max(0, (80 - sds) / 80) * 35      # ranges from 0 to 35

    # ── Physical activity component (35 points) ────────────────────────────
    # PAQ score: 1 = very inactive, 4 = very active. Scale to 0–35 range.
    paq = data.get("PAQ_A-PAQ_A_Total", data.get("PAQ_C-PAQ_C_Total", 2.5))
    activity_score = max(0, (paq - 1) / 3) * 35     # ranges from 0 to 35

    # ── Fitness endurance component (20 points) ────────────────────────────
    # Max stage reached: higher is better. We cap it at 15 stages max.
    fit = data.get("Fitness_Endurance-Max_Stage", 7)
    fitness_score = min(fit / 15, 1.0) * 20         # ranges from 0 to 20

    # ── BMI component (10 points) ──────────────────────────────────────────
    # Ideal BMI is around 21.5. The further away from 21.5, the fewer points.
    bmi = data.get("Physical-BMI", 22)
    bmi_deviation = abs(bmi - 21.5)                 # 0 = perfect, higher = worse
    bmi_score = max(0, (15 - bmi_deviation) / 15) * 10  # ranges from 0 to 10

    # ── Total score ────────────────────────────────────────────────────────
    total = sleep_score + activity_score + fitness_score + bmi_score
    return round(min(total, 100.0), 1)   # Cap at 100 just in case


# =============================================================================
#  FUNCTION 2: predict(input_data)
#  Purpose : Take a dictionary of child health features, run the ML model,
#            and return the predicted risk level + behavior score.
#  Input   : dict with feature names and values (see top of file for full list)
#  Returns : {"status": "ok", "code": 200, "prediction": {...}}  if successful
#            {"status": "error", "message": "...", "code": ...}   if something fails
# =============================================================================
def predict(input_data: dict) -> dict:
    """
    Makes a prediction for a child based on their health data.

    Parameters
    ----------
