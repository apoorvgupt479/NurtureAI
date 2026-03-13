# ============================================================
# NurtureAI - Caregiving Environment Risk Prediction Model
# ============================================================
#
# PROJECT: NurtureAI Adult Behavioral Risk Analysis
# DATASET: BRFSS 2013 (Behavioral Risk Factor Surveillance System)
# MODEL:   Random Forest Classifier (scikit-learn)
# PICKLE:  nurture_model.pkl
#
# ============================================================
# INPUT FEATURES (6 required)
# ============================================================
#
# All features must be provided as a dictionary. Feature order
# is handled internally — just pass the correct keys.
#
#   Feature             | Type  | Valid Values           | Description
#   --------------------|-------|------------------------|------------------------------------------
#   General_Health      | int   | 1, 2, 3, 4, 5         | Self-rated health perception
#                       |       |                        |   1 = Excellent
#                       |       |                        |   2 = Very Good
#                       |       |                        |   3 = Good
#                       |       |                        |   4 = Fair
#                       |       |                        |   5 = Poor
#   --------------------|-------|------------------------|------------------------------------------
#   Sleep_Hours         | int   | 1 to 18                | Average hours of sleep per night
#                       |       |                        |   Optimal range: 7–9 hours
#   --------------------|-------|------------------------|------------------------------------------
#   Exercise_Any        | int   | 1 or 2                 | Any physical activity in past 30 days
#                       |       |                        |   1 = Yes
#                       |       |                        |   2 = No
#   --------------------|-------|------------------------|------------------------------------------
#   Smoked_100_Cigs     | int   | 1 or 2                 | Has the person smoked 100+ cigarettes
#                       |       |                        | in their lifetime?
#                       |       |                        |   1 = Yes
#                       |       |                        |   2 = No
#   --------------------|-------|------------------------|------------------------------------------
#   Income_Level        | int   | 1, 2, 3, 4, 5, 6, 7, 8| Household income bracket
#                       |       |                        |   1 = Less than $10,000
#                       |       |                        |   2 = Less than $15,000
#                       |       |                        |   3 = Less than $20,000
#                       |       |                        |   4 = Less than $25,000
#                       |       |                        |   5 = Less than $35,000
#                       |       |                        |   6 = Less than $50,000
#                       |       |                        |   7 = Less than $75,000
#                       |       |                        |   8 = $75,000 or more
#   --------------------|-------|------------------------|------------------------------------------
#   Marital_Status      | int   | 1, 2, 3, 4, 5, 6      | Current marital status
#                       |       |                        |   1 = Married
#                       |       |                        |   2 = Divorced
#                       |       |                        |   3 = Widowed
#                       |       |                        |   4 = Separated
#                       |       |                        |   5 = Never married
#                       |       |                        |   6 = A member of an unmarried couple
#
# ============================================================
# OPTIONAL EXTENDED FEATURES (for detailed risk scoring)
# ============================================================
#
#   Feature                | Type  | Valid Values  | Description
#   -----------------------|-------|---------------|------------------------------------------
#   Physical_Health_Days   | int   | 0 to 30       | Days of poor physical health (past 30 days)
#   Mental_Health_Days     | int   | 0 to 30       | Days of poor mental health (past 30 days)
#   Depression_Diagnosis   | int   | 1 or 2        | Ever told you have depressive disorder?
#                          |       |               |   1 = Yes, 2 = No
#   BMI_Indicator          | float | ~15.0 to 60.0 | Body Mass Index
#   Alcohol_Days_Monthly   | int   | 0 to 30       | Days of alcohol consumption per month
#
# ============================================================
# OUTPUT FORMAT
# ============================================================
#
# {
#     "status": 200,
#     "prediction": "Healthy Home Environment",
#     "confidence": {
#         "Healthy Home Environment": 0.82,
#         "Moderate Risk Environment": 0.13,
#         "High Risk Environment": 0.05
#     },
#     "risk_scores": {
#         "Mental_Stress_Score": 0.12,
#         "Lifestyle_Risk_Score": 0.08,
#         "Physical_Risk_Score": 0.10,
#         "Overall_Risk": 0.10
#     },
#     "recommendations": ["..."]
# }
#
# ============================================================

import pickle
import pandas as pd

# Global model bundle (loaded by load())
_model_bundle = None


def load(pkl_path="nurture_model.pkl"):
    """
    Load the trained NurtureAI model from a pickle file.

    Parameters:
        pkl_path (str): Path to the .pkl file. Default: 'nurture_model.pkl'

    Returns:
        dict: {"status": 200, "message": "..."} on success
              {"status": 500, "error": "..."} on failure
    """
    global _model_bundle
    try:
        with open(pkl_path, "rb") as f:
            _model_bundle = pickle.load(f)

        # Validate the bundle has all required keys
        required_keys = ["model", "clf_features", "class_labels", "cluster_profiles"]
        for key in required_keys:
            if key not in _model_bundle:
                return {"status": 500, "error": f"Corrupt pickle: missing key '{key}'"}

        return {
            "status": 200,
            "message": f"Model loaded successfully from '{pkl_path}'",
            "features_required": _model_bundle["clf_features"],
            "classes": _model_bundle["class_labels"],
        }

    except FileNotFoundError:
        return {"status": 404, "error": f"Model file '{pkl_path}' not found"}
    except Exception as e:
        return {"status": 500, "error": f"Failed to load model: {str(e)}"}


def predict(input_data):
    """
    Predict the caregiving environment risk label for the given input.

    Parameters:
        input_data (dict): Dictionary with feature values.
            Required keys: General_Health, Sleep_Hours, Exercise_Any,
                           Smoked_100_Cigs, Income_Level, Marital_Status
            Optional keys: Physical_Health_Days, Mental_Health_Days,
                           Depression_Diagnosis, BMI_Indicator,
                           Alcohol_Days_Monthly

    Returns:
        dict: Prediction result with status code, label, confidence,
              risk scores, cluster comparison, and recommendations.
    """
    global _model_bundle

    # --- Check model is loaded ---
    if _model_bundle is None:
        return {"status": 503, "error": "Model not loaded. Call load() first."}

    rf = _model_bundle["model"]
    clf_features = _model_bundle["clf_features"]
    cluster_profiles = _model_bundle["cluster_profiles"]

    # --- Validate required features ---
    required = ["General_Health", "Sleep_Hours", "Exercise_Any",
                "Smoked_100_Cigs", "Income_Level", "Marital_Status"]
    missing = [f for f in required if f not in input_data]
    if missing:
        return {"status": 400, "error": f"Missing required features: {missing}"}

    # --- Validate value ranges ---
    validations = {
        "General_Health":    (1, 5, int),
        "Sleep_Hours":       (1, 18, int),
        "Exercise_Any":      (1, 2, int),
        "Smoked_100_Cigs":   (1, 2, int),
        "Income_Level":      (1, 8, int),
        "Marital_Status":    (1, 6, int),
    }
    for feat, (lo, hi, dtype) in validations.items():
        val = input_data[feat]
        if not isinstance(val, (int, float)):
            return {"status": 400, "error": f"'{feat}' must be numeric, got {type(val).__name__}"}
        if val < lo or val > hi:
