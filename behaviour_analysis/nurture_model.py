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
            return {"status": 400, "error": f"'{feat}' must be between {lo} and {hi}, got {val}"}

    try:
        # --- Random Forest Prediction ---
        rf_input = pd.DataFrame([{k: input_data[k] for k in clf_features}])
        predicted_label = rf.predict(rf_input)[0]
        predicted_proba = rf.predict_proba(rf_input)[0]
        class_labels = rf.classes_

        confidence = {lbl: round(float(prob), 4) for lbl, prob in zip(class_labels, predicted_proba)}

        # --- Compute Risk Scores (if extended features provided) ---
        risk_scores = None
        recommendations = []

        extended_keys = ["Physical_Health_Days", "Mental_Health_Days",
                         "Depression_Diagnosis", "BMI_Indicator", "Alcohol_Days_Monthly"]
        has_extended = all(k in input_data for k in extended_keys)

        if has_extended:
            inp = input_data

            # Mental Stress Score
            depression_flag = 1 if inp["Depression_Diagnosis"] == 1 else 0
            sleep_deviation = abs(inp["Sleep_Hours"] - 7.5)
            mental_days_norm = min(inp["Mental_Health_Days"] / 30.0, 1.0)
            sleep_dev_norm = min(sleep_deviation / 10.0, 1.0)
            gen_health_norm = (inp["General_Health"] - 1) / 4.0

            mental_stress = (
                mental_days_norm * 0.40 +
                depression_flag  * 0.25 +
                gen_health_norm  * 0.20 +
                sleep_dev_norm   * 0.15
            )

            # Lifestyle Risk Score
            no_exercise = 1 if inp["Exercise_Any"] == 2 else 0
            smoking = 1 if inp["Smoked_100_Cigs"] == 1 else 0
            sleep_quality_risk = min(abs(inp["Sleep_Hours"] - 7.5) / 4.5, 1.0)
            alcohol_risk = min(inp["Alcohol_Days_Monthly"] / 30.0, 1.0)
            bmi = inp["BMI_Indicator"]
            bmi_deviation = 0 if 18.5 <= bmi <= 25 else min(abs(bmi - 21.75) / 20, 1.0)

            lifestyle_risk = (
                no_exercise        * 0.25 +
                smoking            * 0.20 +
                sleep_quality_risk * 0.20 +
                alcohol_risk       * 0.20 +
                bmi_deviation      * 0.15
            )

            # Physical Risk Score
            phys_days_norm = min(inp["Physical_Health_Days"] / 30.0, 1.0)
            bmi_risk_cont = 0 if 18.5 <= bmi <= 25 else min(abs(bmi - 21.75) / 20, 1.0)

            physical_risk = (
                phys_days_norm  * 0.45 +
                gen_health_norm * 0.35 +
                bmi_risk_cont   * 0.20
            )

            overall = (mental_stress + lifestyle_risk + physical_risk) / 3

            risk_scores = {
                "Mental_Stress_Score":  round(mental_stress, 4),
                "Lifestyle_Risk_Score": round(lifestyle_risk, 4),
                "Physical_Risk_Score":  round(physical_risk, 4),
                "Overall_Risk":         round(overall, 4),
            }

            # --- Generate Recommendations ---
            if mental_stress > 0.5:
                recommendations.append("HIGH Mental Stress: Consider professional counseling. Practice daily mindfulness.")
                if depression_flag:
                    recommendations.append("Depression detected: Ensure treatment compliance. Consider parent support groups.")
            elif mental_stress > 0.25:
                recommendations.append("Moderate Mental Stress: Try journaling, regular breaks, and social connection.")

            if inp["Sleep_Hours"] < 6:
                recommendations.append("Critical Sleep Deficit: Aim for 7-8 hours. #1 predictor of parenting patience.")
            elif inp["Sleep_Hours"] < 7 or inp["Sleep_Hours"] > 9:
                recommendations.append("Sleep Optimization: Aim for 7-8 hours. Set a consistent bedtime routine.")

            if no_exercise:
                recommendations.append("No Exercise: Start with 15-min walks with your child. Active parents raise active kids.")
            if smoking:
                recommendations.append("Smoking History: Second-hand smoke harms children. Consider a cessation program.")
            if alcohol_risk > 0.5:
                recommendations.append("High Alcohol Consumption: Reducing intake improves sleep and emotional availability.")
            if bmi > 30:
                recommendations.append("BMI indicates obesity risk: Try family meals with more fruits/vegetables.")
            elif bmi > 25:
                recommendations.append("BMI slightly elevated: Small dietary changes and 30 min daily activity help.")
            if inp["Physical_Health_Days"] > 15:
                recommendations.append("Frequent Illness: Prioritize your health plan. A healthy caregiver is the foundation.")
            if inp["Income_Level"] <= 3:
                recommendations.append("Financial Stress: Focus on free activities (reading, outdoor play, household games).")

            if not recommendations:
                recommendations.append("You are in great shape as a caregiver. Keep up the excellent work!")

        # --- Build response ---
        result = {
            "status": 200,
            "prediction": predicted_label,
            "confidence": confidence,
        }

        if risk_scores:
            result["risk_scores"] = risk_scores
            result["cluster_comparison"] = cluster_profiles
            result["recommendations"] = recommendations

        return result

    except Exception as e:
        return {"status": 500, "error": f"Prediction failed: {str(e)}"}


# ============================================================
# Quick test (only runs when script is executed directly)
# ============================================================
if __name__ == "__main__":
    import json

    # Load model
    print("Loading model...")
    load_result = load()
    print(json.dumps(load_result, indent=2))

    if load_result["status"] != 200:
        exit(1)

    # Test prediction with all features
    print("\n" + "=" * 60)
    print("Test: Full prediction with extended features")
    print("=" * 60)

    test_input = {
        "General_Health": 3,
        "Sleep_Hours": 7,
        "Exercise_Any": 1,
        "Smoked_100_Cigs": 2,
        "Income_Level": 6,
        "Marital_Status": 1,
        "Physical_Health_Days": 2,
        "Mental_Health_Days": 3,
        "Depression_Diagnosis": 2,
        "BMI_Indicator": 24.5,
        "Alcohol_Days_Monthly": 4,
    }

    result = predict(test_input)
    print(json.dumps(result, indent=2))

    # Test with only required features
    print("\n" + "=" * 60)
    print("Test: Minimal prediction (6 required features only)")
    print("=" * 60)

    minimal_input = {
        "General_Health": 4,
        "Sleep_Hours": 5,
        "Exercise_Any": 2,
        "Smoked_100_Cigs": 1,
        "Income_Level": 2,
        "Marital_Status": 5,
    }

    result2 = predict(minimal_input)
    print(json.dumps(result2, indent=2))

    # Test error handling
    print("\n" + "=" * 60)
    print("Test: Missing features (should return 400)")
    print("=" * 60)

    bad_input = {"General_Health": 3, "Sleep_Hours": 7}
    result3 = predict(bad_input)
    print(json.dumps(result3, indent=2))
