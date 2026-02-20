=====================================================================
  NurtureAI — Child Behaviour & Mental Health Risk Predictor
  Submission Files
=====================================================================

FILES IN THIS ZIP
─────────────────
  nurture_model.py      ← The Python script (load + predict functions)
  nurture_model.pkl     ← The trained Random Forest model (binary file)
  README.txt            ← This file

HOW TO USE
──────────
1. Put BOTH files (nurture_model.py and nurture_model.pkl)
   in the SAME folder.

2. In your frontend / integration script, import like this:

     from nurture_model import load, predict

3. Call load() ONCE at app startup:

     response = load()
     # Returns: {"status": "ok", "code": 200}

4. Call predict() with a child's data dictionary:

     result = predict({
         "Basic_Demos-Age": 12,
         "Basic_Demos-Sex": 0,
         "Physical-BMI": 20.0,
         "Physical-Height": 148,
         "Physical-Weight": 44,
         "Physical-Waist_Circumference": 65,
         "Physical-Diastolic_BP": 70,
         "Physical-Systolic_BP": 110,
         "Physical-HeartRate": 76,
         "SDS-SDS_Total_T": 48,
         "PAQ_A-PAQ_A_Total": 2.8,
         "Fitness_Endurance-Max_Stage": 8,
         "Fitness_Endurance-Time_Mins": 28,
         "BIA-BIA_Fat": 18,
         "BIA-BIA_FFM": 36,
         "BIA-BIA_SMM": 26
     })

5. The result looks like:
     {
       "status": "ok",
       "code": 200,
       "prediction": {
           "sii": 0,
           "risk_label": "None (Healthy)",
           "behavior_score": 62.4,
           "probabilities": {
               "sii=0 (None (Healthy))": 0.307,
               "sii=1 (Mild)": 0.238,
               "sii=2 (Moderate)": 0.249,
               "sii=3 (Severe)": 0.205
           }
       }
     }

OUTPUT MEANING
──────────────
  sii = 0  →  None (Healthy)
  sii = 1  →  Mild Risk
  sii = 2  →  Moderate Risk
  sii = 3  →  Severe Risk

  behavior_score: 0–100 composite lifestyle score (higher = healthier)

REQUIRED PYTHON PACKAGES
─────────────────────────
  pip install scikit-learn pandas numpy

QUICK TEST
──────────
  python nurture_model.py
  (Runs 3 demo predictions and prints results)

=====================================================================
