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
