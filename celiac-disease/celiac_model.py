"""
=============================================================================
CELIAC DISEASE PREDICTION MODEL - INPUT FEATURE DOCUMENTATION
=============================================================================

This file contains the core prediction logic for Celiac disease. 
Below are the detailed specifications for the input features required by the 
predict() function.

INPUT FEATURES REQUIRED:

1. Age
   - Meaning: The chronological age of the patient in years.
   - Data Type: int
   - Valid Range: 1 to 100

2. Gender
   - Meaning: The biological sex of the patient.
   - Data Type: string
   - Possible Values: "Male", "Female"

3. Diabetes
   - Meaning: Indicates if the patient has a history of diabetes.
   - Data Type: string
   - Possible Values: "Yes", "No"

4. Diabetes Type
   - Meaning: The specific classification of diabetes if present.
   - Data Type: string
   - Possible Values: "Type 1", "Type 2", "Unknown" (if no diabetes)

5. Diarrhoea
   - Meaning: Presence of chronic or persistent diarrhoea.
   - Data Type: string
   - Possible Values: "Yes", "No"

6. Abdominal
   - Meaning: Presence of abdominal pain or discomfort.
   - Data Type: string
   - Possible Values: "Yes", "No"

7. Short_Stature
   - Meaning: Clinical observation of significantly below-average height.
   - Data Type: string
   - Possible Values: "Yes", "No"

8. Sticky_Stool
   - Meaning: Presence of steatorrhea or unusually sticky/greasy stools.
   - Data Type: string
   - Possible Values: "Yes", "No"

9. Weight_loss
   - Meaning: Unexplained or significant reduction in body weight.
   - Data Type: string
   - Possible Values: "Yes", "No"

10. IgA
    - Meaning: Immunoglobulin A level (serological marker).
    - Data Type: float
    - Valid Range: 0.0 to 10.0

11. IgG
    - Meaning: Immunoglobulin G level (serological marker).
    - Data Type: float
    - Valid Range: 0.0 to 25.0

12. IgM
    - Meaning: Immunoglobulin M level (serological marker).
    - Data Type: float
    - Valid Range: 0.0 to 5.0

=============================================================================
"""

import pickle
import numpy as np

# Global variables
model = None

# Encoding mappings (IMPORTANT: must match training)
encoders = {
    "Gender": {"Male": 1, "Female": 0},
    "Diabetes": {"No": 0, "Yes": 1},
    "Diabetes Type": {"Unknown": 0, "Type 1": 1, "Type 2": 2},
    "Diarrhoea": {"No": 0, "Yes": 1},
    "Abdominal": {"No": 0, "Yes": 1},
    "Short_Stature": {"No": 0, "Yes": 1},
    "Sticky_Stool": {"No": 0, "Yes": 1},
    "Weight_loss": {"No": 0, "Yes": 1}
}

# -----------------------------------
# LOAD FUNCTION
# -----------------------------------
def load():
    global model
    try:
        import os
        pkl_path = os.path.join(os.path.dirname(__file__), "celiac_model.pkl")
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)

        return {
            "status": "success",
            "code": 200,
            "message": "Model loaded successfully"
        }

    except Exception as e:
        return {
            "status": "error",
            "code": 500,
            "message": str(e)
        }


# -----------------------------------
# PREDICT FUNCTION
# -----------------------------------
def predict(input_data):
    global model

    try:
        if model is None:
            return {
                "status": "error",
                "code": 500,
                "message": "Model not loaded. Call load() first."
            }

        # Convert input dict → feature list (ORDER MATTERS)
        features = [
            input_data["Age"],
            encoders["Gender"][input_data["Gender"]],
            encoders["Diabetes"][input_data["Diabetes"]],
            encoders["Diabetes Type"][input_data["Diabetes Type"]],
            encoders["Diarrhoea"][input_data["Diarrhoea"]],
            encoders["Abdominal"][input_data["Abdominal"]],
            encoders["Short_Stature"][input_data["Short_Stature"]],
            encoders["Sticky_Stool"][input_data["Sticky_Stool"]],
            encoders["Weight_loss"][input_data["Weight_loss"]],
            input_data["IgA"],
            input_data["IgG"],
            input_data["IgM"]
        ]

        features = np.array(features).reshape(1, -1)

        prediction = model.predict(features)[0]

        return {
            "status": "success",
            "code": 200,
            "prediction": int(prediction)
        }

    except KeyError as e:
        return {
            "status": "error",
            "code": 400,
            "message": f"Missing or invalid input field: {str(e)}"
        }

    except Exception as e:
        return {
            "status": "error",
            "code": 500,
            "message": str(e)
        }