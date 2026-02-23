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
