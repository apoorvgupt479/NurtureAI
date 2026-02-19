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
