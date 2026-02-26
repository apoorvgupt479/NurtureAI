import celiac_model
import json

def run_tests():
    # Load the model
    print("--- Loading Model ---")
    load_result = celiac_model.load()
    print(json.dumps(load_result, indent=2))
    
    if load_result["status"] != "success":
        print("Failed to load model. Exiting.")
        return

    # Define test cases
    test_cases = [
        {
            "name": "Case 1: Typical Positive Scenario",
            "data": {
                "Age": 25,
                "Gender": "Female",
                "Diabetes": "Yes",
                "Diabetes Type": "Type 1",
                "Diarrhoea": "Yes",
                "Abdominal": "Yes",
                "Short_Stature": "No",
                "Sticky_Stool": "Yes",
                "Weight_loss": "Yes",
                "IgA": 5.2,
                "IgG": 15.1,
                "IgM": 1.8
            }
        },
        {
            "name": "Case 2: Typical Negative Scenario",
            "data": {
                "Age": 45,
                "Gender": "Male",
                "Diabetes": "No",
                "Diabetes Type": "Unknown",
                "Diarrhoea": "No",
                "Abdominal": "No",
                "Short_Stature": "No",
                "Sticky_Stool": "No",
                "Weight_loss": "No",
                "IgA": 1.2,
                "IgG": 7.5,
                "IgM": 0.8
            }
        }
    ]

    print("\n--- Running Predictions ---")
    for case in test_cases:
        print(f"\nTesting: {case['name']}")
        result = celiac_model.predict(case["data"])
        
        if result["status"] == "success":
            diagnosis = "Positive" if result["prediction"] == 1 else "Negative"
            print(f"Result: {diagnosis} (Code: {result['prediction']})")
        else:
            print(f"Error: {result['message']}")

if __name__ == "__main__":
    run_tests()