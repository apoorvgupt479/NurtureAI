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
