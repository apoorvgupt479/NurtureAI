import child_health_model

print("Loading Model...")
print(child_health_model.load())

# Create CLEAN input (ONLY allowed features)
sample_input = {}

for col in child_health_model.FEATURE_ORDER:
    sample_input[col] = 0

# Set some valid values (ONLY from cols)
if "Res_Age" in sample_input:
    sample_input["Res_Age"] = 30

if "Edu_level" in sample_input:
    sample_input["Edu_level"] = 2

# Example encoded columns
if "ChildSex_Male" in sample_input:
    sample_input["ChildSex_Male"] = 1

if "State_Bihar" in sample_input:
    sample_input["State_Bihar"] = 1

print("Prediction Result:")
print(child_health_model.predict(sample_input))