"""
=============================================================================
CELIAC DISEASE PREDICTION SYSTEM - INPUT FEATURE DOCUMENTATION
=============================================================================

This application provides a user interface for predicting Celiac disease.
The model requires the following clinical features for accurate prediction:

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
import streamlit as st
import celiac_model
import pandas as pd
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="Celiac Predictive AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Premium Look ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 3em;
        background-color: #4A90E2;
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #357ABD;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .prediction-card {
        padding: 2rem;
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        border: 1px solid rgba(255, 255, 255, 0.18);
        text-align: center;
        margin-top: 2rem;
    }
    
    .positive { color: #D0021B; font-weight: 700; font-size: 2rem; }
    .negative { color: #417505; font-weight: 700; font-size: 2rem; }
    
    .sidebar .sidebar-content {
        background-color: #2c3e50;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# --- Initialize Model ---
@st.cache_resource
def load_celiac_model():
    res = celiac_model.load()
    return res

load_status = load_celiac_model()

# --- Header ---
st.title("🔬 Celiac Disease Prediction System")
st.markdown("Enter patient clinical data below to analyze the likelihood of Celiac disease using our AI model.")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("Patient Data")
    
    age = st.number_input("Age", min_value=1, max_value=100, value=25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    
    st.divider()
    
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    diabetes_type = st.selectbox("Diabetes Type", ["Unknown", "Type 1", "Type 2"])
    
    st.divider()
    
    diarrhoea = st.radio("Diarrhoea", ["No", "Yes"], horizontal=True)
    abdominal = st.radio("Abdominal Pain", ["No", "Yes"], horizontal=True)
    short_stature = st.radio("Short Stature", ["No", "Yes"], horizontal=True)
    sticky_stool = st.radio("Sticky Stool", ["No", "Yes"], horizontal=True)
    weight_loss = st.radio("Unexpected Weight Loss", ["No", "Yes"], horizontal=True)

    st.divider()
    
    st.markdown("### Serological Markers")
    iga = st.slider("IgA Levels", 0.0, 10.0, 2.5)
    igg = st.slider("IgG Levels", 0.0, 25.0, 12.0)
    igm = st.slider("IgM Levels", 0.0, 5.0, 1.5)

# --- Prediction Logic ---
if st.button("🚀 Analyze Patient Profile"):
    input_data = {
        "Age": age,
        "Gender": gender,
        "Diabetes": diabetes,
        "Diabetes Type": diabetes_type,
        "Diarrhoea": diarrhoea,
        "Abdominal": abdominal,
        "Short_Stature": short_stature,
        "Sticky_Stool": sticky_stool,
        "Weight_loss": weight_loss,
        "IgA": iga,
        "IgG": igg,
        "IgM": igm
    }
    
    with st.spinner("Processing clinical markers..."):
        result = celiac_model.predict(input_data)
        
    if result["status"] == "success":
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        if result["prediction"] == 1:
            st.markdown('<p class="positive">⚠️ POSITIVE INDICATION</p>', unsafe_allow_html=True)
            st.warning("The model suggests a high probability of Celiac disease. Further clinical validation (e.g., biopsy) is strongly recommended.")
        else:
