import sys, os, pickle, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
BASE = r'c:\Users\apoor\Projects\Parenting'

with open(os.path.join(BASE,'nurture_model','nurture_model.pkl'),'rb') as f:
    bundle = pickle.load(f)
m = bundle['model']
feat_names = bundle['feature_names']
raw_medians = bundle['raw_medians']

def make_pred(sds, paq, bmi, stage, age=13, sex=0):
    d = {
        'Basic_Demos-Age': age, 'Basic_Demos-Sex': sex,
        'Physical-BMI': float(bmi), 'Physical-Height': 152.0, 'Physical-Weight': 50.0,
        'Physical-Waist_Circumference': 68.0,
        'Physical-Diastolic_BP': 68.0, 'Physical-Systolic_BP': 108.0,
        'Physical-HeartRate': 74.0,
        'SDS-SDS_Total_T': float(sds),
        'PAQ_A-PAQ_A_Total': float(paq), 'PAQ_C-PAQ_C_Total': float(paq),
        'Fitness_Endurance-Max_Stage': float(stage), 'Fitness_Endurance-Time_Mins': float(stage*3.5),
        'BIA-BIA_Fat': 17.0, 'BIA-BIA_FFM': 36.0, 'BIA-BIA_SMM': 26.0
    }
    bmi_val = d['Physical-BMI']
    bmi_cat = 0 if bmi_val<18.5 else (1 if bmi_val<25 else (2 if bmi_val<30 else 3))
    age_v = d['Basic_Demos-Age']
