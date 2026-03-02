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
    age_grp = 0 if age_v<=11 else (1 if age_v<=14 else (2 if age_v<=17 else 3))
    d['BMI_Category'] = float(bmi_cat)
    d['Age_Group'] = float(age_grp)
    d['Pulse_Pressure'] = float(d['Physical-Systolic_BP'] - d['Physical-Diastolic_BP'])
    d['Body_Comp_Index'] = float(d['BIA-BIA_Fat'] / (d['BIA-BIA_FFM'] + 1e-5))
    sds_s = sds; paq_s = paq
    sleep_s = max(0,(80-sds_s)/80)*35
    act_s = max(0,(paq_s-1)/3)*35
    fit_s = min(stage/15,1.0)*20
    bmi_dev = abs(bmi_val-21.5)
    bmi_s = max(0,(15-bmi_dev)/15)*10
    d['Behavior_Score'] = round(min(sleep_s+act_s+fit_s+bmi_s,100),1)
    X = pd.DataFrame([d]).reindex(columns=feat_names, fill_value=np.nan)
    for col in bundle['raw_feat_cols']:
        if col in X.columns and X[col].isna().any():
            X[col] = X[col].fillna(raw_medians.get(col, 0))
    X = X.fillna(0)
    pred = int(m.predict(X)[0])
    proba = m.predict_proba(X)[0]
    return pred, proba, d['Behavior_Score']

found1 = None; found2 = None
SDS_VALS  = [38, 42, 45, 48, 50, 52, 55, 58, 60, 62, 65]
PAQ_VALS  = [1.5, 1.8, 2.0, 2.3, 2.5, 2.8, 3.0, 3.2, 3.5]
BMI_VALS  = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
STG_VALS  = [3, 4, 5, 6, 7, 8, 9, 10, 11]
AGE_VALS  = [10, 12, 13, 14, 15, 16, 17]

for age in AGE_VALS:
    for sds in SDS_VALS:
        for paq in PAQ_VALS:
            for bmi in BMI_VALS:
                for stage in STG_VALS:
                    pred, proba, bscore = make_pred(sds, paq, bmi, stage, age=age)
                    if pred == 1 and found1 is None:
                        found1 = (age, sds, paq, bmi, stage, bscore, proba)
                        print(f"sii=1 FOUND: age={age} SDS={sds} PAQ={paq} BMI={bmi} stage={stage} bscore={bscore}")
                        print(f"  proba={[round(p,3) for p in proba]}")
                    if pred == 2 and found2 is None:
                        found2 = (age, sds, paq, bmi, stage, bscore, proba)
                        print(f"sii=2 FOUND: age={age} SDS={sds} PAQ={paq} BMI={bmi} stage={stage} bscore={bscore}")
                        print(f"  proba={[round(p,3) for p in proba]}")
                    if found1 and found2:
                        break
                if found1 and found2: break
            if found1 and found2: break
        if found1 and found2: break
    if found1 and found2: break

if not found1: print("sii=1 NOT found in sweep")
if not found2: print("sii=2 NOT found in sweep")
