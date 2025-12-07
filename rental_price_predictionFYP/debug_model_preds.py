# debug_model_preds.py
import joblib, pandas as pd, numpy as np
from app import preprocess_user_input  # uses your Flask preprocessing function

# choose an input (use the same values you tested in Flask)
user_input = {
    'region': 'Selangor',
    'city': 'Shah Alam',
    'size': 900,
    'completion_year': 2019,
    'parking': 2,
    'rooms': 3,
    'bathroom': 2,
    'property_type': 'Apartment',
    'furnished': 'Fully Furnished'
}

print("Building user_X using Flask preprocess_user_input(...)")
user_X = preprocess_user_input(user_input)
print("user_X.shape", user_X.shape)
print("sum of features (sanity):", user_X.values.sum())
print("some feature values:", user_X.iloc[0,:10].to_dict())

model_data = joblib.load("hybrid_rent_model_city.pkl")
rf = model_data['rf']
xgb = model_data['xgb']

rf_log = rf.predict(user_X)
xgb_log = xgb.predict(user_X)

print("RF (log) =", rf_log, "=> RM", np.expm1(rf_log)[0])
print("XGB(log) =", xgb_log, "=> RM", np.expm1(xgb_log)[0])
print("Hybrid average log:", (rf_log + xgb_log)/2.0, "=> RM", np.expm1((rf_log + xgb_log)/2.0)[0])
