# app.py (final, paste into your Flask project - replace current)
import os
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import joblib
from geopy.geocoders import Nominatim
import folium
from folium.plugins import MarkerCluster

# ---------------- CONFIG ----------------
CSV_BEFORE = "cleaned_rental_dataset_before_encoding.csv"
CSV_MODEL  = "cleaned_rental_dataset_model_ready.csv"
MODEL_PKL  = "hybrid_rent_model_city.pkl"
MAPS_DIR   = "maps"
os.makedirs(MAPS_DIR, exist_ok=True)

print("Loading datasets and model...")

# ---------- load datasets ----------
if not os.path.exists(CSV_BEFORE):
    raise FileNotFoundError(f"Missing file: {CSV_BEFORE}")
if not os.path.exists(CSV_MODEL):
    raise FileNotFoundError(f"Missing file: {CSV_MODEL}")
if not os.path.exists(MODEL_PKL):
    raise FileNotFoundError(f"Missing file: {MODEL_PKL}")

df_raw = pd.read_csv(CSV_BEFORE)     # before-encoding (has city, region, size, monthly_rent, prop_name)
df_model = pd.read_csv(CSV_MODEL)    # model-ready encoded dataset

print(f"Loaded raw dataset {df_raw.shape} and model-ready dataset {df_model.shape}")

#Check model file used by FLASK
print("MODEL_PKL path:", os.path.abspath(MODEL_PKL))
print("File last modified:", os.path.getmtime(MODEL_PKL))
# ---------- load model ----------
model_data = joblib.load(MODEL_PKL)


if not isinstance(model_data, dict) or 'rf' not in model_data or 'xgb' not in model_data:
    raise KeyError("Model pickle must be a dict containing 'rf' and 'xgb' (and ideally 'columns').")

rf = model_data['rf']
xgb = model_data['xgb']
model_columns = [str(c) for c in model_data.get('columns', df_model.columns.tolist())]
print(f"Loaded RF & XGB. Model expects {len(model_columns)} features.")

print("\n=== DEBUG: CHECK MODEL OBJECTS LOADED BY FLASK ===")
print("RF type:", type(rf))
print("XGB type:", type(xgb))
print("XGB params:", xgb.get_params())
print("XGB n_estimators:", xgb.get_params().get('n_estimators'))

# ---------- template-row ----------
template_row = pd.DataFrame([np.zeros(len(model_columns))], columns=model_columns)

# ---------- helper: city-based price per sqft ----------
def get_city_price_per_sqft(region, city):
    region = str(region).strip().lower()
    city = str(city).strip().lower()
    if 'city' not in df_raw.columns or 'monthly_rent' not in df_raw.columns or 'size' not in df_raw.columns:
        return df_raw['monthly_rent'].mean() / (df_raw['size'].mean() + 1)
    df_city = df_raw[(df_raw['region'].astype(str).str.lower()==region) & (df_raw['city'].astype(str).str.lower()==city)]
    if df_city.empty:
        return df_raw['monthly_rent'].mean() / (df_raw['size'].mean() + 1)
    return df_city['monthly_rent'].mean() / (df_city['size'].mean() + 1)

# ---------- build user features ----------
def build_user_features(d):
    region = str(d.get('region','')).strip().title()
    city = str(d.get('city','')).strip().title()
    prop_type = str(d.get('property_type','')).strip().title()
    furnished = str(d.get('furnished','')).strip().title()

    def safe_float(v, fallback=np.nan):
        try:
            return float(v)
        except:
            return fallback
    def safe_int(v, fallback=0):
        try:
            return int(float(v))
        except:
            return fallback

    size = safe_float(d.get('size', np.nan))
    completion_year = safe_int(d.get('completion_year', 2020))
    parking = safe_int(d.get('parking', 0))
    rooms = safe_float(d.get('rooms', np.nan))
    bathroom = safe_float(d.get('bathroom', np.nan))

    # fill numeric missing using df_raw medians if available
    if pd.isna(size) and 'size' in df_raw.columns:
        size = df_raw['size'].median()
    if pd.isna(rooms) and 'rooms' in df_raw.columns:
        rooms = df_raw['rooms'].median()
    if pd.isna(bathroom) and 'bathroom' in df_raw.columns:
        bathroom = df_raw['bathroom'].median()

    age_of_property = 2025 - completion_year
    rooms_per_bathroom = rooms / (bathroom + 1)
    price_per_sqft = get_city_price_per_sqft(region, city)

    row = template_row.copy()

    numeric_fill = {
        'completion_year': completion_year,
        'rooms': rooms,
        'parking': parking,
        'bathroom': bathroom,
        'size': size,
        'price_per_sqft': price_per_sqft,
        'rooms_per_bathroom': rooms_per_bathroom,
        'age_of_property': age_of_property
    }
    for k,v in numeric_fill.items():
        if k in row.columns:
            row.at[0,k] = v

    def activate(prefix, val):
        if val == "" or val is None:
            return
        colname = f"{prefix}_{val}"
        if colname in row.columns:
            row.at[0,colname] = 1

    activate("region", region)
    activate("city", city)
    activate("property_type", prop_type)
    activate("furnished", furnished)

    # DEBUG info
    print("\n==== MODEL INPUT DEBUG ====")
    print("Shape:", row.shape)
    print("Feature sum (sanity):", float(row.iloc[0].sum()))
    print("Sample values (first 30 cols):")
    print(row.iloc[0].iloc[:30].to_dict())
    return row

# ---------- diagnostics on a sample row ----------
def diagnostics_model_on_sample():
    sample = df_model.iloc[0:1].reindex(columns=model_columns, fill_value=0)
    try:
        rf_pred = rf.predict(sample)[0]
        xgb_pred = xgb.predict(sample)[0]
        print("\n==== MODEL DIAGNOSTICS (sample from model-ready CSV) ====")
        print("sample index = 0")
        print("RF raw output:", rf_pred, "-> expm1:", np.expm1(rf_pred))
        print("XGB raw output:", xgb_pred, "-> expm1:", np.expm1(xgb_pred))
        return rf_pred, xgb_pred
    except Exception as e:
        print("Diagnostics failed:", e)
        return None, None

diagnostics_model_on_sample()

# ---------- map generation ----------
def generate_map_and_recs(region, city, predicted_rent):
    try:
        geolocator = Nominatim(user_agent="rental_app")
        city_loc = geolocator.geocode(f"{city}, {region}, Malaysia")
        if not city_loc:
            return None, []
        user_coords = (city_loc.latitude, city_loc.longitude)

        m = folium.Map(location=user_coords, zoom_start=13)
        MarkerCluster().add_to(m)

        city_props = pd.DataFrame()
        if 'city' in df_raw.columns:
            city_props = df_raw[df_raw['city'].astype(str).str.lower() == city.lower()].copy()
        if city_props.empty and 'region' in df_raw.columns:
            city_props = df_raw[df_raw['region'].astype(str).str.lower() == region.lower()].copy()

        min_rent, max_rent = predicted_rent*0.9, predicted_rent*1.1
        recommended_props = city_props[(city_props['monthly_rent']>=min_rent) & (city_props['monthly_rent']<=max_rent)].copy() if not city_props.empty else pd.DataFrame()
        if recommended_props.empty and not city_props.empty:
            recommended_props = city_props.nlargest(5, 'monthly_rent')

        recommendations = []
        for _, r in (recommended_props.head(10).iterrows() if not recommended_props.empty else []):
            prop_name = r.get('prop_name','Unknown')
            rent = r.get('monthly_rent', 0)
            try:
                loc = Nominatim(user_agent="rental_app").geocode(f"{prop_name}, {city}, {region}, Malaysia")
                coords = (loc.latitude, loc.longitude) if loc else user_coords
            except Exception:
                coords = user_coords

            folium.Marker(location=coords, popup=f"{prop_name} - RM {rent:,.2f}", icon=folium.Icon(color="green")).add_to(m)
            recommendations.append({'name': prop_name, 'rent': round(float(rent),2)})

        map_filename = f"recommendation_map_{city.replace(' ','_')}.html"
        map_filepath = os.path.join(MAPS_DIR, map_filename)
        m.save(map_filepath)
        return map_filename, recommendations
    except Exception as e:
        print("MAP ERROR:", e)
        return None, []

# ---------- Flask app ----------
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("dashboard.html")

@app.route("/predict_form")
def predict_form():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/analytics")
def analytics():
    return render_template("analytics.html")

@app.route("/predict", methods=["POST"])
def predict():
    user_input = {
        'region': request.form.get('region','').strip(),
        'city': request.form.get('city','').strip(),
        'size': request.form.get('size',''),
        'completion_year': request.form.get('completion_year',''),
        'parking': request.form.get('parking','0'),
        'rooms': request.form.get('rooms',''),
        'bathroom': request.form.get('bathroom',''),
        'property_type': request.form.get('property_type',''),
        'furnished': request.form.get('furnished','')
    }

    X_user = build_user_features(user_input)

    rf_raw = rf.predict(X_user)[0]
    xgb_raw = xgb.predict(X_user)[0]
    print("\nPREDICTION RAWS => RF:", rf_raw, "XGB:", xgb_raw)

    # Assume both models are trained in log-space (not raw). Average logs and expm1.
    hybrid_log = (rf_raw + xgb_raw) / 2.0
    predicted_rent = float(np.expm1(hybrid_log))
    print("Computed hybrid_log:", hybrid_log, "predicted_rent:", predicted_rent)

    map_filename, recommendations = generate_map_and_recs(user_input['region'], user_input['city'], predicted_rent)

    return render_template("result.html",
                           rent=round(predicted_rent,2),
                           city=user_input['city'].title(),
                           region=user_input['region'].title(),
                           size=user_input['size'],
                           map_path=map_filename,
                           recommendations=recommendations)

@app.route("/maps/<path:filename>")
def serve_map(filename):
    return send_from_directory(MAPS_DIR, filename)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    X_user = build_user_features(data)
    rf_raw = rf.predict(X_user)[0]
    xgb_raw = xgb.predict(X_user)[0]
    hybrid_log = (rf_raw + xgb_raw) / 2.0
    rent = float(np.expm1(hybrid_log))
    return jsonify({'predicted_rent': rent})



if __name__ == "__main__":
    print("Starting Flask app (debug mode)...")
    diagnostics_model_on_sample()
    app.run(debug=True)



