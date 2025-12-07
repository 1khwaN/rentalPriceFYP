from flask import Flask, render_template, request, jsonify, send_from_directory
import joblib
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
import folium
from folium.plugins import MarkerCluster
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# -------------------------
# FILENAMES - put these next to app.py
# -------------------------
RAW_CSV = "rental_kl_selangor.csv"
MODEL_PKL = "hybrid_rent_model_city.pkl"

# -------------------------
# Helpers
# -------------------------
def safe_read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None

# -------------------------
# Load and clean raw CSV (exact notebook cleaning)
# -------------------------
df_original = safe_read_csv(RAW_CSV)
if df_original is None:
    raise FileNotFoundError(f"Raw CSV not found at '{RAW_CSV}'. Place the original raw CSV in the same folder as app.py.")

def clean_raw_dataframe(df):
    df = df.copy()

    # fill defaults
    df['prop_name'] = df['prop_name'].fillna("Unknown")
    df['completion_year'] = df['completion_year'].fillna(df['completion_year'].median())
    df['parking'] = df['parking'].fillna(df['parking'].median())
    df['facilities'] = df['facilities'].fillna("None")
    df['additional_facilities'] = df['additional_facilities'].fillna("None")
    df['furnished'] = df['furnished'].fillna(df['furnished'].mode()[0] if not df['furnished'].mode().empty else "Unknown")
    df['rooms'] = df['rooms'].fillna(df['rooms'].mode()[0] if not df['rooms'].mode().empty else 0)
    df['bathroom'] = df['bathroom'].fillna(df['bathroom'].median())

    # clean monthly_rent -> numeric
    df['monthly_rent'] = df['monthly_rent'].astype(str).str.replace('[^0-9.]', '', regex=True)
    df['monthly_rent'] = pd.to_numeric(df['monthly_rent'], errors='coerce')
    df['monthly_rent'] = df['monthly_rent'].fillna(df['monthly_rent'].median())

    # clean size
    df['size'] = df['size'].astype(str)
    df['size'] = df['size'].str.replace('[^0-9.]', '', regex=True)
    df['size'] = df['size'].str.replace(r'\.+', '.', regex=True)
    df['size'] = df['size'].str.strip('.')
    df['size'] = pd.to_numeric(df['size'], errors='coerce')
    df['size'] = df['size'].fillna(df['size'].median())

    # rebuild region & city from location
    if 'location' in df.columns:
        tmp = df['location'].astype(str).str.split(' - ', expand=True, n=1)
        if tmp.shape[1] == 2:
            df[['region', 'city']] = tmp
        else:
            df['region'] = df['location'].astype(str)
            df['city'] = ""
    df['region'] = df['region'].astype(str).str.strip().str.title()
    df['city'] = df['city'].astype(str).str.strip().str.title()

    # property_type cleaning
    if 'property_type' in df.columns:
        df['property_type'] = df['property_type'].astype(str).str.strip().str.title()

    # rooms cleaning
    df['rooms'] = df['rooms'].astype(str)
    df['rooms'] = df['rooms'].str.replace(r'[^0-9+]', '', regex=True)
    df['rooms'] = df['rooms'].str.replace(r'\+', '', regex=True)
    df['rooms'] = pd.to_numeric(df['rooms'], errors='coerce')
    df['rooms'] = df['rooms'].fillna(df['rooms'].median())

    # engineered features
    df['price_per_sqft'] = df['monthly_rent'] / (df['size'] + 1)
    df['rooms_per_bathroom'] = df['rooms'] / (df['bathroom'] + 1)
    df['age_of_property'] = 2025 - df['completion_year']

    return df

df_original = clean_raw_dataframe(df_original)

# -------------------------
# Load model (must have been saved with 'columns': X.columns.tolist())
# -------------------------
if not os.path.exists(MODEL_PKL):
    raise FileNotFoundError(f"Model file not found at '{MODEL_PKL}'. Please place the .pkl file saved from notebook here.")

model_data = joblib.load(MODEL_PKL)
if not isinstance(model_data, dict) or 'rf' not in model_data or 'xgb' not in model_data:
    raise KeyError("Model pickle must be a dict containing keys 'rf' and 'xgb' (and ideally 'columns').")

rf = model_data['rf']
xgb = model_data['xgb']

# recover model columns
model_columns = None
if 'columns' in model_data:
    model_columns = model_data['columns']
else:
    # fallback: build from df_original (same encoding as notebook)
    df_tmp = df_original.copy()
    df_tmp = pd.get_dummies(df_tmp, columns=['region','city','property_type','furnished'], drop_first=True)
    Xtmp = df_tmp.drop(columns=['ads_id','prop_name','facilities','additional_facilities','monthly_rent','location'], errors='ignore')
    model_columns = Xtmp.columns.tolist()

model_columns = [str(c) for c in model_columns]

# -------------------------
# Preprocess single input (exact notebook transformations)
# -------------------------
def preprocess_user_input(d):
    region = str(d.get('region','')).strip().title()
    city = str(d.get('city','')).strip().title()
    property_type = str(d.get('property_type','')).strip().title()
    furnished = str(d.get('furnished','')).strip().title()

    def safe_float(v, fallback=np.nan):
        try:
            return float(v)
        except Exception:
            return fallback
    def safe_int(v, fallback=0):
        try:
            return int(float(v))
        except Exception:
            return fallback

    size = safe_float(d.get('size', np.nan))
    completion_year = safe_int(d.get('completion_year', np.nan))
    parking = safe_int(d.get('parking', 0))
    rooms = safe_float(d.get('rooms', np.nan))
    bathroom = safe_float(d.get('bathroom', np.nan))

    # fill numeric missing with medians from df_original
    if np.isnan(size):
        size = df_original['size'].median()
    if np.isnan(completion_year):
        completion_year = int(df_original['completion_year'].median())
    if np.isnan(rooms):
        rooms = df_original['rooms'].median()
    if np.isnan(bathroom):
        bathroom = df_original['bathroom'].median()

    rooms_per_bathroom = rooms / (bathroom + 1)
    age_of_property = 2025 - completion_year

    # compute city price_per_sqft from df_original
    df_city = df_original[df_original['city'].str.lower() == city.lower()]
    if not df_city.empty:
        price_per_sqft = df_city['monthly_rent'].mean() / (df_city['size'].mean() + 1)
    else:
        price_per_sqft = df_original['monthly_rent'].mean() / (df_original['size'].mean() + 1)

    row = {
        'size': size,
        'completion_year': completion_year,
        'parking': parking,
        'rooms': rooms,
        'bathroom': bathroom,
        'region': region,
        'city': city,
        'property_type': property_type,
        'furnished': furnished,
        'price_per_sqft': price_per_sqft,
        'rooms_per_bathroom': rooms_per_bathroom,
        'age_of_property': age_of_property
    }

    user_df = pd.DataFrame([row])

    # one-hot encode same way
    user_df = pd.get_dummies(user_df, columns=['region','city','property_type','furnished'], drop_first=True)

    # align columns
    user_df = user_df.reindex(columns=model_columns, fill_value=0)

    return user_df

# -------------------------
# Routes
# -------------------------
@app.route('/')
def dashboard():
    # pass example metrics or implement reading saved performance if available
    metrics = {'r2': {'rf':0.92,'xgb':0.94,'hybrid':0.95}}
    return render_template('dashboard.html', metrics=metrics)

@app.route('/predict_form')
def predict_form():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # read form inputs
    user_input = {
        'region': request.form.get('region',''),
        'city': request.form.get('city',''),
        'size': request.form.get('size',''),
        'completion_year': request.form.get('completion_year',''),
        'parking': request.form.get('parking','0'),
        'rooms': request.form.get('rooms',''),
        'bathroom': request.form.get('bathroom',''),
        'property_type': request.form.get('property_type',''),
        'furnished': request.form.get('furnished','')
    }

    user_X = preprocess_user_input(user_input)

    # --- DEBUG: print shapes, columns and values to console (remove later) ---
    print("DEBUG: len(model_columns) =", len(model_columns))
    print("DEBUG: model_columns[:30] =", model_columns[:30])
    print("DEBUG: user_X.shape =", user_X.shape)
    print("DEBUG: user_X.columns[:40] =", user_X.columns.tolist()[:40])
    print("DEBUG: user_X.values =", user_X.values.tolist())

    # predict (models were trained on log1p(y))
    rf_pred_log = rf.predict(user_X)
    xgb_pred_log = xgb.predict(user_X)
    print("DEBUG: rf_pred_log =", rf_pred_log, "xgb_pred_log =", xgb_pred_log)

    hybrid_log = (rf_pred_log + xgb_pred_log) / 2.0
    predicted_rent = float(np.expm1(hybrid_log)[0])

    # create map and recommendations
    map_file, recommendations = generate_map_and_recs(user_input['region'], user_input['city'], predicted_rent)

    return render_template('result.html',
                            rent=round(predicted_rent,2),
                            city=user_input['city'].title(),
                            region=user_input['region'].title(),
                            size=user_input['size'],
                            map_path=map_file,
                            recommendations=recommendations)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)
    required = ['region','city','size','completion_year','parking','rooms','bathroom','property_type','furnished']
    for r in required:
        if r not in data:
            return jsonify({'error': f"Missing field {r}"}), 400

    user_X = preprocess_user_input(data)

    rf_pred_log = rf.predict(user_X)
    xgb_pred_log = xgb.predict(user_X)
    hybrid_log = (rf_pred_log + xgb_pred_log) / 2.0
    rent = float(np.expm1(hybrid_log)[0])
    return jsonify({'predicted_rent': rent})

# -------------------------
# Map & recommendations
# -------------------------
def generate_map_and_recs(region, city, predicted_rent):
    try:
        geolocator = Nominatim(user_agent="rental_app")
        city_loc = geolocator.geocode(f"{city}, {region}, Malaysia")
        if not city_loc:
            return None, []
        user_coords = (city_loc.latitude, city_loc.longitude)

        m = folium.Map(location=user_coords, zoom_start=13)
        MarkerCluster().add_to(m)

        city_props = df_original[df_original['city'].str.lower() == city.lower()].copy()
        if city_props.empty:
            city_props = df_original[df_original['region'].str.lower() == region.lower()].copy()

        min_rent, max_rent = predicted_rent*0.9, predicted_rent*1.1
        recommended_props = city_props[(city_props['monthly_rent'] >= min_rent) & (city_props['monthly_rent'] <= max_rent)].copy()
        if recommended_props.empty:
            recommended_props = city_props.nlargest(5, 'monthly_rent')

        recommendations = []
        for _, row in recommended_props.head(10).iterrows():
            prop_name = row.get('prop_name','Unknown')
            rent = row.get('monthly_rent', 0)
            try:
                loc = geolocator.geocode(f"{prop_name}, {city}, {region}, Malaysia")
                coords = (loc.latitude, loc.longitude) if loc else user_coords
            except Exception:
                coords = user_coords

            folium.Marker(location=coords, popup=f"{prop_name} - RM {rent:,.2f}", icon=folium.Icon(color="green")).add_to(m)
            recommendations.append({'name': prop_name, 'rent': round(float(rent),2)})

        os.makedirs("maps", exist_ok=True)
        map_filename = f"recommendation_map_{city.replace(' ','_')}.html"
        map_path = map_filename
        m.save(os.path.join("maps", map_filename))
        return map_path, recommendations
    except Exception as e:
        print("MAP ERROR:", e)
        return None, []

# Serve map files from maps/ folder
@app.route('/maps/<path:filename>')
def serve_map(filename):
    return send_from_directory('maps', filename)

# -------------------------
# Run server
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
