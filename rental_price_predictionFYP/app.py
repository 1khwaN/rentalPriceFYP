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
from flask import render_template_string

# ---------------- CONFIG ----------------
CSV_BEFORE = "cleaned_rental_dataset_before_encoding.csv"
CSV_MODEL  = "cleaned_rental_dataset_model_ready.csv"
MODEL_PKL  = "hybrid_rent_model_city.pkl"
MAPS_DIR   = "maps"
os.makedirs(MAPS_DIR, exist_ok=True)

print("Loading datasets and model...")

# ---------- load datasets ----------
df_raw = pd.read_csv(CSV_BEFORE)
df_model = pd.read_csv(CSV_MODEL)

print(f"Loaded cleaned dataset {df_raw.shape} and model-ready dataset {df_model.shape}")

# ---------- load model ----------
model_data = joblib.load(MODEL_PKL)

rf  = model_data["rf"]
xgb = model_data["xgb"]
model_columns = model_data["columns"]

template_row = pd.DataFrame([np.zeros(len(model_columns))], columns=model_columns)

# ---------- helper functions ----------
def get_city_price_per_sqft(region, city):
    region = str(region).strip().lower()
    city = str(city).strip().lower()

    df_city = df_raw[
        (df_raw["region"].astype(str).str.lower() == region) &
        (df_raw["city"].astype(str).str.lower() == city)
    ]

    if df_city.empty:
        return df_raw["monthly_rent"].mean() / (df_raw["size"].mean() + 1)

    return df_city["monthly_rent"].mean() / (df_city["size"].mean() + 1)


def build_user_features(d):
    region = d.get("region", "").strip().title()
    city = d.get("city", "").strip().title()
    prop_type = d.get("property_type", "").strip().title()
    furnished = d.get("furnished", "").strip().title()

    def safe_float(v):
        try: return float(v)
        except: return np.nan

    def safe_int(v):
        try: return int(float(v))
        except: return 0

    size = safe_float(d.get("size"))
    completion_year = safe_int(d.get("completion_year"))
    parking = safe_int(d.get("parking"))
    rooms = safe_float(d.get("rooms"))
    bathroom = safe_float(d.get("bathroom"))

    if pd.isna(size): size = df_raw["size"].median()
    if pd.isna(rooms): rooms = df_raw["rooms"].median()
    if pd.isna(bathroom): bathroom = df_raw["bathroom"].median()

    age_of_property = 2025 - completion_year
    rooms_per_bathroom = rooms / (bathroom + 1)
    price_per_sqft = get_city_price_per_sqft(region, city)

    row = template_row.copy()

    numeric = {
        "completion_year": completion_year,
        "rooms": rooms,
        "parking": parking,
        "bathroom": bathroom,
        "size": size,
        "price_per_sqft": price_per_sqft,
        "rooms_per_bathroom": rooms_per_bathroom,
        "age_of_property": age_of_property
    }

    for k, v in numeric.items():
        if k in row.columns:
            row.at[0, k] = v

    def activate(prefix, val):
        colname = f"{prefix}_{val}"
        if colname in row.columns:
            row.at[0, colname] = 1

    activate("region", region)
    activate("city", city)
    activate("property_type", prop_type)
    activate("furnished", furnished)

    return row


# ============================================================
# 🔥 POPUP SUPPORT — Load popup_property.html into Folium popup
# ============================================================
def build_property_popup(r):
    """Render popup_property.html into folium popup"""
    html = render_template(
        "popup_property.html",
        prop_name=r.get("prop_name", "Unknown"),
        rent=f"{r.get('monthly_rent', 0):,.2f}",
        size=r.get("size", "N/A"),
        rooms=r.get("rooms", "N/A"),
        bathroom=r.get("bathroom", "N/A"),
        parking=r.get("parking", "N/A"),
        property_type=r.get("property_type", "N/A"),
        furnished=r.get("furnished", "N/A"),
        facilities=r.get("facilities", "N/A")
    )
    return folium.Popup(html, max_width=350)


# ============================================================
# MAP GENERATION
# ============================================================
def generate_map_and_recs(region, city, predicted_rent):
    geolocator = Nominatim(user_agent="rental_app")
    try:
        city_loc = geolocator.geocode(f"{city}, {region}, Malaysia")
        if not city_loc:
            return None, []

        user_coords = (city_loc.latitude, city_loc.longitude)

        m = folium.Map(location=user_coords, zoom_start=13)
        marker_cluster = MarkerCluster().add_to(m)

        # Filter properties
        props = df_raw[df_raw["city"].astype(str).str.lower() == city.lower()].copy()

        min_rent, max_rent = predicted_rent * 0.9, predicted_rent * 1.1
        recs = props[(props["monthly_rent"] >= min_rent) & (props["monthly_rent"] <= max_rent)]

        if recs.empty:
            recs = props.nlargest(5, "monthly_rent")

        recommendations = []

        for _, r in recs.iterrows():
            prop_name = r.get("prop_name", "Unknown")
            rent = r.get("monthly_rent", 0)

            # Get coordinates
            try:
                loc = geolocator.geocode(f"{prop_name}, {city}, {region}, Malaysia")
                coords = (loc.latitude, loc.longitude) if loc else user_coords
            except:
                coords = user_coords

            popup = build_property_popup(r)

            folium.Marker(
                location=coords,
                popup=popup,
                icon=folium.Icon(color="green", icon="home")
            ).add_to(marker_cluster)

            recommendations.append({
                "name": prop_name,
                "rent": float(rent)
            })

        filename = f"recommendation_map_{city.replace(' ', '_')}.html"
        filepath = os.path.join(MAPS_DIR, filename)
        m.save(filepath)

        return filename, recommendations

    except Exception as e:
        print("MAP ERROR:", e)
        return None, []


# ============================================================
# FLASK ROUTES
# ============================================================
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
    user_input = request.form.to_dict()
    X_user = build_user_features(user_input)

    rf_raw = rf.predict(X_user)[0]
    xgb_raw = xgb.predict(X_user)[0]

    hybrid_log = (rf_raw + xgb_raw) / 2
    predicted_rent = float(np.expm1(hybrid_log))

    map_filename, recommendations = generate_map_and_recs(
        user_input["region"], user_input["city"], predicted_rent
    )

    return render_template(
        "result.html",
        rent=round(predicted_rent, 2),
        city=user_input["city"],
        region=user_input["region"],
        size=user_input["size"],
        map_path=map_filename,
        recommendations=recommendations
    )


@app.route("/maps/<path:filename>")
def serve_map(filename):
    return send_from_directory(MAPS_DIR, filename)


@app.route("/api/options")
def api_options():
    regions = sorted(df_raw["region"].dropna().unique().tolist())
    property_types = sorted(df_raw["property_type"].dropna().unique().tolist())
    furnished = sorted(df_raw["furnished"].dropna().unique().tolist())

    cities_by_region = {
        region: sorted(df_raw[df_raw["region"] == region]["city"].dropna().unique().tolist())
        for region in regions
    }

    return jsonify({
        "regions": regions,
        "property_types": property_types,
        "furnished": furnished,
        "cities_by_region": cities_by_region
    })


if __name__ == "__main__":
    print("Starting Flask app (debug mode)...")
    app.run(debug=True)
