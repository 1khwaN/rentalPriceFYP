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

# ---------------- MODEL ERROR STATS ----------------
# Estimate error from training data (stacked hybrid safe)
try:
# Ground truth
    y_true = df_model["monthly_rent"]

    # 🔥 FORCE FEATURE ALIGNMENT
    X_full = df_model[model_columns].copy()

    # Base model prediction
    rf_preds = rf.predict(X_full)

    # Stacked input
    X_full_h = X_full.copy()
    X_full_h["rf_pred"] = rf_preds

    # Meta model prediction
    hybrid_preds = xgb.predict(X_full_h)
    hybrid_preds = np.expm1(hybrid_preds)

    abs_errors = np.abs(hybrid_preds - y_true)

    ERROR_MEDIAN = np.median(abs_errors)
    ERROR_75 = np.percentile(abs_errors, 75)
    ERROR_90 = np.percentile(abs_errors, 90)

except Exception as e:
    print("⚠️ Error stats fallback:", e)
    ERROR_MEDIAN = 100
    ERROR_75 = 200
    ERROR_90 = 300

# ---------------- GEO CACHE ----------------
city_cache = {}

# ---------- helper functions ----------
from geopy.extra.rate_limiter import RateLimiter
from geopy.distance import geodesic

def get_prediction_confidence(predicted_rent, rf_log, hybrid_log, region, city):
    """
    Dynamic confidence estimation based on:
    - historical MAE
    - model disagreement
    - price deviation from city average
    """

    # --- Safety check ---
    if predicted_rent <= 0:
        return {
            "confidence": "N/A",
            "error_range": 0,
            "color": "secondary",
            "percent": 0
        }

    # --- City statistics ---
    df_city = df_raw[
        (df_raw["region"].str.lower() == region.lower()) &
        (df_raw["city"].str.lower() == city.lower())
    ]

    city_median = df_city["monthly_rent"].median() if not df_city.empty else df_raw["monthly_rent"].median()

    # --- Model disagreement (log-space → RM) ---
    rf_rm = np.expm1(rf_log)
    hybrid_rm = np.expm1(hybrid_log)
    model_gap = abs(rf_rm - hybrid_rm)

    # --- Error components ---
    base_error = ERROR_MEDIAN
    price_risk = abs(predicted_rent - city_median) / city_median
    disagreement_penalty = model_gap / predicted_rent

    # --- Final dynamic error ---
    dynamic_error = base_error * (1 + price_risk + disagreement_penalty)

    # --- Confidence percentage ---
    confidence_pct = 100 - (dynamic_error / predicted_rent * 100)
    confidence_pct = max(60, min(99, confidence_pct))

    # --- Color ---
    if confidence_pct >= 85:
        color = "success"
    elif confidence_pct >= 70:
        color = "warning"
    else:
        color = "danger"

    return {
    "confidence": f"{confidence_pct:.1f}%",
    "percent": round(confidence_pct, 1),
    "error_range": round(dynamic_error, 2),
    "color": color,
    "explanation": (
        f"Confidence is estimated using historical model error (MAE), "
        f"agreement between Random Forest and XGBoost models, and how close "
        f"the predicted rent is to typical rental prices in {city}. "
        f"Higher agreement and typical prices result in higher confidence."
    )
}




def get_city_coords(region, city):
    key = f"{city}_{region}".lower()

    if key in city_cache:
        return city_cache[key]

    try:
        geolocator = Nominatim(user_agent="rental_app")
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        loc = geocode(f"{city}, {region}, Malaysia", timeout=3)

        if loc:
            city_cache[key] = (loc.latitude, loc.longitude)
            return city_cache[key]
    except Exception as e:
        print("Geocode error:", e)

    return None

def detect_rail_proximity(text):
    if not isinstance(text, str):
        return False

    keywords = ["mrt", "lrt", "ktm", "monorail", "rail"]
    text = text.lower()

    return any(k in text for k in keywords)

def add_rail_stations(map_obj, region, city):
    geolocator = Nominatim(user_agent="rental_app")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

    stations = [
        f"MRT station {city}, {region}, Malaysia",
        f"LRT station {city}, {region}, Malaysia",
        f"KTM station {city}, {region}, Malaysia",
        f"{city} MRT Station, Malaysia",
        f"{city} LRT Station, Malaysia",
        f"{city} KTM Station, Malaysia"
    ]

    for s in stations:
        try:
            loc = geocode(s, timeout=3)
            if loc:
                folium.Marker(
                    location=(loc.latitude, loc.longitude),
                    tooltip=s.split(" station")[0],
                    icon=folium.Icon(
                        color="red",
                        icon="train",
                        prefix="fa"
                    )
                ).add_to(map_obj)
        except:
            continue

def is_near_station(prop_coords, station_coords, threshold_km=2.0):
    return geodesic(prop_coords, station_coords).km <= threshold_km

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
def generate_map_and_recs(region, city, predicted_rent, size, property_type):
    coords = get_city_coords(region, city)
    if not coords:
        return None, [], "⚠️ Unable to locate the selected city on map."

    m = folium.Map(location=coords, zoom_start=13)
    marker_cluster = MarkerCluster().add_to(m)

    city_l = city.lower()
    prop_type_l = property_type.lower()

    props = df_raw.copy()
    props["city"] = props["city"].astype(str).str.lower()
    props["property_type"] = props["property_type"].astype(str).str.lower()

    message = None

    # ---------------- STAGE 1: Strict filter ----------------
    filtered = props[
        (props["city"] == city_l) &
        (props["property_type"].str.contains(prop_type_l, na=False)) &
        (props["size"].between(size * 0.9, size * 1.1))
    ]

    # ---------------- STAGE 2: Relax property type ----------------
    if filtered.empty:
        filtered = props[
            (props["city"] == city_l) &
            (props["size"].between(size * 0.8, size * 1.2))
        ]
        message = (
            "ℹ️ No properties found for the selected property type. "
            "Showing similar-sized properties instead."
        )

    # ---------------- STAGE 3: City-only fallback ----------------
    if filtered.empty:
        filtered = props[props["city"] == city_l]
        message = (
            "⚠️ Limited data available. Showing general properties in this city."
        )

    # ---------------- STAGE 4: No properties at all ----------------
    if filtered.empty:
        add_rail_stations(m, region, city)
        filename = f"recommendation_map_{city_l.replace(' ', '_')}.html"
        m.save(os.path.join(MAPS_DIR, filename))

        return filename, [], (
            "❌ No rental listings were found for this city. "
            "Only nearby MRT/LRT/KTM stations are shown."
        )

    # ---------------- RANK BY RENT PROXIMITY ----------------
    filtered["rent_diff"] = abs(filtered["monthly_rent"] - predicted_rent)
    recs = filtered.sort_values("rent_diff").head(10)

    recommendations = []

    for _, r in recs.iterrows():
        near_rail = detect_rail_proximity(
            f"{r.get('facilities','')} {r.get('additional_facilities','')}"
        )

        lat, lon = coords
        if "latitude" in r and "longitude" in r and not pd.isna(r["latitude"]):
            lat, lon = r["latitude"], r["longitude"]

        popup = build_property_popup(r)

        folium.Marker(
            location=(lat, lon),
            popup=popup,
            icon=folium.Icon(
                color="blue" if near_rail else "green",
                icon="train" if near_rail else "home",
                prefix="fa"
            )
        ).add_to(marker_cluster)

        recommendations.append({
            "name": r.get("prop_name", "Unknown"),
            "rent": float(r.get("monthly_rent", 0)),
            "near_rail": near_rail,
            "rent_diff": float(r["rent_diff"])
        })

    add_rail_stations(m, region, city)

    filename = f"recommendation_map_{city_l.replace(' ', '_')}.html"
    m.save(os.path.join(MAPS_DIR, filename))

    return filename, recommendations, message






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

    # --- SAFETY INPUT VALIDATION ---
    size = float(user_input.get("size", 0))
    rooms = int(float(user_input.get("rooms", 0)))
    bathroom = int(float(user_input.get("bathroom", 1)))
    ptype = user_input.get("property_type", "").lower()

    if size < 200 or size > 10000:
        return "Invalid size input", 400

    if "studio" in ptype or "soho" in ptype:
        if rooms > 1 or bathroom != 1:
            return "Invalid studio configuration", 400

    if rooms < 0 or rooms > 10:
        return "Invalid room count", 400

    if bathroom < 1 or bathroom > 8:
        return "Invalid bathroom count", 400



    user_input = request.form.to_dict()
    X_user = build_user_features(user_input)

    # Base model
    rf_log = rf.predict(X_user)[0]

    # Stacked input
    X_user_h = X_user.copy()
    X_user_h["rf_pred"] = rf_log

    hybrid_log = xgb.predict(X_user_h)[0]
    predicted_rent = float(np.expm1(hybrid_log))

    confidence_info = get_prediction_confidence(
        predicted_rent,
        rf_log,
        hybrid_log,
        user_input["region"],
        user_input["city"]
    )

    return render_template(
        "result.html",
        rent=round(predicted_rent, 2),
        city=user_input["city"],
        region=user_input["region"],
        size=user_input["size"],
        property_type=user_input["property_type"],
        confidence=confidence_info,
        map_path=None,
        recommendations=[]
    )

    # return render_template(
    #     "result.html",
    #     rent=round(predicted_rent, 2),
    #     city=user_input["city"],
    #     region=user_input["region"],
    #     size=user_input["size"],
    #     map_path=map_filename,
    #     recommendations=recommendations
    # )
@app.route("/generate_map", methods=["POST"])
def generate_map():
    data = request.json

    region = data["region"]
    city = data["city"]
    rent = float(data["rent"])
    size = float(data["size"])
    property_type = data["property_type"]

    map_filename, recommendations, message= generate_map_and_recs(
        region, city, rent, size, property_type
    )

    return jsonify({
        "map_path": map_filename,
        "recommendations": recommendations,
        "message": message
    })


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

@app.route("/api/analytics")
def analytics_data():
    df = df_raw.copy()

    region = request.args.get("region")
    city = request.args.get("city")
    property_type = request.args.get("property_type")

    if region:
        df = df[df["region"] == region]
    if city:
        df = df[df["city"] == city]
    if property_type:
        df = df[df["property_type"] == property_type]

    response = {
        "kpis": {
            "avg_rent": round(df["monthly_rent"].mean(), 2),
            "max_rent": round(df["monthly_rent"].max(), 2),
            "median_rent": round(df["monthly_rent"].median(), 2),
            "avg_size": round(df["size"].mean(), 2),
            "total_listings": int(len(df)),
            "avg_psf": round(df["monthly_rent"].sum() / df["size"].sum(), 2)
        },

        "avg_rent_by_region": (
            df.groupby("region")["monthly_rent"]
            .mean()
            .round(0)
            .to_dict()
        ),

        "avg_rent_by_property_type": (
            df.groupby("property_type")["monthly_rent"]
            .mean()
            .round(0)
            .sort_values(ascending=False)
            .to_dict()
        ),

        "top_cities": (
            df.groupby("city")["monthly_rent"]
            .mean()
            .round(0)
            .sort_values(ascending=False)
            .head(10)
            .to_dict()
        )
    }

    return jsonify(response)


if __name__ == "__main__":
    print("Starting Flask app (debug mode)...")
    app.run(debug=True)
