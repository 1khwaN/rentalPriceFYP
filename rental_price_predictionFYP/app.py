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

# ---------------- GEO CACHE ----------------
city_cache = {}

# ---------- helper functions ----------
from geopy.extra.rate_limiter import RateLimiter

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
    coords = get_city_coords(region, city)
    if not coords:
        return None, []

    # Base map
    m = folium.Map(location=coords, zoom_start=13)
    marker_cluster = MarkerCluster().add_to(m)

    # Filter properties by city
    props = df_raw[df_raw["city"].astype(str).str.lower() == city.lower()].copy()

    if props.empty:
        return None, []

    # Filter by range first
    min_rent, max_rent = predicted_rent * 0.9, predicted_rent * 1.1
    recs = props[(props["monthly_rent"] >= min_rent) & (props["monthly_rent"] <= max_rent)].copy()

    # Fallback if empty
    if recs.empty:
        recs = props.copy()

    # Rank by closeness to predicted rent
    recs["rent_diff"] = abs(recs["monthly_rent"] - predicted_rent)
    recs = recs.sort_values("rent_diff").head(11)

    # HARD LIMIT (important!)
    recs = recs.head(10)

    recommendations = []

    for _, r in recs.iterrows():
        popup = build_property_popup(r)

        prop_name = r.get("prop_name", "Unknown")
        rent = r.get("monthly_rent", 0)

        #Detect MRT/LRT
        near_rail = detect_rail_proximity(
            f"{r.get('facilities','')} {r.get('additional_facilities','')}"
        )

        # Use city center coords (FAST & STABLE)
        icon_color = "blue" if near_rail else "green"
        icon_name = "train" if near_rail else "home"

        folium.Marker(
            location=coords,
            popup=popup,
            icon=folium.Icon(color=icon_color, icon=icon_name, prefix="fa")
        ).add_to(marker_cluster)

        recommendations.append({
            "name": prop_name,
            "rent": float(rent),
            "near_rail": near_rail,
            "rent_diff": float(r["rent_diff"])
        })

    legend_html = """
    <div style="
        position: fixed;
        bottom: 30px;
        left: 30px;
        background: white;
        padding: 10px 14px;
        border-radius: 10px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.2);
        font-size: 14px;
    ">
    <b>Legend</b><br>
    <i class="fa fa-home" style="color:green"></i> Property<br>
    <i class="fa fa-train" style="color:blue"></i> Near MRT/LRT
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


    filename = f"recommendation_map_{city.replace(' ', '_')}.html"
    filepath = os.path.join(MAPS_DIR, filename)
    m.save(filepath)

    return filename, recommendations




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

    # DO NOT generate map here
    return render_template(
        "result.html",
        rent=round(predicted_rent, 2),
        city=user_input["city"],
        region=user_input["region"],
        size=user_input["size"],
        map_path=None,      # <-- IMPORTANT
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

    map_filename, recommendations = generate_map_and_recs(region, city, rent)

    return jsonify({
        "map_path": map_filename,
        "recommendations": recommendations
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


if __name__ == "__main__":
    print("Starting Flask app (debug mode)...")
    app.run(debug=True)
