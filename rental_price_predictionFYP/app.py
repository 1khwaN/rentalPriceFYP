from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import joblib
import folium
from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim
import os
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ============================================================
# FILES — MUST EXIST IN SAME FOLDER AS app.py
# ============================================================
MODEL_PKL = "hybrid_rent_model_city.pkl"
MODEL_READY_CSV = "cleaned_rental_dataset_model_ready.csv"
RAW_CSV = "cleaned_rental_dataset_before_encoding.csv"   # for recommendations

# ============================================================
# LOAD CLEANED MODEL-READY DATA (NO RE-CLEANING!)
# ============================================================
df_model_ready = pd.read_csv(MODEL_READY_CSV)

# Rebuild X exactly the same way as the notebook
drop_cols = ["ads_id", "prop_name", "facilities", "additional_facilities",
             "monthly_rent", "location"]

X_reference = df_model_ready.drop(columns=[c for c in drop_cols if c in df_model_ready],
                                  errors="ignore")

MODEL_COLUMNS = X_reference.columns.tolist()   # EXACT training columns

print(f"Loaded model-ready CSV → {len(MODEL_COLUMNS)} model columns (must match notebook).")


# ============================================================
# LOAD RAW CLEANED DATA FOR RECOMMENDATIONS
# ============================================================
df_original = pd.read_csv(RAW_CSV)

# Normalize text formats
df_original["region"] = df_original["region"].astype(str).str.title()
df_original["city"] = df_original["city"].astype(str).str.title()


# ============================================================
# LOAD MODEL
# ============================================================
if not os.path.exists(MODEL_PKL):
    raise FileNotFoundError("Model file hybrid_rent_model_city.pkl not found!")

model_data = joblib.load(MODEL_PKL)
rf = model_data["rf"]
xgb = model_data["xgb"]

print("RandomForest & XGBoost loaded successfully.")


# ============================================================
# USER INPUT → MODEL FEATURES (MATCH NOTEBOOK EXACTLY)
# ============================================================
def preprocess_user_input(data):
    # Normalize text input
    region = str(data["region"]).title()
    city = str(data["city"]).title()
    property_type = str(data["property_type"]).title()
    furnished = str(data["furnished"]).title()

    # Safe conversions
    size = float(data["size"])
    completion_year = int(float(data["completion_year"]))
    parking = int(float(data["parking"]))
    rooms = float(data["rooms"])
    bathroom = float(data["bathroom"])

    # Derived features (same as notebook)
    rooms_per_bathroom = rooms / (bathroom + 1)
    age_of_property = 2025 - completion_year

    # Determine price_per_sqft using city-based baseline
    df_city = df_original[df_original["city"].str.lower() == city.lower()]
    if len(df_city) > 0:
        price_per_sqft = df_city["monthly_rent"].mean() / (df_city["size"].mean() + 1)
    else:
        price_per_sqft = df_original["monthly_rent"].mean() / (df_original["size"].mean() + 1)

    row = {
        "size": size,
        "completion_year": completion_year,
        "parking": parking,
        "rooms": rooms,
        "bathroom": bathroom,
        "region": region,
        "city": city,
        "property_type": property_type,
        "furnished": furnished,
        "price_per_sqft": price_per_sqft,
        "rooms_per_bathroom": rooms_per_bathroom,
        "age_of_property": age_of_property
    }

    df_input = pd.DataFrame([row])

    # SAME ONE-HOT ENCODING AS NOTEBOOK
    df_input = pd.get_dummies(df_input,
        columns=["region", "city", "property_type", "furnished"],
        drop_first=True
    )

    # ALIGN COLUMNS EXACTLY LIKE TRAINING DATA
    aligned = pd.DataFrame(columns=MODEL_COLUMNS)
    df_input = df_input.reindex(columns=MODEL_COLUMNS, fill_value=0)

    return df_input


# ============================================================
# PREDICT ENDPOINT
# ============================================================
@app.route("/predict", methods=["POST"])
def predict():
    data = {
        "region": request.form["region"],
        "city": request.form["city"],
        "size": request.form["size"],
        "completion_year": request.form["completion_year"],
        "parking": request.form["parking"],
        "rooms": request.form["rooms"],
        "bathroom": request.form["bathroom"],
        "property_type": request.form["property_type"],
        "furnished": request.form["furnished"]
    }

    X_user = preprocess_user_input(data)

    # Debugging
    print("==== MODEL INPUT DEBUG ====")
    print("Shape:", X_user.shape)
    print("Non-zero features:", X_user.sum().sum())

    # Predict (log scale)
    rf_log = rf.predict(X_user)[0]
    xgb_log = xgb.predict(X_user)[0]

    # Correct hybrid prediction
    hybrid_log = (rf_log + xgb_log) / 2
    rent = float(np.expm1(hybrid_log))

    # Generate recommendations
    map_file, recs = generate_map_and_recs(data["region"], data["city"], rent)

    return render_template("result.html",
        rent=round(rent, 2),
        city=data["city"],
        region=data["region"],
        size=data["size"],
        recommendations=recs,
        map_path=map_file
    )


# ============================================================
# MAP + RECOMMENDATIONS
# ============================================================
def generate_map_and_recs(region, city, predicted_rent):
    try:
        geolocator = Nominatim(user_agent="rent_app")
        loc = geolocator.geocode(f"{city}, {region}, Malaysia")

        if not loc:
            return None, []

        user_coords = (loc.latitude, loc.longitude)

        # Base map
        m = folium.Map(location=user_coords, zoom_start=12)
        MarkerCluster().add_to(m)

        # Filter city properties
        df_city = df_original[df_original["city"].str.lower() == city.lower()].copy()

        # Recommend properties within ±10%
        min_rent = predicted_rent * 0.9
        max_rent = predicted_rent * 1.1

        recs = df_city[(df_city["monthly_rent"] >= min_rent) &
                       (df_city["monthly_rent"] <= max_rent)]

        if recs.empty:
            recs = df_city.sort_values("monthly_rent").head(5)

        recommendations = []
        for _, row in recs.iterrows():
            name = row["prop_name"]
            rent = row["monthly_rent"]

            try:
                ploc = geolocator.geocode(f"{name}, {city}, Malaysia")
                coords = (ploc.latitude, ploc.longitude) if ploc else user_coords
            except:
                coords = user_coords

            folium.Marker(
                coords,
                popup=f"{name} — RM {rent}",
                icon=folium.Icon(color="green")
            ).add_to(m)

            recommendations.append({"name": name, "rent": rent})

        os.makedirs("maps", exist_ok=True)
        file_path = f"recommendation_map_{city.replace(' ', '_')}.html"
        m.save(os.path.join("maps", file_path))

        return file_path, recommendations

    except Exception as e:
        print("MAP ERROR:", e)
        return None, []


@app.route("/maps/<path:filename>")
def serve_map(filename):
    return send_from_directory("maps", filename)


# ============================================================
# HOME & FORM ROUTES
# ============================================================
@app.route("/")
def home():
    return render_template("dashboard.html")

@app.route("/predict_form")
def predict_form():
    return render_template("index.html")


# ============================================================
# START FLASK APP
# ============================================================
if __name__ == "__main__":
    app.run(debug=True)
