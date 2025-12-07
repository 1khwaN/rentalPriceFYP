from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import joblib
from geopy.geocoders import Nominatim
import folium
from folium.plugins import MarkerCluster
import os
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# =========================================================
# FILES
# =========================================================
CLEANED_BEFORE_CSV = "cleaned_rental_dataset_before_encoding.csv"
MODEL_READY_CSV    = "cleaned_rental_dataset_model_ready.csv"
MODEL_PKL          = "hybrid_rent_model_city.pkl"

# =========================================================
# LOAD DATASETS
# =========================================================
if not os.path.exists(CLEANED_BEFORE_CSV):
    raise FileNotFoundError(f"Missing {CLEANED_BEFORE_CSV}")

if not os.path.exists(MODEL_READY_CSV):
    raise FileNotFoundError(f"Missing {MODEL_READY_CSV}")

df_original = pd.read_csv(CLEANED_BEFORE_CSV)       # contains region, city, monthly_rent, etc.
df_model_ready = pd.read_csv(MODEL_READY_CSV)       # used to get model columns


# =========================================================
# LOAD MODEL
# =========================================================
model_data = joblib.load(MODEL_PKL)
rf = model_data["rf"]
xgb = model_data["xgb"]

# Get the true feature columns — EXACTLY as used during training
drop_cols = ["ads_id","prop_name","facilities","additional_facilities","monthly_rent","location"]
X_train_used = df_model_ready.drop(columns=[c for c in drop_cols if c in df_model_ready.columns], errors='ignore')
model_columns = X_train_used.columns.tolist()

print("\n=== Loaded Model Columns ===")
print(len(model_columns), "columns")
print(model_columns[:40])
print("=============================\n")


# =========================================================
# PREPROCESS USER INPUT
# =========================================================
def preprocess_user_input(d):
    def safe_float(v, fb=np.nan):
        try: return float(v)
        except: return fb

    def safe_int(v, fb=0):
        try: return int(float(v))
        except: return fb

    region = d.get("region","").title().strip()
    city = d.get("city","").title().strip()
    property_type = d.get("property_type","").title().strip()
    furnished = d.get("furnished","").title().strip()

    size = safe_float(d.get("size"))
    comp_year = safe_int(d.get("completion_year"))
    rooms = safe_float(d.get("rooms"))
    bath = safe_float(d.get("bathroom"))
    parking = safe_int(d.get("parking"))

    # fill missing using dataset medians
    if np.isnan(size): size = df_original["size"].median()
    if np.isnan(rooms): rooms = df_original["rooms"].median()
    if np.isnan(bath): bath = df_original["bathroom"].median()
    if comp_year <= 0: comp_year = int(df_original["completion_year"].median())

    # Compute city avg price_per_sqft
    df_city = df_original[df_original["city"].str.lower() == city.lower()]
    if not df_city.empty:
        price_per_sqft = df_city["monthly_rent"].mean() / (df_city["size"].mean() + 1)
    else:
        price_per_sqft = df_original["monthly_rent"].mean() / (df_original["size"].mean() + 1)

    row = {
        "size": size,
        "completion_year": comp_year,
        "parking": parking,
        "rooms": rooms,
        "bathroom": bath,
        "region": region,
        "city": city,
        "property_type": property_type,
        "furnished": furnished,
        "price_per_sqft": price_per_sqft,
        "rooms_per_bathroom": rooms / (bath + 1),
        "age_of_property": 2025 - comp_year
    }

    df = pd.DataFrame([row])

    df = pd.get_dummies(df, columns=["region","city","property_type","furnished"], drop_first=True)

    df = df.reindex(columns=model_columns, fill_value=0)

    return df


# =========================================================
# ROUTES
# =========================================================
@app.route("/")
def home():
    return render_template("dashboard.html")

@app.route("/predict_form")
def form():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    user_input = {
        "region": request.form.get("region",""),
        "city": request.form.get("city",""),
        "size": request.form.get("size",""),
        "completion_year": request.form.get("completion_year",""),
        "parking": request.form.get("parking","0"),
        "rooms": request.form.get("rooms",""),
        "bathroom": request.form.get("bathroom",""),
        "property_type": request.form.get("property_type",""),
        "furnished": request.form.get("furnished","")
    }

    X = preprocess_user_input(user_input)

    rf_log = rf.predict(X)[0]
    xgb_log = xgb.predict(X)[0]
    hybrid_log = (rf_log + xgb_log) / 2
    rent = float(np.expm1(hybrid_log))

    map_path, recommendations = generate_map_and_recs(
        user_input["region"], user_input["city"], rent
    )

    return render_template("result.html",
                           rent=round(rent,2),
                           city=user_input["city"].title(),
                           region=user_input["region"].title(),
                           size=user_input["size"],
                           map_path=map_path,
                           recommendations=recommendations)


# =========================================================
# MAP + RECOMMENDATIONS
# =========================================================
def generate_map_and_recs(region, city, predicted_rent):
    geolocator = Nominatim(user_agent="rental_app")

    try:
        loc = geolocator.geocode(f"{city}, {region}, Malaysia")
        if not loc:
            return None, []

        user_coords = (loc.latitude, loc.longitude)

        m = folium.Map(location=user_coords, zoom_start=13)
        MarkerCluster().add_to(m)

        # Filter properties in this city
        city_df = df_original[df_original["city"].str.lower() == city.lower()]

        if city_df.empty:
            city_df = df_original[df_original["region"].str.lower() == region.lower()]

        low, high = predicted_rent*0.9, predicted_rent*1.1
        recs = city_df[(city_df["monthly_rent"] >= low) & (city_df["monthly_rent"] <= high)]

        if recs.empty:
            recs = city_df.nlargest(5, "monthly_rent")

        recommendations = []

        for _, row in recs.iterrows():
            prop = row["prop_name"]
            rent = row["monthly_rent"]

            p_loc = geolocator.geocode(f"{prop}, {city}, {region}, Malaysia")
            marker_coords = (p_loc.latitude, p_loc.longitude) if p_loc else user_coords

            folium.Marker(
                marker_coords,
                popup=f"{prop} - RM {rent:,.2f}",
                icon=folium.Icon(color="green")
            ).add_to(m)

            recommendations.append({"name": prop, "rent": float(rent)})

        os.makedirs("maps", exist_ok=True)
        file = f"map_{city.replace(' ','_')}.html"
        m.save(f"maps/{file}")
        return file, recommendations

    except Exception as e:
        print("MAP ERROR:", e)
        return None, []


@app.route("/maps/<path:filename>")
def maps(filename):
    return send_from_directory("maps", filename)


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)
