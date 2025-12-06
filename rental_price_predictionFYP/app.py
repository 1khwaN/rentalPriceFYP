from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import folium
from folium.plugins import MarkerCluster
import requests
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

model_data = joblib.load("hybrid_rent_model_city.pkl")
rf = model_data["rf"]
xgb = model_data["xgb"]
model_columns = model_data["columns"]

@app.route("/")
def home():
    return render_template("dashboard.html")

@app.route("/predict_form")
def predict_form():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    region = request.form["region"]
    city = request.form["city"]
    size = float(request.form["size"])
    completion_year = int(request.form["completion_year"])
    parking = int(request.form["parking"])
    rooms = int(request.form["rooms"])
    bathroom = int(request.form["bathroom"])
    property_type = request.form["property_type"]
    furnished = request.form["furnished"]

    rooms_per_bathroom = rooms / (bathroom + 1)
    age_of_property = 2025 - completion_year

    df = pd.DataFrame([{
        "size": size,
        "completion_year": completion_year,
        "parking": parking,
        "rooms": rooms,
        "bathroom": bathroom,
        "region": region,
        "city": city,
        "property_type": property_type,
        "furnished": furnished,
        "price_per_sqft": 0,
        "rooms_per_bathroom": rooms_per_bathroom,
        "age_of_property": age_of_property
    }])

    df = pd.get_dummies(df, columns=["region","city","property_type","furnished"], drop_first=True)
    df = df.reindex(columns=model_columns, fill_value=0)

    rf_pred = rf.predict(df)
    xgb_pred = xgb.predict(df)
    hybrid = (rf_pred + xgb_pred) / 2
    rent = float(np.expm1(hybrid)[0])

    return render_template("result.html", rent=rent, city=city, region=region, size=size)

@app.route("/api/predict", methods=["POST"])
def api():
    data = request.get_json()
    df = pd.DataFrame([data])
    df["price_per_sqft"]=0
    df["rooms_per_bathroom"]=df["rooms"]/(df["bathroom"]+1)
    df["age_of_property"]=2025-df["completion_year"]
    df=pd.get_dummies(df,columns=["region","city","property_type","furnished"],drop_first=True)
    df=df.reindex(columns=model_columns,fill_value=0)
    rf_pred=rf.predict(df)
    xgb_pred=xgb.predict(df)
    hybrid=(rf_pred+xgb_pred)/2
    return jsonify({"predicted_rent":float(np.expm1(hybrid)[0])})

if __name__=="__main__":
    app.run(debug=True)
