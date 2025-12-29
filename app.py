import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

model = pickle.load(open("regmodel.pkl","rb"))
scaler = pickle.load(open("scaling.pkl","rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.json["data"]

    df = pd.DataFrame([data])          # IMPORTANT
    scaled = scaler.transform(df)      # StandardScaler
    output = model.predict(scaled)

    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)
