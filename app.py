from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

data = pickle.load(open("diabetes_model.pkl", "rb"))
model  = data["model"]
scaler = data["scaler"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [
        float(request.form["pregnancies"]),
        float(request.form["glucose"]),
        float(request.form["blood_pressure"]),
        float(request.form["skin_thickness"]),
        float(request.form["insulin"]),
        float(request.form["bmi"]),
        float(request.form["dpf"]),
        float(request.form["age"]),
    ]
    scaled = scaler.transform([features])
    prediction = model.predict(scaled)
    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)