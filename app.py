import numpy as np
from flask import Flask, render_template, request
import pickle
import os

# Initialize Flask app
app = Flask(__name__)

# ==============================
# Load trained models
# ==============================
slr_model, mlr_model = None, None

# Load SLR model
if os.path.exists("models/SLR_model.pkl"):
    with open("models/SLR_model.pkl", "rb") as f:
        slr_model = pickle.load(f)
else:
    print("⚠️ SLR_model.pkl not found in models/")

# Load MLR model
if os.path.exists("models/MLR_model.pkl"):
    with open("models/MLR_model.pkl", "rb") as f:
        mlr_model = pickle.load(f)
else:
    print("⚠️ MLR_model.pkl not found in models/")

# ==============================
# Routes
# ==============================
@app.route("/")
def home():
    return render_template("index.html")

# ---------- SLR ----------
@app.route("/slr")
def slr_page():
    return render_template("slr.html")

@app.route("/predict_slr", methods=["POST"])
def predict_slr():
    if slr_model is None:
        return render_template("slr.html", slr_result="⚠️ Model not available. Train first!")
    try:
        exp = float(request.form["experience"])
        prediction = slr_model.predict(np.array([[exp]]))[0]
        return render_template("slr.html", slr_result=f"Predicted Salary: {prediction:.2f}")
    except Exception as e:
        return render_template("slr.html", slr_result=f"Error: {e}")

# ---------- MLR ----------
@app.route("/mlr")
def mlr_page():
    return render_template("mlr.html")

@app.route("/predict_mlr", methods=["POST"])
def predict_mlr():
    if mlr_model is None:
        return render_template("mlr.html", mlr_result="⚠️ Model not available. Train first!")
    try:
        rnd = float(request.form["rnd"])
        admin = float(request.form["admin"])
        marketing = float(request.form["marketing"])
        state = int(request.form["state"])

        input_features = np.array([[rnd, admin, marketing, state]])
        prediction = mlr_model.predict(input_features)[0]
        return render_template("mlr.html", mlr_result=f"Predicted Profit: {prediction:.2f}")
    except Exception as e:
        return render_template("mlr.html", mlr_result=f"Error: {e}")

# ---------- About ----------
@app.route("/about")
def about():
    return render_template("about.html")

# ---------- Contact ----------
@app.route("/contact")
def contact():
    return render_template("contact.html")

# ==============================
if __name__ == "__main__":
    app.run(debug=True)
