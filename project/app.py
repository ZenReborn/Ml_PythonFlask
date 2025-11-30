# app.py
from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import tensorflow as tf
import joblib
import os

MODEL_DIR = "saved_model"
SCALER_FILE = os.path.join(MODEL_DIR, "scalers.pkl")

app = Flask(__name__)

# Load model and scalers once at startup
model = tf.keras.models.load_model(MODEL_DIR)
scalers = joblib.load(SCALER_FILE)

def normalize_x(x):
    return (x - scalers["X_min"]) / (scalers["X_max"] - scalers["X_min"])

def denormalize_y(y_norm):
    return y_norm * (scalers["y_max"] - scalers["y_min"]) + scalers["y_min"]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON: {"horsepower": 150}
    Returns JSON: {"horsepower":150, "mpg": 25.3}
    """
    data = request.get_json() or {}
    try:
        hp = float(data.get("horsepower", None))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid horsepower value"}), 400

    x_norm = normalize_x(hp)
    pred_norm = model.predict(np.array([[x_norm]]), verbose=0).flatten()[0]
    mpg = float(denormalize_y(pred_norm))
    return jsonify({"horsepower": hp, "mpg": mpg})

# Serve the training plot image and saved model assets if needed
@app.route("/model/plot")
def model_plot():
    return send_from_directory(MODEL_DIR, "fit_plot.png")

if __name__ == "__main__":
    # For development only; for production use gunicorn/uvicorn etc.
    app.run(debug=True, host="0.0.0.0", port=5000)
