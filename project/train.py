# train.py
# Trains the model from data/cardata.json and saves the model and normalization params.

import json
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

DATA_PATH = "data/cardata.json"
MODEL_DIR = "saved_model"
SCALER_FILE = os.path.join(MODEL_DIR, "scalers.pkl")

def load_data(path=DATA_PATH):
    with open(path, "r") as f:
        data = json.load(f)
    # original code used fields Horsepower and Miles_per_Gallon
    df = pd.DataFrame(data)
    df = df[["Horsepower", "Miles_per_Gallon"]].copy()
    df = df.dropna()
    df = df.astype(float)
    return df

def normalize(series):
    minv = series.min()
    maxv = series.max()
    return (series - minv) / (maxv - minv), minv, maxv

def denormalize(normed, minv, maxv):
    return normed * (maxv - minv) + minv

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1,)),
        tf.keras.layers.Dense(1, use_bias=True),
        tf.keras.layers.Dense(1, use_bias=True)
    ])
    model.compile(optimizer=tf.keras.optimizers.SGD(), loss='mean_squared_error')
    return model

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_data()
    X = df["Horsepower"].values.reshape(-1,1)
    y = df["Miles_per_Gallon"].values.reshape(-1,1)

    # Normalize (store params to re-use at inference)
    X_norm, X_min, X_max = normalize(pd.Series(X.flatten()))
    y_norm, y_min, y_max = normalize(pd.Series(y.flatten()))
    X_norm = X_norm.values.reshape(-1,1)
    y_norm = y_norm.values.reshape(-1,1)

    # train/val split
    X_train, X_val, y_train, y_val = train_test_split(X_norm, y_norm, test_size=0.1, random_state=42)

    model = build_model()
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=50, batch_size=25, shuffle=True, verbose=1)

    # Save model
    model.save(MODEL_DIR)

    # Save normalization params so the flask app can un-normalize predictions
    scalers = {
        "X_min": float(X_min),
        "X_max": float(X_max),
        "y_min": float(y_min),
        "y_max": float(y_max)
    }
    joblib.dump(scalers, SCALER_FILE)

    # Plot original points and model prediction line (denormalized)
    xs = np.linspace(X_min, X_max, 200)
    xs_norm = (xs - X_min) / (X_max - X_min)
    preds_norm = model.predict(xs_norm.reshape(-1,1)).flatten()
    preds = denormalize(preds_norm, y_min, y_max)

    plt.figure(figsize=(6,4))
    plt.scatter(df["Horsepower"], df["Miles_per_Gallon"], label="Original")
    plt.plot(xs, preds, label="Model prediction (line)", linewidth=2)
    plt.xlabel("Horsepower")
    plt.ylabel("MPG")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "fit_plot.png"))
    print("Training complete. Model and scalers saved to", MODEL_DIR)

if __name__ == "__main__":
    main()
    