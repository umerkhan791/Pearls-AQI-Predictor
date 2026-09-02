import json
import os
import shutil
from pathlib import Path

import hopsworks
import joblib

MODEL_PATH = Path("aqi_model.pkl")
METRICS_PATH = Path("reports/training_metrics.json")
UPLOAD_DIR = Path("registry_upload")
MODEL_NAME = "karachi_aqi_next_hour_rf"

print("=" * 70)
print("HOPSWORKS MODEL REGISTRY")
print("=" * 70)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not METRICS_PATH.exists():
    raise FileNotFoundError(f"Metrics file not found: {METRICS_PATH}")

with open(METRICS_PATH, "r", encoding="utf-8") as f:
    metrics = json.load(f)

joblib.load(MODEL_PATH)

print(f"Model: {MODEL_PATH}")
print(f"Metrics: {metrics}")

api_key = os.getenv("HOPSWORKS_API_KEY")
if not api_key:
    raise RuntimeError("HOPSWORKS_API_KEY not found.")

# Prepare a real local directory containing the model artifact
if UPLOAD_DIR.exists():
    shutil.rmtree(UPLOAD_DIR)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
shutil.copy2(MODEL_PATH, UPLOAD_DIR / MODEL_PATH.name)

print(f"Upload directory: {UPLOAD_DIR.resolve()}")

project = hopsworks.login(api_key_value=api_key)
mr = project.get_model_registry()
fs = project.get_feature_store()

registered_model = mr.sklearn.create_model(
    name=MODEL_NAME,
    metrics={
        "mae": float(metrics["mae"]),
        "rmse": float(metrics["rmse"]),
        "r2": float(metrics["r2"]),
    },
    description=(
        "Random Forest model for Karachi next-hour AQI prediction. "
        "Target is AQI at t+1 using features available at time t."
    ),
    feature_view=fs.get_feature_view("karachi_aqi_fv", 1),
    training_dataset_version=1,
)

registered_model.save(UPLOAD_DIR)

print("\n" + "=" * 70)
print("MODEL REGISTRATION SUCCESSFUL")
print("=" * 70)
print(f"Model name: {MODEL_NAME}")
print(f"Metrics: {metrics}")
print("=" * 70)
