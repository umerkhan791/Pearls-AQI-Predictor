import os
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "aqi_model.pkl"
FEATURE_DB = BASE_DIR / "data" / "feature_store.db"

MODEL_NAME = "RandomForestRegressor"

# Current metrics from the latest training run
MODEL_METRICS = {
    "mae": 0.5297,
    "rmse": 0.7140,
    "r2": 0.9944,
}


# ============================================================
# LOAD MODEL ONCE
# ============================================================

model = joblib.load(MODEL_PATH)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Pearls AQI Predictor API",
    description="Karachi AQI prediction API",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATABASE HELPERS
# ============================================================

def get_latest_features():
    """
    Get the latest engineered feature row from the local
    SQLite feature-store cache.
    """

    if not FEATURE_DB.exists():
        raise FileNotFoundError(
            f"Feature database not found: {FEATURE_DB}"
        )

    connection = sqlite3.connect(FEATURE_DB)

    try:
        query = """
            SELECT *
            FROM features
            ORDER BY timestamp DESC
            LIMIT 1
        """

        df = pd.read_sql_query(query, connection)

    finally:
        connection.close()

    if df.empty:
        raise ValueError("No feature data available.")

    return df.iloc[0]


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

FEATURE_COLUMNS = [
    "hour",
    "day",
    "month",
    "day_of_week",
    "is_weekend",
    "aqi_lag_1h",
    "aqi_lag_3h",
    "aqi_lag_6h",
    "aqi_lag_12h",
    "aqi_lag_24h",
    "aqi_roll_mean_3h",
    "aqi_roll_mean_6h",
    "aqi_roll_mean_12h",
    "aqi_roll_mean_24h",
    "aqi_roll_std_6h",
    "aqi_roll_std_24h",
    "aqi_change_1h",
    "aqi_change_6h",
    "aqi_change_24h",
    "pm25",
    "pm10",
    "o3",
    "no2",
    "so2",
    "co",
    "temperature",
    "humidity",
    "pressure",
    "wind",
]


# ============================================================
# AQI CATEGORY
# ============================================================

def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "name": "Pearls AQI Predictor",
        "status": "running",
        "model": MODEL_NAME,
        "features": len(FEATURE_COLUMNS),
        "data_source": "SQLite feature store cache",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "feature_database_exists": FEATURE_DB.exists(),
    }


# ============================================================
# TEST
# ============================================================

@app.get("/test")
def test():
    return {
        "message": "Pearls AQI Predictor API is working",
        "model": MODEL_NAME,
        "metrics": MODEL_METRICS,
    }


# ============================================================
# NEXT-HOUR PREDICTION
# ============================================================

@app.get("/predict")
def predict():
    latest = get_latest_features()

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in latest.index
    ]

    if missing:
        return {
            "status": "error",
            "message": "Missing required features",
            "missing_features": missing,
        }

    # Use the latest available features at time t
    # to predict AQI at t+1.
    input_data = pd.DataFrame(
        [[latest[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )

    prediction = float(model.predict(input_data)[0])

    current_aqi = float(latest["aqi"])

    timestamp = pd.to_datetime(latest["timestamp"])

    prediction_time = timestamp + pd.Timedelta(hours=1)

    response = {
        "status": "success",
        "timestamp": timestamp.isoformat(),
        "prediction_for": prediction_time.isoformat(),
        "current_aqi": round(current_aqi, 2),
        "predicted_aqi": round(prediction, 2),
        "category": get_aqi_category(prediction),
        "model": MODEL_NAME,
        "prediction_horizon": "1 hour",
        "data_source": "SQLite feature store cache",
        "model_metrics": {
            "mae": MODEL_METRICS["mae"],
            "rmse": MODEL_METRICS["rmse"],
            "r2": MODEL_METRICS["r2"],
        },
        "pollutants": {
            "pm25": float(latest["pm25"]),
            "pm10": float(latest["pm10"]),
            "o3": float(latest["o3"]),
            "no2": float(latest["no2"]),
            "so2": float(latest["so2"]),
            "co": float(latest["co"]),
        },
        "weather": {
            "temperature": float(latest["temperature"]),
            "humidity": float(latest["humidity"]),
            "pressure": float(latest["pressure"]),
            "wind": float(latest["wind"]),
        },
    }

    return response