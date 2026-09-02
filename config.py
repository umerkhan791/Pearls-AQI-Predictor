"""
config.py — Central configuration for AQI Predictor
All API keys, city settings, paths, and model parameters live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

DB_PATH = DATA_DIR / "feature_store.db"

# ─── API Configuration ────────────────────────────────────────────────────────
AQICN_BASE_URL = "https://api.waqi.info"

# ─── City Configuration ───────────────────────────────────────────────────────
# You can change this to any city supported by AQICN
CITY = "karachi"
CITY_DISPLAY_NAME = "Karachi, Pakistan"

# Alternate cities you can switch to:
# CITY = "beijing"
# CITY = "delhi"
# CITY = "lahore"

# ─── Feature Engineering Settings ────────────────────────────────────────────
# Pollutant columns returned by AQICN
POLLUTANT_COLS = ["pm25", "pm10", "o3", "no2", "so2", "co"]
WEATHER_COLS = ["t", "h", "p", "w"]  # temp, humidity, pressure, wind
ALL_RAW_COLS = POLLUTANT_COLS + WEATHER_COLS

# Time-based features to compute
TIME_FEATURES = ["hour", "day", "month", "day_of_week", "is_weekend"]

# Lag features (AQI values from N hours ago)
LAG_HOURS = [1, 3, 6, 12, 24]

# Rolling window sizes for mean/std features
ROLLING_WINDOWS = [3, 6, 12, 24]

# ─── Model Configuration ──────────────────────────────────────────────────────
# Target: predict AQI N hours into the future
FORECAST_HORIZON_HOURS = 72   # 3 days = 72 hours
FORECAST_DAYS = 3

# Train/test split ratio
TRAIN_TEST_SPLIT = 0.8

# Random seed for reproducibility
RANDOM_SEED = 42

# Model evaluation metrics
METRICS = ["rmse", "mae", "r2"]

# Model filenames saved to models/
MODEL_REGISTRY = {
    "random_forest": MODELS_DIR / "random_forest.pkl",
    "xgboost": MODELS_DIR / "xgboost.pkl",
    "best_model": MODELS_DIR / "best_model.pkl",
    "scaler": MODELS_DIR / "scaler.pkl",
}

# ─── AQI Level Thresholds (US EPA standard) ───────────────────────────────────
AQI_LEVELS = {
    "Good":              (0,   50,   "#00e400"),
    "Moderate":          (51,  100,  "#ffff00"),
    "Unhealthy for SG":  (101, 150,  "#ff7e00"),
    "Unhealthy":         (151, 200,  "#ff0000"),
    "Very Unhealthy":    (201, 300,  "#8f3f97"),
    "Hazardous":         (301, 500,  "#7e0023"),
}

# Alert threshold — send alert if predicted AQI exceeds this
ALERT_THRESHOLD = 150   # "Unhealthy" level

# ─── CI/CD Scheduling ─────────────────────────────────────────────────────────
FEATURE_PIPELINE_CRON = "0 * * * *"    # every hour
TRAINING_PIPELINE_CRON = "0 2 * * *"  # every day at 2am


def get_aqi_level(aqi_value: float) -> tuple[str, str]:
    """Return (level_name, hex_color) for a given AQI value."""
    for level, (low, high, color) in AQI_LEVELS.items():
        if low <= aqi_value <= high:
            return level, color
    return "Hazardous", "#7e0023"


def ensure_dirs():
    """Create all required directories if they don't exist."""
    for d in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
