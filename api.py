import sqlite3
from pathlib import Path

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

MODEL_PATH = BASE_DIR / "aqi_model.pkl"
FEATURE_DB = BASE_DIR / "data" / "feature_store.db"

MODEL_NAME = "RandomForestRegressor"

MODEL_METRICS = {
    "mae": 0.5297,
    "rmse": 0.7140,
    "r2": 0.9944,
}


# ============================================================
# LOAD MODEL
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
# FEATURE COLUMNS
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


def get_historical_aqi():
    """
    Load historical Karachi AQI values from the local
    SQLite feature-store cache.
    """

    if not FEATURE_DB.exists():
        raise FileNotFoundError(
            f"Feature database not found: {FEATURE_DB}"
        )

    connection = sqlite3.connect(FEATURE_DB)

    try:
        query = """
            SELECT timestamp, aqi
            FROM raw_readings
            WHERE city = 'karachi'
              AND aqi IS NOT NULL
            ORDER BY timestamp ASC
        """

        df = pd.read_sql_query(query, connection)

    finally:
        connection.close()

    if df.empty:
        raise ValueError("No historical AQI data available.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")

    return (
        df.dropna(subset=["timestamp", "aqi"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


# ============================================================
# AQI CATEGORY
# ============================================================

def get_aqi_category(aqi):
    """
    Convert US AQI value into a standard category.
    """

    aqi = float(aqi)

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
# LIVE AQI + WEATHER
# ============================================================

def get_live_aqi():
    """
    Fetch current Karachi AQI, pollutants and weather
    from Open-Meteo.
    """

    aq_url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
    )

    weather_url = (
        "https://api.open-meteo.com/v1/forecast"
    )

    aq_params = {
        "latitude": 24.8607,
        "longitude": 67.0011,
        "current": (
            "us_aqi,"
            "pm2_5,"
            "pm10,"
            "ozone,"
            "nitrogen_dioxide,"
            "sulphur_dioxide,"
            "carbon_monoxide"
        ),
        "timezone": "Asia/Karachi",
    }

    weather_params = {
        "latitude": 24.8607,
        "longitude": 67.0011,
        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "surface_pressure,"
            "wind_speed_10m"
        ),
        "timezone": "Asia/Karachi",
    }

    try:

        # ----------------------------------------------------
        # AQI
        # ----------------------------------------------------

        aq_response = requests.get(
            aq_url,
            params=aq_params,
            timeout=15,
        )

        aq_response.raise_for_status()

        aq_payload = aq_response.json()
        aq_data = aq_payload.get("current", {})

        aqi = aq_data.get("us_aqi")
        timestamp = aq_data.get("time")

        if aqi is None or timestamp is None:
            return None

        pollutants = {
            "pm25": aq_data.get("pm2_5"),
            "pm10": aq_data.get("pm10"),
            "o3": aq_data.get("ozone"),
            "no2": aq_data.get("nitrogen_dioxide"),
            "so2": aq_data.get("sulphur_dioxide"),
            "co": aq_data.get("carbon_monoxide"),
        }

        # ----------------------------------------------------
        # WEATHER
        # ----------------------------------------------------

        weather_response = requests.get(
            weather_url,
            params=weather_params,
            timeout=15,
        )

        weather_response.raise_for_status()

        weather_payload = weather_response.json()
        weather_data = weather_payload.get("current", {})

        weather = {
            "temperature": weather_data.get(
                "temperature_2m"
            ),
            "humidity": weather_data.get(
                "relative_humidity_2m"
            ),
            "pressure": weather_data.get(
                "surface_pressure"
            ),
            "wind": weather_data.get(
                "wind_speed_10m"
            ),
        }

        return {
            "aqi": float(aqi),
            "timestamp": pd.to_datetime(timestamp),
            "pollutants": pollutants,
            "weather": weather,
        }

    except Exception as exc:

        print(
            f"Live Open-Meteo request failed: {exc}"
        )

        return None


# ============================================================
# LIVE FEATURE ENGINEERING
# ============================================================

def build_live_features(live, history):
    """
    Build the same 29 model features used during training,
    using the current live AQI plus historical AQI lags.
    """

    live_timestamp = pd.to_datetime(
        live["timestamp"]
    )

    current_aqi = float(live["aqi"])

    # --------------------------------------------------------
    # Keep only historical observations before live timestamp
    # --------------------------------------------------------

    history = history[
        history["timestamp"] < live_timestamp
    ].copy()

    if len(history) < 24:
        raise ValueError(
            "At least 24 historical AQI observations are "
            "required for live prediction."
        )

    # --------------------------------------------------------
    # Add current live AQI
    # --------------------------------------------------------

    current_row = pd.DataFrame(
        [
            {
                "timestamp": live_timestamp,
                "aqi": current_aqi,
            }
        ]
    )

    series_df = pd.concat(
        [
            history[["timestamp", "aqi"]],
            current_row,
        ],
        ignore_index=True,
    )

    series_df = (
        series_df
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    values = series_df["aqi"]

    # --------------------------------------------------------
    # Helper for exact hourly lag
    # --------------------------------------------------------

    def get_lag(hours):

        target_timestamp = (
            live_timestamp
            - pd.Timedelta(hours=hours)
        )

        matching = series_df[
            series_df["timestamp"]
            == target_timestamp
        ]

        if not matching.empty:
            return float(
                matching.iloc[-1]["aqi"]
            )

        # Fallback to positional lag
        if len(values) > hours:
            return float(
                values.iloc[-(hours + 1)]
            )

        raise ValueError(
            f"Could not calculate {hours}-hour AQI lag."
        )

    # --------------------------------------------------------
    # AQI lags
    # --------------------------------------------------------

    lag1 = get_lag(1)
    lag3 = get_lag(3)
    lag6 = get_lag(6)
    lag12 = get_lag(12)
    lag24 = get_lag(24)

    # --------------------------------------------------------
    # Rolling features
    # --------------------------------------------------------

    roll3 = float(
        values.tail(3).mean()
    )

    roll6 = float(
        values.tail(6).mean()
    )

    roll12 = float(
        values.tail(12).mean()
    )

    roll24 = float(
        values.tail(24).mean()
    )

    std6 = float(
        values.tail(6).std()
    )

    std24 = float(
        values.tail(24).std()
    )

    # --------------------------------------------------------
    # Live values
    # --------------------------------------------------------

    pollutants = live.get(
        "pollutants",
        {}
    )

    weather = live.get(
        "weather",
        {}
    )

    # --------------------------------------------------------
    # Model feature vector
    # --------------------------------------------------------

    feature_row = {

        "hour": live_timestamp.hour,

        "day": live_timestamp.day,

        "month": live_timestamp.month,

        "day_of_week": (
            live_timestamp.dayofweek
        ),

        "is_weekend": int(
            live_timestamp.dayofweek >= 5
        ),

        "aqi_lag_1h": lag1,

        "aqi_lag_3h": lag3,

        "aqi_lag_6h": lag6,

        "aqi_lag_12h": lag12,

        "aqi_lag_24h": lag24,

        "aqi_roll_mean_3h": roll3,

        "aqi_roll_mean_6h": roll6,

        "aqi_roll_mean_12h": roll12,

        "aqi_roll_mean_24h": roll24,

        "aqi_roll_std_6h": std6,

        "aqi_roll_std_24h": std24,

        "aqi_change_1h": (
            current_aqi - lag1
        ),

        "aqi_change_6h": (
            current_aqi - lag6
        ),

        "aqi_change_24h": (
            current_aqi - lag24
        ),

        "pm25": float(
            pollutants.get("pm25") or 0
        ),

        "pm10": float(
            pollutants.get("pm10") or 0
        ),

        "o3": float(
            pollutants.get("o3") or 0
        ),

        "no2": float(
            pollutants.get("no2") or 0
        ),

        "so2": float(
            pollutants.get("so2") or 0
        ),

        "co": float(
            pollutants.get("co") or 0
        ),

        "temperature": float(
            weather.get("temperature") or 0
        ),

        "humidity": float(
            weather.get("humidity") or 0
        ),

        "pressure": float(
            weather.get("pressure") or 0
        ),

        "wind": float(
            weather.get("wind") or 0
        ),
    }

    # --------------------------------------------------------
    # Validate features
    # --------------------------------------------------------

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in feature_row
    ]

    if missing:
        raise ValueError(
            f"Missing model features: {missing}"
        )

    X = pd.DataFrame(
        [
            [
                feature_row[column]
                for column in FEATURE_COLUMNS
            ]
        ],
        columns=FEATURE_COLUMNS,
    )

    return X


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "Pearls AQI Predictor API",
        "model": MODEL_NAME,
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.get("/predict")
def predict():

    # --------------------------------------------------------
    # Get live AQI
    # --------------------------------------------------------

    live = get_live_aqi()

    if live is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Live AQI data is currently "
                "unavailable."
            ),
        )

    # --------------------------------------------------------
    # Historical data
    # --------------------------------------------------------

    try:

        history = get_historical_aqi()

        X = build_live_features(
            live,
            history,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    try:

        prediction = float(
            model.predict(X)[0]
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Model prediction failed: {exc}"
            ),
        )

    prediction = max(
        0.0,
        prediction,
    )

    # --------------------------------------------------------
    # Prediction timestamp
    # --------------------------------------------------------

    live_timestamp = pd.to_datetime(
        live["timestamp"]
    )

    prediction_time = (
        live_timestamp
        + pd.Timedelta(hours=1)
    )

    current_aqi = float(
        live["aqi"]
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {

        "status": "success",

        "timestamp": (
            live_timestamp.isoformat()
        ),

        "current_aqi": round(
            current_aqi,
            2,
        ),

        "current_category": (
            get_aqi_category(
                current_aqi
            )
        ),

        "model_observation_timestamp": (
            live_timestamp.isoformat()
        ),

        "model_observation_aqi": round(
            current_aqi,
            2,
        ),

        "prediction_for": (
            prediction_time.isoformat()
        ),

        "predicted_aqi": round(
            prediction,
            2,
        ),

        "category": (
            get_aqi_category(
                prediction
            )
        ),

        "model": MODEL_NAME,

        "prediction_horizon": "1 hour",

        "data_source": (
            "Open-Meteo current AQI and weather "
            "plus SQLite historical AQI features"
        ),

        "model_metrics": MODEL_METRICS,

        "pollutants": live.get(
            "pollutants",
            {},
        ),

        "weather": live.get(
            "weather",
            {},
        ),
    }