"""
PEARLS AQI PREDICTOR — FEATURE ENGINEERING PIPELINE
─────────────────────────────────────────────────────
Step 1 of the ML system.

What it does:
  1. Fetches current AQI + pollutant data from AQICN API
  2. Engineers time-based, lag, rolling, and change-rate features
  3. Stores everything in the SQLite feature store

IMPORTANT:
  All AQI-derived forecasting features are strictly historical.

  For a prediction at time t:
      - lag features use t-1, t-3, t-6, etc.
      - rolling features use values BEFORE t
      - change features use current and previous observations

Run manually:
    python pipelines/feature_pipeline.py

Scheduled:
    GitHub Actions runs this every hour
"""

import sys
import os
import requests
import pandas as pd
import numpy as np

from datetime import datetime
from pathlib import Path
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Allow imports from project root
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent.parent))


from config import (
    AQICN_BASE_URL,
    CITY,
    CITY_DISPLAY_NAME,
    POLLUTANT_COLS,
    WEATHER_COLS,
    LAG_HOURS,
    ROLLING_WINDOWS,
)

# config.py does not export the AQICN token, so read it from .env.
AQICN_TOKEN = (
    os.getenv("AQICN_TOKEN")
    or os.getenv("AQICN_API_TOKEN")
    or os.getenv("AQICN_API_KEY")
)

if not AQICN_TOKEN:
    raise RuntimeError(
        "AQICN token not found. Add AQICN_TOKEN=your_token to .env "
        "without committing the .env file."
    )

from database import (
    init_db,
    save_raw_reading,
    save_features,
    load_raw_readings,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. FETCH RAW DATA
# ─────────────────────────────────────────────────────────────────────────────

def fetch_current_aqi(city: str = CITY) -> dict | None:
    """
    Fetch current AQI and pollutant readings from AQICN API.

    Returns a flat dictionary ready to insert into the raw_readings table.
    """

    url = f"{AQICN_BASE_URL}/feed/{city}/?token={AQICN_TOKEN}"

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        data = response.json()

        if data["status"] != "ok":
            logger.error(
                f"AQICN returned status: {data['status']}"
            )
            return None

        d = data["data"]
        iaqi = d.get("iaqi", {})

        # Parse timestamp returned by AQICN.
        #
        # The timestamp string represents the local station time.
        # We keep the original timestamp format used by the project/database.
        ts_str = d["time"]["s"]
        timestamp = datetime.strptime(
            ts_str,
            "%Y-%m-%d %H:%M:%S"
        )

        record = {
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "city": city,

            # AQI
            "aqi": float(
                d.get("aqi", np.nan)
            ),

            # Pollutants
            "pm25": float(
                iaqi.get("pm25", {}).get("v", np.nan)
            ),
            "pm10": float(
                iaqi.get("pm10", {}).get("v", np.nan)
            ),
            "o3": float(
                iaqi.get("o3", {}).get("v", np.nan)
            ),
            "no2": float(
                iaqi.get("no2", {}).get("v", np.nan)
            ),
            "so2": float(
                iaqi.get("so2", {}).get("v", np.nan)
            ),
            "co": float(
                iaqi.get("co", {}).get("v", np.nan)
            ),

            # Weather
            "temperature": float(
                iaqi.get("t", {}).get("v", np.nan)
            ),
            "humidity": float(
                iaqi.get("h", {}).get("v", np.nan)
            ),
            "pressure": float(
                iaqi.get("p", {}).get("v", np.nan)
            ),
            "wind": float(
                iaqi.get("w", {}).get("v", np.nan)
            ),

            "dominant_pol": d.get("dominentpol", ""),
        }

        logger.info(
            f"Fetched AQI={record['aqi']} "
            f"for {city} at {record['timestamp']}"
        )

        return record

    except requests.RequestException as e:
        logger.error(
            f"API request failed: {e}"
        )
        return None

    except Exception as e:
        logger.error(
            f"Unexpected error fetching data: {e}"
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 1B. AQICN FORECAST
# ─────────────────────────────────────────────────────────────────────────────

def fetch_forecast_aqi(city: str = CITY) -> pd.DataFrame:
    """
    Fetch the AQICN built-in forecast.

    Used to supplement our ML predictions.

    Returns a DataFrame with columns such as:

        day
        pm25_avg
        pm25_max
        pm25_min
        pm10_avg
        pm10_max
        pm10_min
    """

    url = f"{AQICN_BASE_URL}/feed/{city}/?token={AQICN_TOKEN}"

    try:
        response = requests.get(
            url,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()

        if data["status"] != "ok":
            return pd.DataFrame()

        forecast = (
            data["data"]
            .get("forecast", {})
            .get("daily", {})
        )

        all_days = {}

        # Merge all pollutant forecasts by day.
        for pol, days_list in forecast.items():

            for entry in days_list:

                day = entry["day"]

                if day not in all_days:
                    all_days[day] = {
                        "day": day
                    }

                all_days[day][f"{pol}_avg"] = (
                    entry.get("avg", np.nan)
                )

                all_days[day][f"{pol}_max"] = (
                    entry.get("max", np.nan)
                )

                all_days[day][f"{pol}_min"] = (
                    entry.get("min", np.nan)
                )

        df = pd.DataFrame(
            list(all_days.values())
        )

        if df.empty:
            return df

        df["day"] = pd.to_datetime(
            df["day"]
        )

        df = (
            df.sort_values("day")
            .reset_index(drop=True)
        )

        logger.info(
            f"Fetched {len(df)} forecast days from AQICN"
        )

        return df

    except Exception as e:
        logger.error(
            f"Forecast fetch failed: {e}"
        )
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Takes a DataFrame of raw readings sorted by timestamp and computes:

      - Time-based features
      - AQI lag features
      - STRICTLY HISTORICAL AQI rolling features
      - AQI change-rate features

    Leakage-safe definition:

        Feature row at time t
                   ↓
        only information available at or before t
                   ↓
        target = AQI at t+1

    Rolling features deliberately use shift(1), meaning the current
    AQI value is NOT included in the rolling window.

    NaN values are NOT filled using the entire dataset's median.
    This prevents future observations from influencing earlier rows.

    The training pipeline should drop rows that do not have enough
    historical information.
    """

    df = df.copy()

    # ── Timestamp ────────────────────────────────────────────────────────────

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )


    # ── Time features ────────────────────────────────────────────────────────

    df["hour"] = (
        df["timestamp"].dt.hour
    )

    df["day"] = (
        df["timestamp"].dt.day
    )

    df["month"] = (
        df["timestamp"].dt.month
    )

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    # Monday = 0
    # Sunday = 6
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)


    # ── AQI lag features ─────────────────────────────────────────────────────

    # These are already strictly historical.
    #
    # At time t:
    #
    # aqi_lag_1h  = AQI at t-1
    # aqi_lag_3h  = AQI at t-3
    # aqi_lag_6h  = AQI at t-6
    # etc.

    for lag in LAG_HOURS:

        df[f"aqi_lag_{lag}h"] = (
            df["aqi"].shift(lag)
        )


    # ── STRICTLY HISTORICAL ROLLING FEATURES ─────────────────────────────────

    # IMPORTANT:
    #
    # We first shift AQI by one row.
    #
    # previous_aqi at row t = AQI at t-1
    #
    # Therefore:
    #
    # rolling_mean_3h at t
    #     = mean(AQI at t-1, t-2, t-3)
    #
    # It does NOT contain AQI at t.
    #
    # This is the strictest leakage-safe formulation for our
    # next-hour forecasting experiment.

    previous_aqi = (
        df["aqi"].shift(1)
    )


    # Rolling means

    for window in ROLLING_WINDOWS:

        df[f"aqi_roll_mean_{window}h"] = (
            previous_aqi
            .rolling(
                window=window,
                min_periods=window
            )
            .mean()
        )


    # Rolling standard deviation

    for window in [6, 24]:

        df[f"aqi_roll_std_{window}h"] = (
            previous_aqi
            .rolling(
                window=window,
                min_periods=window
            )
            .std()
        )


    # ── AQI change-rate features ────────────────────────────────────────────

    # These use the current observation and earlier observations.
    #
    # For example:
    #
    # change_1h at t = AQI(t) - AQI(t-1)
    #
    # This is valid if current AQI is available at prediction time.

    df["aqi_change_1h"] = (
        df["aqi"].diff(1)
    )

    df["aqi_change_6h"] = (
        df["aqi"].diff(6)
    )

    df["aqi_change_24h"] = (
        df["aqi"].diff(24)
    )


    # ── Weather column names ─────────────────────────────────────────────────

    rename_map = {
        "temperature": "temperature",
        "humidity": "humidity",
        "pressure": "pressure",
        "wind": "wind",
    }

    # Already named correctly by fetch_current_aqi.
    # Kept here for compatibility with the original pipeline.
    df = df.rename(
        columns=rename_map
    )


    # ── DO NOT USE FULL-DATASET MEDIAN IMPUTATION ────────────────────────────
    #
    # The previous version did:
    #
    #     df[col] = df[col].fillna(df[col].median())
    #
    # That calculates the median using the entire dataset, including
    # observations that occur AFTER the row being predicted.
    #
    # For a strict time-series experiment this is undesirable.
    #
    # We therefore leave missing values as NaN.
    #
    # train_model.py removes incomplete rows before training.
    #
    # Live predictions use the latest row, which has sufficient history.


    # ── Select database schema columns ───────────────────────────────────────

    feature_cols = [

        # Identity
        "timestamp",
        "city",
        "aqi",

        # Time
        "hour",
        "day",
        "month",
        "day_of_week",
        "is_weekend",

        # AQI lags
        "aqi_lag_1h",
        "aqi_lag_3h",
        "aqi_lag_6h",
        "aqi_lag_12h",
        "aqi_lag_24h",

        # AQI rolling means
        "aqi_roll_mean_3h",
        "aqi_roll_mean_6h",
        "aqi_roll_mean_12h",
        "aqi_roll_mean_24h",

        # AQI rolling standard deviation
        "aqi_roll_std_6h",
        "aqi_roll_std_24h",

        # AQI changes
        "aqi_change_1h",
        "aqi_change_6h",
        "aqi_change_24h",

        # Pollutants
        "pm25",
        "pm10",
        "o3",
        "no2",
        "so2",
        "co",

        # Weather
        "temperature",
        "humidity",
        "pressure",
        "wind",
    ]


    # Only keep columns that exist.

    available = [
        c
        for c in feature_cols
        if c in df.columns
    ]

    return df[available]


# ─────────────────────────────────────────────────────────────────────────────
# 3. MAIN PIPELINE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def run_feature_pipeline(
    city: str = CITY
):
    """
    Full feature pipeline:

        fetch
          ↓
        save raw
          ↓
        reload history
          ↓
        engineer features
          ↓
        save features
    """

    logger.info(
        f"=== Feature Pipeline START for {city} ==="
    )


    # ── Step 1: Fetch current reading ────────────────────────────────────────

    record = fetch_current_aqi(
        city
    )

    if record is None:

        logger.error(
            "Could not fetch data. Aborting pipeline."
        )

        return False


    # ── Step 2: Save raw reading ─────────────────────────────────────────────

    save_raw_reading(
        record
    )


    # ── Step 3: Load recent raw history ──────────────────────────────────────
    #
    # 200 hourly observations gives enough history for the 24-hour
    # lag/rolling features used by the model.

    raw_df = load_raw_readings(
        city,
        limit=200
    )

    if raw_df.empty:

        logger.warning(
            "No historical data found. "
            "Feature engineering needs history."
        )

        return False


    # ── Step 4: Engineer features ────────────────────────────────────────────

    features_df = engineer_features(
        raw_df
    )


    # ── Step 5: Save features ────────────────────────────────────────────────

    save_features(
        features_df
    )


    logger.info(
        f"=== Feature Pipeline DONE. "
        f"{len(features_df)} rows saved. ==="
    )

    return True


# ─────────────────────────────────────────────────────────────────────────────
# 4. CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    init_db()

    success = run_feature_pipeline()

    if success:

        print(
            "Feature pipeline completed successfully."
        )

    else:

        print(
            "Feature pipeline failed. "
            "Check logs."
        )

        sys.exit(1)