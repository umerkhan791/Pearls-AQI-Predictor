"""
Real historical AQI backfill for Karachi.

Sources:
- Open-Meteo Air Quality API: historical pollutants + US AQI
- Open-Meteo Historical Weather API: historical weather

The AQICN API remains our live/current AQI source in
feature_pipeline.py.

Run:
    python pipelines/backfill.py
    python pipelines/backfill.py --days 90
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import requests
import pandas as pd
from loguru import logger

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CITY
from database import init_db, get_connection, load_raw_readings
from pipelines.feature_pipeline import engineer_features
from database import save_features


# ---------------------------------------------------------------------------
# Karachi configuration
# ---------------------------------------------------------------------------

LATITUDE = 24.8607
LONGITUDE = 67.0011
TIMEZONE = "Asia/Karachi"

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"


# ---------------------------------------------------------------------------
# Fetch historical air-quality data
# ---------------------------------------------------------------------------

def fetch_historical_air_quality(start_date: str, end_date: str) -> pd.DataFrame:
    logger.info(
        f"Fetching real historical air-quality data: "
        f"{start_date} → {end_date}"
    )

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": (
            "pm2_5,"
            "pm10,"
            "carbon_monoxide,"
            "nitrogen_dioxide,"
            "sulphur_dioxide,"
            "ozone,"
            "us_aqi"
        ),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": TIMEZONE,
    }

    response = requests.get(
        AIR_QUALITY_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:
        raise RuntimeError("Air-quality API returned no hourly data.")

    hourly = data["hourly"]

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "pm25": hourly["pm2_5"],
        "pm10": hourly["pm10"],
        "co": hourly["carbon_monoxide"],
        "no2": hourly["nitrogen_dioxide"],
        "so2": hourly["sulphur_dioxide"],
        "o3": hourly["ozone"],
        "aqi": hourly["us_aqi"],
    })

    logger.info(f"Received {len(df)} real air-quality records.")

    return df


# ---------------------------------------------------------------------------
# Fetch historical weather data
# ---------------------------------------------------------------------------

def fetch_historical_weather(start_date: str, end_date: str) -> pd.DataFrame:
    logger.info(
        f"Fetching real historical weather data: "
        f"{start_date} → {end_date}"
    )

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "surface_pressure,"
            "wind_speed_10m"
        ),
        "timezone": TIMEZONE,
    }

    response = requests.get(
        WEATHER_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    if "hourly" not in data:
        raise RuntimeError("Weather API returned no hourly data.")

    hourly = data["hourly"]

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"]),
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "pressure": hourly["surface_pressure"],
        "wind": hourly["wind_speed_10m"],
    })

    logger.info(f"Received {len(df)} real weather records.")

    return df


# ---------------------------------------------------------------------------
# Build complete historical dataset
# ---------------------------------------------------------------------------

def build_historical_dataset(days: int = 90) -> pd.DataFrame:

    end_date = datetime.now().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=days)

    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    air_df = fetch_historical_air_quality(
        start_str,
        end_str
    )

    weather_df = fetch_historical_weather(
        start_str,
        end_str
    )

    logger.info("Merging air-quality and weather datasets...")

    df = pd.merge(
        air_df,
        weather_df,
        on="timestamp",
        how="inner"
    )

    df["city"] = CITY
    df["dominant_pol"] = "pm25"

    # Remove rows without a target AQI.
    df = df.dropna(subset=["aqi"])

    # Remove duplicate timestamps.
    df = df.drop_duplicates(
        subset=["timestamp", "city"]
    )

    df = df.sort_values("timestamp")

    logger.info(
        f"Final merged dataset: {len(df)} hourly records"
    )

    logger.info(
        f"AQI range: "
        f"{df['aqi'].min():.1f} – {df['aqi'].max():.1f}"
    )

    logger.info(
        f"AQI mean: {df['aqi'].mean():.1f}"
    )

    return df


# ---------------------------------------------------------------------------
# Clear old synthetic data
# ---------------------------------------------------------------------------

def clear_old_data():

    logger.warning(
        "Removing existing raw readings and engineered features..."
    )

    conn = get_connection()

    try:
        conn.execute("DELETE FROM features")
        conn.execute("DELETE FROM raw_readings")
        conn.commit()

    finally:
        conn.close()

    logger.info("Old training data removed.")


# ---------------------------------------------------------------------------
# Save raw data
# ---------------------------------------------------------------------------

def save_raw_data(df: pd.DataFrame):

    logger.info("Saving real historical readings to database...")

    save_df = df.copy()

    save_df["timestamp"] = (
        pd.to_datetime(save_df["timestamp"])
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    conn = get_connection()

    try:

        save_df.to_sql(
            "raw_readings_tmp",
            conn,
            if_exists="replace",
            index=False
        )

        conn.execute("""
            INSERT OR IGNORE INTO raw_readings
            (
                timestamp,
                city,
                aqi,
                pm25,
                pm10,
                o3,
                no2,
                so2,
                co,
                temperature,
                humidity,
                pressure,
                wind,
                dominant_pol
            )
            SELECT
                timestamp,
                city,
                aqi,
                pm25,
                pm10,
                o3,
                no2,
                so2,
                co,
                temperature,
                humidity,
                pressure,
                wind,
                dominant_pol
            FROM raw_readings_tmp
        """)

        conn.execute(
            "DROP TABLE IF EXISTS raw_readings_tmp"
        )

        conn.commit()

    finally:
        conn.close()

    logger.info(
        f"Saved {len(save_df)} real raw readings."
    )


# ---------------------------------------------------------------------------
# Engineer and save features
# ---------------------------------------------------------------------------

def build_features():

    logger.info("Loading real readings from database...")

    raw_df = load_raw_readings(
        CITY,
        limit=10000
    )

    if raw_df.empty:
        raise RuntimeError(
            "No raw data found. Cannot engineer features."
        )

    logger.info(
        f"Engineering features from {len(raw_df)} records..."
    )

    features_df = engineer_features(raw_df)

    if features_df.empty:
        raise RuntimeError(
            "Feature engineering produced no rows."
        )

    save_features(features_df)

    logger.info(
        f"Saved {len(features_df)} engineered feature rows."
    )

    return features_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_backfill(days: int = 90):

    logger.info("=" * 70)
    logger.info("REAL DATA AQI HISTORICAL BACKFILL")
    logger.info("=" * 70)

    init_db()

    if days > 92:
        raise ValueError(
            "Open-Meteo historical air-quality data supports "
            "up to approximately 92 past days per request. "
            "Use days <= 92."
        )

    # Fetch real API data FIRST.
    # This protects the existing database if the external APIs fail.
    df = build_historical_dataset(days)

    if df.empty:
        raise RuntimeError("No historical data was returned. Existing data was not changed.")

    # Only clear old data after the new dataset was fetched successfully.
    clear_old_data()

    # Save raw readings.
    save_raw_data(df)

    # Feature engineering.
    features_df = build_features()

    print("\n" + "=" * 70)
    print("REAL DATA BACKFILL COMPLETE")
    print("=" * 70)

    print(f"City:          {CITY}")
    print(f"Period:        {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"Latest data:   {df['timestamp'].max()}")
    print(f"Raw rows:      {len(df)}")
    print(f"Feature rows:  {len(features_df)}")
    print(
        f"AQI range:     "
        f"{df['aqi'].min():.1f} – {df['aqi'].max():.1f}"
    )
    print(
        f"AQI mean:       "
        f"{df['aqi'].mean():.1f}"
    )

    print("=" * 70)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Backfill real historical AQI/weather data"
    )

    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of historical days (maximum 92)"
    )

    args = parser.parse_args()

    run_backfill(args.days)