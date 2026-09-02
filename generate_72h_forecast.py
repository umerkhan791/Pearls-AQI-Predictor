"""
Pearls AQI Predictor
72-Hour Recursive Forecast Generator

The first forecast hour uses the exact latest observed feature row.
Therefore:

    observation at t  ->  prediction at t+1

The remaining 71 hours are generated recursively using previous
predictions for the AQI lag/rolling features. Future pollutant and
weather inputs are carried forward from the latest observation.
"""

import os
import sqlite3

import joblib
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "aqi_model.pkl"
DB_PATH = os.path.join("data", "feature_store.db")
OUTPUT_PATH = os.path.join("data", "forecast_72h.csv")

CITY = "karachi"


# ============================================================
# MODEL FEATURES
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

def aqi_category(aqi):

    if aqi <= 50:
        return "Good"

    if aqi <= 100:
        return "Moderate"

    if aqi <= 150:
        return "Unhealthy for Sensitive Groups"

    if aqi <= 200:
        return "Unhealthy"

    if aqi <= 300:
        return "Very Unhealthy"

    return "Hazardous"


# ============================================================
# LOAD HISTORICAL FEATURES
# ============================================================

def load_history():

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database not found: {DB_PATH}"
        )

    connection = sqlite3.connect(DB_PATH)

    try:

        history = pd.read_sql_query(
            """
            SELECT *
            FROM features
            WHERE city = ?
            ORDER BY timestamp ASC
            """,
            connection,
            params=(CITY,),
        )

    finally:

        connection.close()

    if history.empty:
        raise RuntimeError(
            f"No feature data found for city: {CITY}"
        )

    history["timestamp"] = pd.to_datetime(
        history["timestamp"],
        errors="coerce",
    )

    history = history.dropna(
        subset=["timestamp"]
    )

    history = history.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return history


# ============================================================
# MAIN FORECAST
# ============================================================

def main():

    print("=" * 65)
    print("Pearls AQI Predictor - 72 Hour Forecast")
    print("=" * 65)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = joblib.load(
        MODEL_PATH
    )

    print(
        f"Model: {type(model).__name__}"
    )

    # --------------------------------------------------------
    # Load historical feature data
    # --------------------------------------------------------

    print("\nLoading feature data...")

    history = load_history()

    print(
        f"Historical rows: {len(history)}"
    )

    # --------------------------------------------------------
    # Latest observation
    # --------------------------------------------------------

    latest = history.iloc[-1].copy()

    latest_timestamp = pd.Timestamp(
        latest["timestamp"]
    )

    current_aqi = float(
        latest["aqi"]
    )

    print(
        f"Latest observation: {latest_timestamp}"
    )

    print(
        f"Current AQI: {current_aqi:.2f}"
    )

    # --------------------------------------------------------
    # Historical AQI values
    #
    # These are used to construct lag and rolling features.
    # --------------------------------------------------------

    aqi_history = (
        history["aqi"]
        .astype(float)
        .tolist()
    )

    forecasts = []

    # ========================================================
    # GENERATE 72 HOURS
    # ========================================================

    for step in range(1, 73):

        forecast_time = (
            latest_timestamp
            + pd.Timedelta(hours=step)
        )

        # ====================================================
        # FIRST HOUR
        #
        # CRITICAL:
        #
        # Do NOT recalculate the feature row.
        #
        # Use the EXACT latest observation at time t.
        #
        # This guarantees:
        #
        # t -> t+1
        #
        # and therefore matches the FastAPI prediction.
        # ====================================================

        if step == 1:

            values = latest.copy()

        # ====================================================
        # HOURS 2-72
        #
        # Recursively update time and AQI-derived features.
        # ====================================================

        else:

            values = latest.copy()

            # ------------------------------------------------
            # Calendar features
            # ------------------------------------------------

            values["hour"] = forecast_time.hour

            values["day"] = forecast_time.day

            values["month"] = forecast_time.month

            values["day_of_week"] = (
                forecast_time.dayofweek
            )

            values["is_weekend"] = int(
                forecast_time.dayofweek >= 5
            )

            # ------------------------------------------------
            # AQI lag helper
            # ------------------------------------------------

            def get_lag(hours):

                if len(aqi_history) >= hours:

                    return float(
                        aqi_history[-hours]
                    )

                return float(
                    aqi_history[0]
                )

            # ------------------------------------------------
            # AQI lags
            # ------------------------------------------------

            values["aqi_lag_1h"] = get_lag(1)

            values["aqi_lag_3h"] = get_lag(3)

            values["aqi_lag_6h"] = get_lag(6)

            values["aqi_lag_12h"] = get_lag(12)

            values["aqi_lag_24h"] = get_lag(24)

            # ------------------------------------------------
            # Rolling means
            # ------------------------------------------------

            def get_mean(hours):

                recent = aqi_history[-hours:]

                return float(
                    np.mean(recent)
                )

            values["aqi_roll_mean_3h"] = (
                get_mean(3)
            )

            values["aqi_roll_mean_6h"] = (
                get_mean(6)
            )

            values["aqi_roll_mean_12h"] = (
                get_mean(12)
            )

            values["aqi_roll_mean_24h"] = (
                get_mean(24)
            )

            # ------------------------------------------------
            # Rolling standard deviations
            # ------------------------------------------------

            def get_std(hours):

                recent = aqi_history[-hours:]

                return float(
                    np.std(recent)
                )

            values["aqi_roll_std_6h"] = (
                get_std(6)
            )

            values["aqi_roll_std_24h"] = (
                get_std(24)
            )

            # ------------------------------------------------
            # AQI changes
            # ------------------------------------------------

            if len(aqi_history) >= 2:

                values["aqi_change_1h"] = (
                    aqi_history[-1]
                    - aqi_history[-2]
                )

            else:

                values["aqi_change_1h"] = 0.0

            if len(aqi_history) >= 7:

                values["aqi_change_6h"] = (
                    aqi_history[-1]
                    - aqi_history[-7]
                )

            else:

                values["aqi_change_6h"] = 0.0

            if len(aqi_history) >= 25:

                values["aqi_change_24h"] = (
                    aqi_history[-1]
                    - aqi_history[-25]
                )

            else:

                values["aqi_change_24h"] = 0.0

        # ====================================================
        # CREATE MODEL INPUT
        # ====================================================

        X = pd.DataFrame(
            [
                [
                    values[column]
                    for column in FEATURE_COLUMNS
                ]
            ],
            columns=FEATURE_COLUMNS,
        )

        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        prediction = float(
            model.predict(X)[0]
        )

        # AQI cannot be negative.
        prediction = max(
            0.0,
            prediction
        )

        category = aqi_category(
            prediction
        )

        # ====================================================
        # SAVE FORECAST
        # ====================================================

        forecasts.append(
            {
                # Use the API/dashboard naming convention.
                "predicted_for": forecast_time,
                "predicted_aqi": round(prediction, 2),
                "category": category,
            }
        )

        # ====================================================
        # RECURSIVE UPDATE
        #
        # This prediction becomes the latest AQI value for
        # constructing the next future feature row.
        # ====================================================

        aqi_history.append(
            prediction
        )

    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    forecast_df = pd.DataFrame(
        forecasts
    )

    # ========================================================
    # VALIDATE FORECAST OUTPUT
    # ========================================================

    if len(forecast_df) != 72:
        raise RuntimeError(
            f"Expected exactly 72 forecast rows, got {len(forecast_df)}."
        )

    if forecast_df["predicted_for"].duplicated().any():
        raise RuntimeError(
            "Forecast contains duplicate timestamps."
        )

    expected_times = pd.date_range(
        start=latest_timestamp + pd.Timedelta(hours=1),
        periods=72,
        freq="h",
    )

    actual_times = pd.DatetimeIndex(
        pd.to_datetime(
            forecast_df["predicted_for"],
            errors="raise",
        )
    )

    # Compare timestamp values, not the DataFrame Series/index metadata.
    # This avoids false failures caused by different index types/names.
    if actual_times.tolist() != expected_times.tolist():
        raise RuntimeError(
            "Forecast timestamps are not a continuous 72-hour sequence."
        )

    if forecast_df["predicted_aqi"].isna().any():
        raise RuntimeError(
            "Forecast contains missing AQI predictions."
        )

    # ========================================================
    # SAVE CSV
    # =======================================================

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True,
    )

    forecast_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    first_prediction = float(
        forecast_df.iloc[0]["predicted_aqi"]
    )

    last_prediction = float(
        forecast_df.iloc[-1]["predicted_aqi"]
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "=" * 65)
    print("FORECAST GENERATED SUCCESSFULLY")
    print("=" * 65)

    print(
        f"\nLatest observation:"
        f" {latest_timestamp}"
    )

    print(
        f"Current AQI:"
        f" {current_aqi:.2f}"
    )

    print(
        f"Forecast hours:"
        f" {len(forecast_df)}"
    )

    print(
        f"Forecast start:"
        f" {forecast_df.iloc[0]['predicted_for']}"
    )

    print(
        f"Forecast end:"
        f" {forecast_df.iloc[-1]['predicted_for']}"
    )

    print(
        f"First prediction:"
        f" {first_prediction:.2f}"
    )

    print(
        f"Final prediction:"
        f" {last_prediction:.2f}"
    )

    print(
        f"\nSaved to:"
        f" {OUTPUT_PATH}"
    )

    print("\nFirst 10 forecast hours:")

    print(
        forecast_df
        .head(10)
        .to_string(index=False)
    )

    print("\nLast 5 forecast hours:")

    print(
        forecast_df
        .tail(5)
        .to_string(index=False)
    )

    print("\n" + "=" * 65)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
