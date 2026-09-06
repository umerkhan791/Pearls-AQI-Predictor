"""
Pearls AQI Predictor - Multi-Horizon Evaluation

Evaluates separate supervised Random Forest models at:
    +24 hours
    +48 hours
    +72 hours

Uses the same feature set and chronological 80/20 methodology
as the main training pipeline.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from database import load_features
from config import CITY, REPORTS_DIR, RANDOM_SEED, TRAIN_TEST_SPLIT


FEATURE_COLS = [
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


def main():

    print("=" * 70)
    print("MULTI-HORIZON AQI EVALUATION")
    print("=" * 70)

    df = load_features(
        CITY,
        limit=10000
    )

    if df.empty:
        raise RuntimeError("No feature data available.")

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = (
        df.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    missing = [
        c for c in FEATURE_COLS
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"Missing features: {missing}"
        )

    df = df.dropna(
        subset=FEATURE_COLS + ["aqi"]
    ).reset_index(drop=True)

    split_index = int(
        len(df) * TRAIN_TEST_SPLIT
    )

    results = {}

    for horizon in [24, 48, 72]:

        print()
        print(
            f"Evaluating +{horizon}h..."
        )

        data = df.copy()

        data["target"] = (
            data["aqi"].shift(-horizon)
        )

        data = data.dropna(
            subset=["target"]
        ).reset_index(drop=True)

        X = data[FEATURE_COLS]
        y = data["target"]

        split = min(
            split_index,
            len(data) - 1
        )

        X_train = X.iloc[:split]
        X_test = X.iloc[split:]

        y_train = y.iloc[:split]
        y_test = y.iloc[split:]

        model = RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        results[f"{horizon}h"] = {
            "horizon_hours": horizon,
            "model": "RandomForestRegressor",
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "mae": round(float(mae), 4),
            "rmse": round(float(rmse), 4),
            "r2": round(float(r2), 4),
        }

        print(
            f"+{horizon:>2}h | "
            f"MAE={mae:.4f} | "
            f"RMSE={rmse:.4f} | "
            f"R2={r2:.4f} | "
            f"test={len(X_test)}"
        )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        REPORTS_DIR /
        "horizon_metrics.json"
    )

    report = {
        "city": CITY,
        "evaluation_type": (
            "Horizon-specific supervised "
            "Random Forest evaluation"
        ),
        "methodology": (
            "Separate AQI target shifted by "
            "24, 48 and 72 hours with "
            "chronological 80/20 train-test split."
        ),
        "horizons": results,
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("HORIZON EVALUATION COMPLETE")
    print("=" * 70)
    print(
        f"Saved: {output_path}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
