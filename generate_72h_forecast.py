from pathlib import Path

import pandas as pd

from api import (
    model,
    get_live_aqi,
    get_historical_aqi,
    build_live_features,
    get_aqi_category,
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "data" / "forecast_72h.csv"


def main():
    print("=" * 65)
    print("Pearls AQI Predictor - 72 Hour Forecast")
    print("=" * 65)

    print("\nLoading live observation...")
    live = get_live_aqi()

    if live is None:
        raise RuntimeError("Live Open-Meteo AQI data unavailable.")

    history = get_historical_aqi()

    live_timestamp = pd.to_datetime(live["timestamp"])
    current_aqi = float(live["aqi"])

    print(f"Live observation : {live_timestamp}")
    print(f"Current AQI      : {current_aqi:.2f}")
    print(f"Historical rows  : {len(history)}")

    # This history is progressively extended with predictions.
    # The API's exact feature-engineering function is used for
    # every forecast hour.
    working_history = pd.concat(
        [
            history,
            pd.DataFrame(
                [{"timestamp": live_timestamp, "aqi": current_aqi}]
            ),
        ],
        ignore_index=True,
    )

    forecasts = []

    for step in range(1, 73):

        prediction_time = (
            live_timestamp + pd.Timedelta(hours=step)
        )

        if step == 1:
            source_timestamp = live_timestamp
            source_aqi = current_aqi
        else:
            source_timestamp = forecasts[-1]["predicted_for"]
            source_aqi = float(
                forecasts[-1]["predicted_aqi"]
            )

        source = {
            "timestamp": source_timestamp,
            "aqi": source_aqi,
            "pollutants": live.get("pollutants", {}).copy(),
            "weather": live.get("weather", {}).copy(),
        }

        # CRITICAL:
        # This is the same function used by /predict.
        X = build_live_features(
            source,
            working_history,
        )

        prediction = float(
            model.predict(X)[0]
        )

        prediction = max(0.0, prediction)

        forecasts.append(
            {
                "predicted_for": prediction_time,
                "predicted_aqi": round(prediction, 2),
                "category": get_aqi_category(prediction),
            }
        )

        # Add this prediction to the history so the next
        # recursive hour gets updated lag/rolling features.
        working_history = pd.concat(
            [
                working_history,
                pd.DataFrame(
                    [
                        {
                            "timestamp": prediction_time,
                            "aqi": prediction,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    forecast_df = pd.DataFrame(forecasts)

    # ========================================================
    # VALIDATION
    # ========================================================

    if len(forecast_df) != 72:
        raise RuntimeError(
            f"Expected 72 rows, got {len(forecast_df)}"
        )

    if forecast_df["predicted_for"].duplicated().any():
        raise RuntimeError(
            "Duplicate forecast timestamps detected."
        )

    expected_times = pd.date_range(
        start=live_timestamp + pd.Timedelta(hours=1),
        periods=72,
        freq="h",
    )

    actual_times = pd.DatetimeIndex(
        forecast_df["predicted_for"]
    )

    if actual_times.tolist() != expected_times.tolist():
        raise RuntimeError(
            "Forecast timestamps are not a continuous "
            "72-hour sequence."
        )

    if forecast_df["predicted_aqi"].isna().any():
        raise RuntimeError(
            "Forecast contains missing AQI values."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\n" + "=" * 65)
    print("FORECAST GENERATED SUCCESSFULLY")
    print("=" * 65)
    print(f"Current AQI      : {current_aqi:.2f}")
    print(f"Forecast rows    : {len(forecast_df)}")
    print(
        f"Forecast start   : "
        f"{forecast_df.iloc[0]['predicted_for']}"
    )
    print(
        f"Forecast end     : "
        f"{forecast_df.iloc[-1]['predicted_for']}"
    )
    print(
        f"First prediction : "
        f"{forecast_df.iloc[0]['predicted_aqi']:.2f}"
    )
    print(
        f"Final prediction : "
        f"{forecast_df.iloc[-1]['predicted_aqi']:.2f}"
    )
    print(f"Saved            : {OUTPUT_PATH}")

    print("\nFirst 5 forecast hours:")
    print(
        forecast_df.head(5).to_string(index=False)
    )


if __name__ == "__main__":
    main()
