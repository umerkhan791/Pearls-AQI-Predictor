"""
Pearls AQI Predictor
Next-hour AQI prediction using the trained Random Forest model.
"""

import os
import joblib
import pandas as pd
import hopsworks

from dotenv import load_dotenv


# =====================================================================
# CONFIGURATION
# =====================================================================

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"
HOPSWORKS_PROJECT = "pearls_aqi_predictors"

FEATURE_GROUP_NAME = "karachi_aqi_features"
FEATURE_GROUP_VERSION = 4

MODEL_PATH = "aqi_model.pkl"


# =====================================================================
# MODEL FEATURES
# =====================================================================

MODEL_FEATURES = [
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


# =====================================================================
# VALIDATION
# =====================================================================

if not HOPSWORKS_API_KEY:
    raise RuntimeError(
        "HOPSWORKS_API_KEY is missing from .env"
    )


if not os.path.exists(MODEL_PATH):
    raise RuntimeError(
        f"Model file not found: {MODEL_PATH}"
    )


# =====================================================================
# START
# =====================================================================

print("=" * 70)
print("PEARLS AQI PREDICTOR - NEXT-HOUR PREDICTION")
print("=" * 70)


# =====================================================================
# LOAD MODEL
# =====================================================================

print("\nLoading trained model...")

model = joblib.load(MODEL_PATH)

print(
    f"Model: {type(model).__name__}"
)

print(
    f"Input features: {model.n_features_in_}"
)


# Make sure the saved model is the expected model.
if model.n_features_in_ != len(MODEL_FEATURES):
    raise RuntimeError(
        "Model feature count does not match expected feature count."
    )


# =====================================================================
# CONNECT TO HOPSWORKS
# =====================================================================

print("\nConnecting to Hopsworks...")

project = hopsworks.login(
    host=HOPSWORKS_HOST,
    project=HOPSWORKS_PROJECT,
    api_key_value=HOPSWORKS_API_KEY,
    engine="python",
)

print(
    f"Connected to project: {project.name}"
)

fs = project.get_feature_store()

print(
    f"Feature Store: {fs.name}"
)


# =====================================================================
# GET FEATURE GROUP
# =====================================================================

print(
    f"\nLoading Feature Group: "
    f"{FEATURE_GROUP_NAME}, version {FEATURE_GROUP_VERSION}"
)

fg = fs.get_feature_group(
    name=FEATURE_GROUP_NAME,
    version=FEATURE_GROUP_VERSION,
)

if fg is None:
    raise RuntimeError(
        "Feature Group could not be found."
    )


print(
    f"Feature Group: {fg.name}"
)

print(
    f"Version:       {fg.version}"
)

print(
    f"Format:        {fg.time_travel_format}"
)


# =====================================================================
# READ DATA
# =====================================================================

print("\nReading latest feature data...")

df = fg.select_all().read()

if df.empty:
    raise RuntimeError(
        "No feature data found in Hopsworks."
    )


print(
    f"Rows available: {len(df)}"
)


# =====================================================================
# PREPARE TIMESTAMP
# =====================================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce",
)

if df["timestamp"].isna().any():
    raise RuntimeError(
        "Invalid timestamp found in Feature Group."
    )


# Sort chronologically.
df = (
    df
    .sort_values("timestamp")
    .reset_index(drop=True)
)


# =====================================================================
# GET LATEST OBSERVATION
# =====================================================================

latest = df.iloc[-1]

latest_timestamp = latest["timestamp"]


print("\nLatest available observation:")

print(
    f"Timestamp: {latest_timestamp}"
)

if "city" in latest:
    print(
        f"City:      {latest['city']}"
    )

if "aqi" in latest:
    print(
        f"Current AQI: {float(latest['aqi']):.2f}"
    )


# =====================================================================
# DETERMINE NEXT HOUR
# =====================================================================

prediction_time = (
    latest_timestamp
    + pd.Timedelta(hours=1)
)


# =====================================================================
# CHECK FEATURES
# =====================================================================

missing_features = [
    feature
    for feature in MODEL_FEATURES
    if feature not in df.columns
]

if missing_features:
    raise RuntimeError(
        "Missing model features:\n"
        + ", ".join(missing_features)
    )


# =====================================================================
# CREATE INPUT
# =====================================================================

X = pd.DataFrame(
    [
        [
            latest[feature]
            for feature in MODEL_FEATURES
        ]
    ],
    columns=MODEL_FEATURES,
)


# Make sure everything is numeric.
X = X.apply(
    pd.to_numeric,
    errors="coerce",
)


if X.isna().any().any():
    raise RuntimeError(
        "Latest observation contains invalid/missing "
        "model input values."
    )


# =====================================================================
# PREDICT
# =====================================================================

print(
    "\nGenerating next-hour prediction..."
)

prediction = float(
    model.predict(X)[0]
)


# AQI shouldn't be negative.
prediction = max(
    0.0,
    prediction,
)


# =====================================================================
# AQI CATEGORY
# =====================================================================

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


category = get_aqi_category(
    prediction
)


# =====================================================================
# RESULT
# =====================================================================

print("\n" + "=" * 70)
print("NEXT-HOUR AQI PREDICTION")
print("=" * 70)

print(
    f"\nCurrent observation:"
)

print(
    f"  {latest_timestamp}"
)

print(
    f"\nPrediction for:"
)

print(
    f"  {prediction_time}"
)

print(
    f"\nPredicted AQI:"
)

print(
    f"  {prediction:.2f}"
)

print(
    f"\nAQI Category:"
)

print(
    f"  {category}"
)

print("\n" + "=" * 70)
print("PREDICTION COMPLETE")
print("=" * 70)