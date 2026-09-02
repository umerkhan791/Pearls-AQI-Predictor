"""
Pearls AQI Predictor - Next-Hour Model Training

Uses features at time t to predict AQI at time t+1.
"""

import os
import joblib
import numpy as np
import pandas as pd
import hopsworks

from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================================================
# CONFIGURATION
# =====================================================================

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"
HOPSWORKS_PROJECT = "pearls_aqi_predictors"

FEATURE_VIEW_NAME = "karachi_aqi_fv"
FEATURE_VIEW_VERSION = 1

MODEL_PATH = "aqi_model.pkl"
PREDICTIONS_PATH = "next_hour_predictions.csv"


# =====================================================================
# FEATURES
# =====================================================================

FEATURES = [
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


# =====================================================================
# START
# =====================================================================

print("=" * 70)
print("PEARLS AQI PREDICTOR - NEXT-HOUR MODEL TRAINING")
print("=" * 70)


# =====================================================================
# CONNECT
# =====================================================================

print("\nConnecting to Hopsworks...")

project = hopsworks.login(
    host=HOPSWORKS_HOST,
    project=HOPSWORKS_PROJECT,
    api_key_value=HOPSWORKS_API_KEY,
    engine="python",
)

print(f"Connected to project: {project.name}")

fs = project.get_feature_store()

print(f"Feature Store: {fs.name}")


# =====================================================================
# FEATURE VIEW
# =====================================================================

print(
    f"\nGetting Feature View: "
    f"{FEATURE_VIEW_NAME}, version {FEATURE_VIEW_VERSION}"
)

fv = fs.get_feature_view(
    name=FEATURE_VIEW_NAME,
    version=FEATURE_VIEW_VERSION,
)

print(f"Feature View: {fv.name}")
print(f"Version:      {fv.version}")


# =====================================================================
# GET FEATURES + LABEL
# =====================================================================

print("\nReading features and labels from Hopsworks...")

features_df, labels_df = fv.get_training_data(
    training_dataset_version=1,
    dataframe_type="pandas",
)


print(f"Feature rows: {len(features_df)}")
print(f"Label rows:   {len(labels_df)}")


# =====================================================================
# VERIFY LABEL
# =====================================================================

if "aqi" not in labels_df.columns:
    raise RuntimeError(
        "AQI label was not returned by Hopsworks."
    )


# =====================================================================
# PREPARE FEATURES
# =====================================================================

print("\nPreparing dataset...")


# Make copies so we don't modify Hopsworks-returned objects.
features_df = features_df.copy()
labels_df = labels_df.copy()


# Timestamp
features_df["timestamp"] = pd.to_datetime(
    features_df["timestamp"],
    errors="coerce",
)

if features_df["timestamp"].isna().any():
    raise RuntimeError(
        "Invalid timestamp values found."
    )


# AQI label
labels_df["aqi"] = pd.to_numeric(
    labels_df["aqi"],
    errors="coerce",
)

if labels_df["aqi"].isna().any():
    raise RuntimeError(
        "Invalid AQI label values found."
    )


# =====================================================================
# COMBINE FEATURES AND LABEL
# =====================================================================

print("\nCombining features and labels...")

if len(features_df) != len(labels_df):
    raise RuntimeError(
        "Feature and label row counts do not match."
    )


# The Hopsworks Feature View returns the matching rows.
df = features_df.copy()

df["aqi"] = labels_df["aqi"].values


print(
    f"Combined dataset: "
    f"{len(df)} rows × {len(df.columns)} columns"
)


# =====================================================================
# SORT CHRONOLOGICALLY
# =====================================================================

df = (
    df
    .sort_values("timestamp")
    .reset_index(drop=True)
)


# =====================================================================
# CHECK REQUIRED FEATURES
# =====================================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    raise RuntimeError(
        "Missing input features:\n"
        + ", ".join(missing_features)
    )


# =====================================================================
# CREATE NEXT-HOUR TARGET
# =====================================================================

print("\nCreating next-hour target...")

print(
    "Target definition:"
)

print(
    "Features at time t → AQI at time t+1"
)


df["target_aqi"] = df["aqi"].shift(-1)


# Remove final row because it has no next-hour target.
df = df.dropna(
    subset=["target_aqi"]
).reset_index(drop=True)


print(
    f"Rows after target creation: {len(df)}"
)


# =====================================================================
# CHECK MISSING VALUES
# =====================================================================

df = df.dropna(
    subset=FEATURES + ["target_aqi"]
).reset_index(drop=True)


print(
    f"Rows after missing-value removal: {len(df)}"
)


# =====================================================================
# CREATE X AND Y
# =====================================================================

X = df[FEATURES].copy()

y = df["target_aqi"].astype(float)


print("\nTarget:")
print("AQI at the next hour")


print("\nInput features:")

for feature in FEATURES:
    print(f"  {feature}")


print(
    f"\nNumber of input features: {len(FEATURES)}"
)


# =====================================================================
# CHRONOLOGICAL SPLIT
# =====================================================================

print("\nCreating chronological train/test split...")


split_index = int(
    len(df) * 0.80
)


X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

test_timestamps = df["timestamp"].iloc[split_index:]


print(
    f"Training rows: {len(X_train)}"
)

print(
    f"Testing rows:  {len(X_test)}"
)


print("\nTraining period:")

print(
    f"From: {df['timestamp'].iloc[0]}"
)

print(
    f"To:   {df['timestamp'].iloc[split_index - 1]}"
)


print("\nTesting period:")

print(
    f"From: {df['timestamp'].iloc[split_index]}"
)

print(
    f"To:   {df['timestamp'].iloc[-1]}"
)


# =====================================================================
# TRAIN RANDOM FOREST
# =====================================================================

print("\nTraining Random Forest...")


model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
)


model.fit(
    X_train,
    y_train,
)


print("Model training complete.")


# =====================================================================
# PREDICT
# =====================================================================

print("\nGenerating next-hour predictions...")


predictions = model.predict(
    X_test
)


# =====================================================================
# METRICS
# =====================================================================

mae = mean_absolute_error(
    y_test,
    predictions,
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions,
    )
)

r2 = r2_score(
    y_test,
    predictions,
)


print("\n" + "=" * 70)
print("NEXT-HOUR MODEL RESULTS")
print("=" * 70)

print(
    f"\nMAE:  {mae:.4f}"
)

print(
    f"RMSE: {rmse:.4f}"
)

print(
    f"R²:   {r2:.4f}"
)


# =====================================================================
# SAMPLE PREDICTIONS
# =====================================================================

results = pd.DataFrame(
    {
        "feature_timestamp": test_timestamps.values,
        "actual_next_hour_aqi": y_test.values,
        "predicted_next_hour_aqi": predictions,
    }
)


print("\nSample next-hour predictions:")

print(
    results
    .head(10)
    .to_string(index=False)
)


# =====================================================================
# FEATURE IMPORTANCE
# =====================================================================

importance = pd.DataFrame(
    {
        "feature": FEATURES,
        "importance": model.feature_importances_,
    }
).sort_values(
    "importance",
    ascending=False,
)


print("\nTop 15 feature importance:")

print(
    importance
    .head(15)
    .to_string(index=False)
)


# =====================================================================
# SAVE MODEL
# =====================================================================

joblib.dump(
    model,
    MODEL_PATH,
)


print(
    f"\nModel saved locally: {MODEL_PATH}"
)


# =====================================================================
# SAVE PREDICTIONS
# =====================================================================

results.to_csv(
    PREDICTIONS_PATH,
    index=False,
)


print(
    f"Predictions saved: {PREDICTIONS_PATH}"
)


# =====================================================================
# COMPLETE
# =====================================================================

print("\n" + "=" * 70)
print("NEXT-HOUR MODEL TRAINING COMPLETE")
print("=" * 70)