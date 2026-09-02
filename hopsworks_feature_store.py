"""
Hopsworks Feature Store upload for Pearls AQI Predictor.

Uses the Python Hopsworks engine with a HUDI Feature Group.

The feature pipeline intentionally leaves the first rows with NaN values
because lag/rolling features require historical observations.

Those incomplete warm-up rows are removed here before uploading.
They are NOT median-imputed, which avoids introducing information from
later observations into the time-series dataset.
"""

import os

import hopsworks
import pandas as pd
from dotenv import load_dotenv

from database import load_features
from config import CITY


# =====================================================================
# CONFIGURATION
# =====================================================================

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

HOPSWORKS_HOST = "eu-west.cloud.hopsworks.ai"
HOPSWORKS_PROJECT = "pearls_aqi_predictors"

FEATURE_GROUP_NAME = "karachi_aqi_features"

# Existing HUDI Feature Group
FEATURE_GROUP_VERSION = 4


# =====================================================================
# VALIDATION
# =====================================================================

if not HOPSWORKS_API_KEY:
    raise RuntimeError(
        "HOPSWORKS_API_KEY is missing from .env"
    )


# =====================================================================
# LOAD DATA
# =====================================================================

print("=" * 70)
print("HOPSWORKS FEATURE STORE UPLOAD")
print("=" * 70)

print("\nLoading features from SQLite...")

df = load_features(
    CITY,
    limit=10000,
)

if df.empty:
    raise RuntimeError(
        "No feature data found in SQLite database."
    )

print(f"Loaded {len(df)} rows.")
print(f"Columns: {len(df.columns)}")


# =====================================================================
# PREPARE DATA
# =====================================================================

# Remove local SQLite-only columns.
df = df.drop(
    columns=[
        c
        for c in ["id", "created_at"]
        if c in df.columns
    ]
)


# ---------------------------------------------------------------------
# Timestamp
# ---------------------------------------------------------------------

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    errors="coerce",
)

if df["timestamp"].isna().any():
    raise RuntimeError(
        "Invalid timestamp values found."
    )

# Hopsworks/Arrow timestamp compatibility.
df["timestamp"] = df["timestamp"].dt.floor("us")


# ---------------------------------------------------------------------
# City
# ---------------------------------------------------------------------

df["city"] = df["city"].astype(str)


# ---------------------------------------------------------------------
# Sort and remove duplicate primary keys
# ---------------------------------------------------------------------

df = (
    df
    .sort_values(
        ["timestamp", "city"]
    )
    .drop_duplicates(
        subset=[
            "timestamp",
            "city",
        ],
        keep="last",
    )
    .reset_index(drop=True)
)


# =====================================================================
# EXPECTED SCHEMA
# =====================================================================

EXPECTED_COLUMNS = [
    "timestamp",
    "city",
    "aqi",

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


missing_columns = [
    c
    for c in EXPECTED_COLUMNS
    if c not in df.columns
]

if missing_columns:
    raise RuntimeError(
        "Local dataset is missing columns:\n"
        + ", ".join(missing_columns)
    )


# Remove unexpected columns.
df = df[
    EXPECTED_COLUMNS
]


# =====================================================================
# REMOVE WARM-UP ROWS
# =====================================================================

print("\nChecking historical feature warm-up rows...")

before_drop = len(df)

# The lag/rolling features require up to 24 hours of history.
#
# We intentionally remove rows that cannot have complete historical
# features instead of filling them with statistics calculated from
# the entire dataset.

df = (
    df
    .dropna(
        subset=[
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
        ]
    )
    .reset_index(drop=True)
)

removed_rows = before_drop - len(df)

print(
    f"Removed {removed_rows} incomplete warm-up rows."
)

print(
    f"Rows remaining: {len(df)}"
)


if df.empty:
    raise RuntimeError(
        "No complete feature rows remain after removing "
        "warm-up rows."
    )


# =====================================================================
# FINAL DATA VALIDATION
# =====================================================================

missing_count = int(
    df.isna()
    .sum()
    .sum()
)

if missing_count > 0:

    print("\nUnexpected missing values remain:")

    print(
        df.isna()
        .sum()
        .loc[
            lambda x: x > 0
        ]
        .to_string()
    )

    raise RuntimeError(
        "Dataset still contains missing values after "
        "removing incomplete warm-up rows."
    )


print("\nPrepared dataset:")
print(f"Rows:    {len(df)}")
print(f"Columns: {len(df.columns)}")

print("\nMissing values:")
print(missing_count)

print("\nTimestamp range:")
print(
    f"From: {df['timestamp'].min()}"
)

print(
    f"To:   {df['timestamp'].max()}"
)


# =====================================================================
# CONNECT
# =====================================================================

print("\nConnecting to Hopsworks...")

# IMPORTANT:
# Use the Python engine.
#
# Do NOT use:
#     engine="spark"
#
# Do NOT use:
#     engine="spark-delta"

project = hopsworks.login(
    host=HOPSWORKS_HOST,
    project=HOPSWORKS_PROJECT,
    api_key_value=HOPSWORKS_API_KEY,
    engine="python",
)

print(
    f"Connected to project: {project.name}"
)


# =====================================================================
# FEATURE STORE
# =====================================================================

fs = project.get_feature_store()

print(
    f"Feature Store: {fs.name}"
)


# =====================================================================
# FEATURE GROUP
# =====================================================================

print(
    f"\nGetting/creating Feature Group: "
    f"{FEATURE_GROUP_NAME}, "
    f"version {FEATURE_GROUP_VERSION}"
)


fg = fs.get_or_create_feature_group(
    name=FEATURE_GROUP_NAME,

    version=FEATURE_GROUP_VERSION,

    description=(
        "Hourly real Karachi AQI observations and "
        "engineered features for the Pearls AQI "
        "Predictor internship project. "
        "Incomplete historical warm-up rows are "
        "excluded from the uploaded dataset."
    ),

    primary_key=[
        "timestamp",
        "city",
    ],

    event_time="timestamp",

    online_enabled=False,

    time_travel_format="HUDI",

    hudi_precombine_key="timestamp",
)


if fg is None:
    raise RuntimeError(
        "Hopsworks returned None for the Feature Group."
    )


print(
    f"\nFeature Group: {fg.name}"
)

print(
    f"Version:       {fg.version}"
)

print(
    f"Time travel:   {fg.time_travel_format}"
)

print(
    f"Online:        {fg.online_enabled}"
)


# =====================================================================
# VERIFY / DISPLAY SCHEMA
# =====================================================================

fg_columns = [
    c.name
    for c in fg.columns
]

print("\nHopsworks schema:")

if fg_columns:
    print(
        ", ".join(fg_columns)
    )
else:
    print(
        "Schema will be inferred from dataframe."
    )


# =====================================================================
# UPLOAD
# =====================================================================

print(
    f"\nUploading {len(df)} rows..."
)

print(
    "Engine:        Python"
)

print(
    "Format:        HUDI"
)

print(
    "Storage:       offline"
)

print(
    "Operation:     upsert"
)


try:

    job, report = fg.insert(
        df,
        operation="upsert",
        wait=True,
    )

except Exception as exc:

    print("\n" + "=" * 70)
    print("UPLOAD FAILED")
    print("=" * 70)

    print(
        f"\nError type: {type(exc).__name__}"
    )

    print(
        f"Error: {exc}"
    )

    raise


# =====================================================================
# SUCCESS
# =====================================================================

print("\n" + "=" * 70)
print("UPLOAD SUCCESSFUL")
print("=" * 70)

print(
    f"\nFeature Group: {fg.name}"
)

print(
    f"Version:       {fg.version}"
)

print(
    f"Format:        {fg.time_travel_format}"
)

print(
    f"Rows uploaded: {len(df)}"
)

if job is not None:
    print(
        "Job returned successfully."
    )


# =====================================================================
# SAMPLE
# =====================================================================

print("\nSample rows:")

print(
    df.head(5)
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("HOPSWORKS FEATURE STORE UPLOAD COMPLETE")
print("=" * 70)

print(
    "\nHopsworks project:"
)

print(
    "https://eu-west.cloud.hopsworks.ai:443/p/43146"
)