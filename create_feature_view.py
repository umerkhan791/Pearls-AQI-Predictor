import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

project = hopsworks.login(
    host="eu-west.cloud.hopsworks.ai",
    project="pearls_aqi_predictors",
    api_key_value=HOPSWORKS_API_KEY,
    engine="python",
)

fs = project.get_feature_store()

# IMPORTANT: use the working Feature Group
fg = fs.get_feature_group(
    name="karachi_aqi_features",
    version=4,
)

print("Feature Group:")
print(f"  Name:    {fg.name}")
print(f"  Version: {fg.version}")
print(f"  Format:  {fg.time_travel_format}")

# Select the features for the model.
#
# We exclude:
#   timestamp -> used as event time
#   city      -> identifier, not useful as a numeric model feature
#   aqi       -> TARGET
#
# Everything else becomes an input feature.

feature_columns = [
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

# Build query.
query = fg.select(
    feature_columns + ["aqi", "timestamp", "city"]
)

print("\nFeature View columns:")
print(
    [f.name for f in query.features]
)

# Create Feature View.
fv = fs.get_or_create_feature_view(
    name="karachi_aqi_fv",
    version=1,
    query=query,
    description=(
        "Feature View for Karachi AQI prediction. "
        "Uses historical AQI lag, rolling statistics, "
        "pollutants and weather features."
    ),
    labels=["aqi"],
)

print("\n" + "=" * 70)
print("FEATURE VIEW CREATED")
print("=" * 70)

print(f"Name:    {fv.name}")
print(f"Version: {fv.version}")

print("\nFeatures:")
for feature in fv.features:
    print(
        f"  {feature.name}: {feature.type}"
    )