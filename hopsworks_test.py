import os
from dotenv import load_dotenv
import hopsworks

load_dotenv()

api_key = os.getenv("HOPSWORKS_API_KEY")

if not api_key:
    raise RuntimeError("HOPSWORKS_API_KEY is missing from .env")

project = hopsworks.login(
    host="eu-west.cloud.hopsworks.ai",
    project="pearls_aqi_predictors",
    api_key_value=api_key,
)

print("Hopsworks connection successful!")
print("Project:", project.name)