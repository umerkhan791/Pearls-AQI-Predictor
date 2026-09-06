# Pearls AQI Predictor

A machine learning system for forecasting **Karachi's Air Quality Index (AQI)**, combining automated data ingestion, feature engineering, a cloud-hosted feature store, model training, model registry management, a FastAPI prediction service, recursive 72-hour forecasting, and an interactive Streamlit dashboard.

Built for the **10Pearls SHINE Internship — Data Sciences Track**.

---

## Live Demo

<img width="1626" height="666" alt="image" src="https://github.com/user-attachments/assets/90d43ac1-9b14-44a8-905b-5ffe6fd7bf30" />


**Live Dashboard:** https://pearls-aqi-predictor-karachicity.streamlit.app/

**Prediction API:** https://pearls-aqi-api-xc81.onrender.com

**GitHub Repository:** https://github.com/umerkhan791/Pearls-AQI-Predictor

---

## Overview

Pearls AQI Predictor is an end-to-end machine learning system designed to forecast Karachi's air quality and present the results through a publicly accessible dashboard.

The system combines historical AQI and weather data, automated feature engineering, Hopsworks Feature Store, Random Forest regression, Hopsworks Model Registry, a FastAPI prediction service, GitHub Actions automation, recursive 72-hour forecasting, SHAP-based model explainability, an interactive Streamlit dashboard, and AQI health-category classification.

The production model is optimized for **next-hour AQI prediction**, while the dashboard also provides a recursive 72-hour forecast for longer-term visibility.

---

## System Architecture

```
      ┌─────────────────────────┐
      │     External Data APIs  │
      │                         │
      │  Open-Meteo             │
      │  • AQI                  │
      │  • Pollutants           │
      │  • Weather              │
      │                         │
      │  AQICN                  │
      │  • AQI / Station Data   │
      └────────────┬────────────┘
                   │
                   ▼
      ┌─────────────────────────┐
      │   Feature Pipeline      │
      │                         │
      │ • Data collection       │
      │ • Cleaning              │
      │ • Feature engineering   │
      │ • AQI features          │
      └────────────┬────────────┘
                   │
                   ▼
      ┌─────────────────────────┐
      │ Hopsworks Feature Store │
      │                         │
      │ karachi_aqi_features    │
      │ karachi_aqi_fv          │
      └────────────┬────────────┘
                   │
                   ▼
      ┌─────────────────────────┐
      │   Training Pipeline     │
      │                         │
      │ • Random Forest         │
      │ • Evaluation            │
      │ • Model Selection       │
      └────────────┬────────────┘
                   │
                   ▼
      ┌─────────────────────────┐
      │ Hopsworks Model Registry│
      │                         │
      │ RF Model v2             │
      └─────────────────────────┘


      Open-Meteo Current Data
               │
               ▼
      ┌─────────────────────────┐
      │   Prediction Service    │
      │                         │
      │ FastAPI                 │
      │ SQLite Historical Data  │
      │ Bundled RF Model        │
      │ Next-hour AQI           │
      └────────────┬────────────┘
                   │
                   ▼
      ┌─────────────────────────┐
      │ 72-Hour Forecast Engine |
      │                         |
      │ GitHub Actions          │
      │ Recursive predictions   │
      │ generated hourly        │
      └────────────┬────────────┘
                   │
                   ▼
      ┌─────────────────────────┐
      │ Streamlit Dashboard     │
      │                         │
      │ • Current AQI estimate  │
      │ • Next-hour forecast    │
      │ • 72-hour forecast      │
      │ • Pollutants            │
      │ • Weather               │
      │ • Model metrics         │
      │ • SHAP explainability   │
      └─────────────────────────┘
```

---

## Data Sources

### Open-Meteo

Open-Meteo is used for historical and current environmental data, including US AQI, PM2.5, PM10, ozone, nitrogen dioxide, sulfur dioxide, carbon monoxide, temperature, relative humidity, atmospheric pressure, and wind speed.

Historical Open-Meteo data supports the local dataset backfill, feature engineering, and model training. The live prediction service uses Open-Meteo's current AQI and weather data to construct the latest prediction features.

### AQICN

AQICN is included as an external AQI and station-data source in the project architecture and configuration. The current production prediction path uses **Open-Meteo current AQI and weather data** because the available AQICN Karachi feed was not suitable as a reliable current-data source.

---

## Historical Data Backfill

The project includes a historical data backfill pipeline that collects hourly Karachi AQI and weather records. The local SQLite feature database contains over **2,000 hourly records** covering multiple months of Karachi data, used for historical analysis, feature engineering, model training, time-series validation, and forecast evaluation.

---

## Feature Engineering

The model uses temporal, historical AQI, pollutant, and weather features.

**Calendar features** — hour, day, day of week, month, weekend indicator

**AQI lag features** — 1-hour, 3-hour, 6-hour, 12-hour, 24-hour lags

**Rolling statistics** — 3, 6, 12, and 24-hour AQI means; 6 and 24-hour AQI standard deviations

**AQI change features** — 1-hour, 6-hour, and 24-hour AQI changes

**Pollutant features** — PM2.5, PM10, ozone, nitrogen dioxide, sulfur dioxide, carbon monoxide

**Weather features** — temperature, relative humidity, atmospheric pressure, wind speed

---

## Machine Learning Model

The production model is a **Random Forest Regressor** trained to predict the next-hour AQI from current conditions, historical AQI patterns, pollutant concentrations, weather variables, and temporal features.

```
Model:             RandomForestRegressor
Trees:             300
Random State:      42
Parallel Jobs:     Enabled
Prediction Target: AQI at t+1 hour
```

The 72-hour forecast is generated recursively — each predicted AQI value is fed back as an input for the next prediction step.

---

## Model Performance

Evaluation uses a chronological train/test split to preserve the temporal structure of the data rather than randomly shuffling observations.

### Next-Hour Production Model

| Metric | Result |
|--------|-------:|
| MAE    | 0.530  |
| RMSE   | 0.714  |
| R²     | 0.994  |

The model demonstrates strong next-hour predictive performance on the held-out test period. The most influential feature is the previous-hour AQI (`aqi_lag_1h`), reflecting the strong short-term persistence of air quality conditions.

### Longer-Horizon Validation

When applied recursively for longer horizons, prediction uncertainty increases because each step depends partly on the previous prediction. Independent longer-horizon validation produced:

| Horizon   | MAE   | RMSE  | R²     |
|-----------|------:|------:|-------:|
| +24 hours | 6.10  | 7.20  | 0.549  |
| +48 hours | 10.00 | 11.97 | -0.175 |
| +72 hours | 11.29 | 12.93 | -0.286 |

These results are reported transparently. The dashboard separates the validated next-hour production performance from the longer-horizon recursive forecast.

---

## Hopsworks Feature Store

Engineered features are stored and versioned using Hopsworks Feature Store, which provides the training data consumed by the model-training pipeline.

```
Project:        pearls_aqi_predictors
Feature Group:  karachi_aqi_features
Feature View:   karachi_aqi_fv
```

---

## Hopsworks Model Registry

The production model is registered in the Hopsworks Model Registry for versioned model management and experiment tracking.

```
Model:    karachi_aqi_next_hour_rf
Version:  2
Type:     Random Forest Regressor

MAE:   0.5297
RMSE:  0.7140
R²:    0.9944
```

---

## Automated Pipelines

Three GitHub Actions workflows automate the project's data, model, and forecasting lifecycle.

### Hourly Feature Pipeline

Runs every hour. Collects AQI and weather data, processes and cleans it, generates engineered features, and updates the Hopsworks Feature Store.

```
.github/workflows/feature_pipeline.yml
```

### Daily Training Pipeline

Runs daily. Retrieves historical training data, trains the machine learning model, evaluates performance, and registers the trained model in Hopsworks Model Registry.

```
.github/workflows/training_pipeline.yml
```

### Hourly 72-Hour Forecast Pipeline

Runs every hour. Generates the latest recursive 72-hour AQI forecast, saves it to `data/forecast_72h.csv`, commits the updated file, and pushes it to the repository — allowing the deployed dashboard to always consume a fresh forecast.

```
.github/workflows/forecast_pipeline.yml
```

---

## Automation Schedule

| Workflow          | Frequency | Purpose                                           |
|-------------------|-----------|---------------------------------------------------|
| Feature Pipeline  | Hourly    | Update environmental data and Hopsworks features  |
| Training Pipeline | Daily     | Train, evaluate and register the production model |
| Forecast Pipeline | Hourly    | Generate and publish the latest 72-hour forecast  |

---

## 72-Hour Forecasting

The system generates a 72-step recursive AQI forecast. At each step, the current AQI is used to predict the next hour, that prediction becomes the next input, and this repeats for all 72 hours.

The dashboard summarizes the resulting forecast into three daily windows:

- **Day 1:** 0–24 hours
- **Day 2:** 24–48 hours
- **Day 3:** 48–72 hours

The forecast is saved to `data/forecast_72h.csv`. Each record contains a forecast timestamp, predicted AQI, and AQI category.

---

## Prediction API

The project includes a FastAPI prediction service deployed on Render.

**Endpoint:** `https://pearls-aqi-api-xc81.onrender.com/predict`

The API retrieves current AQI and weather data, loads recent historical AQI values, constructs the required model features, loads the production Random Forest, and returns the next-hour prediction alongside current environmental information.

The response includes the current AQI estimate and category, prediction timestamp, predicted next-hour AQI and category, model information and evaluation metrics, pollutant measurements, weather conditions, and data-source information.

---

## Interactive Dashboard

The Streamlit dashboard provides a public interface for the full system.

**Current AQI** — current AQI estimate, health category, next-hour prediction, pollutant readings, weather conditions.

**72-Hour Forecast** — 24, 48, and 72-hour forecast with an interactive chart, hourly table, and AQI categories.

**Model Evaluation** — MAE, RMSE, R², and longer-horizon validation metrics.

**Explainability** — SHAP-based feature contributions showing which variables drive each prediction, making the model more transparent rather than treating it as a black box.

---

## AQI Health Categories

| AQI Range | Category                       |
|----------:|-------------------------------|
| 0–50      | Good                           |
| 51–100    | Moderate                       |
| 101–150   | Unhealthy for Sensitive Groups |
| 151–200   | Unhealthy                      |
| 201–300   | Very Unhealthy                 |
| 301+      | Hazardous                      |

Elevated and hazardous AQI levels are visually highlighted in the dashboard to make potentially dangerous conditions easier to identify.

---

## Project Structure

```
Pearls-AQI-Predictor/
│
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml
│       ├── training_pipeline.yml
│       └── forecast_pipeline.yml
│
├── data/
│   ├── feature_store.db
│   └── forecast_72h.csv
│
├── models/
│
├── reports/
│   ├── training_metrics.json
│   ├── horizon_metrics.json
│   ├── actual_vs_predicted.png
│   ├── residuals.png
│   └── next_hour_predictions.csv
│
├── notebooks/
├── pipelines/
│
├── api.py
├── backfill.py
├── config.py
├── create_feature_view.py
├── database.py
├── dashboard.py
├── evaluate_horizons.py
├── generate_72h_forecast.py
├── hopsworks_feature_store.py
├── hopsworks_test.py
├── predict.py
├── register_model.py
├── train_model.py
├── aqi_model.pkl
├── requirements.txt
├── requirements-api.txt
├── requirements-ci.txt
├── requirements-dashboard.txt
├── .gitignore
└── README.md
```

### Component Reference

| File | Purpose |
|------|---------|
| `api.py` | FastAPI service, live AQI retrieval, feature construction and prediction |
| `dashboard.py` | Streamlit interactive dashboard |
| `backfill.py` | Historical AQI and weather data collection |
| `hopsworks_feature_store.py` | Hopsworks Feature Store integration |
| `create_feature_view.py` | Creates the Hopsworks Feature View |
| `train_model.py` | Model training and evaluation |
| `register_model.py` | Hopsworks Model Registry integration |
| `generate_72h_forecast.py` | Generates the recursive 72-hour forecast |
| `evaluate_horizons.py` | Evaluates longer forecast horizons |
| `predict.py` | Prediction utilities |
| `database.py` | Local feature and data storage |
| `config.py` | Project configuration |
| `hopsworks_test.py` | Hopsworks connectivity testing |

---

## Technologies

**Machine Learning** — Python, pandas, NumPy, scikit-learn, Random Forest, SHAP

**Data and Feature Engineering** — Open-Meteo, AQICN, Hopsworks Feature Store

**Model Management** — Hopsworks Model Registry

**Application** — FastAPI, Uvicorn, Streamlit, Render

**Automation** — GitHub Actions

**Storage** — Hopsworks Feature Store, SQLite

---

## Installation

Clone the repository:

```bash
git clone https://github.com/umerkhan791/Pearls-AQI-Predictor.git
cd Pearls-AQI-Predictor
```

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows PowerShell
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
HOPSWORKS_API_KEY=your_hopsworks_api_key
AQICN_TOKEN=your_aqicn_token
```

Never commit API keys or secrets to version control.

---

## Running the Project

**Start the API:**

```bash
uvicorn api:app --reload
```

Local API available at `http://127.0.0.1:8000`.

**Start the dashboard:**

```bash
streamlit run dashboard.py
```

**Generate a 72-hour forecast:**

```bash
python generate_72h_forecast.py
```

The forecast is saved to `data/forecast_72h.csv`.

---

## Forecast Data Format

The generated forecast CSV contains 72 hourly records:

```
predicted_for,predicted_aqi,category
2026-09-06 09:00:00,69.37,Moderate
2026-09-06 10:00:00,68.80,Moderate
```

---

## Known Limitations

**Forecast horizon** — The production model is strongly validated for next-hour prediction. Recursive forecasting over 24–72 hours introduces increasing uncertainty, as shown in the longer-horizon validation results.

**Recursive error propagation** — Each future prediction depends partly on previous model outputs, so prediction errors can accumulate over longer horizons.

**Live data availability** — The dashboard depends on third-party environmental APIs. Outages, stale data, or changes in API coverage can affect live predictions.

**AQI data representation** — The live AQI displayed by the prediction API is an Open-Meteo AQI estimate rather than a direct measurement from a physical Karachi monitoring station.

**Geographic coverage** — The current system provides a Karachi-level prediction rather than a neighborhood-resolution AQI map.

---

## Potential Improvements

- Train dedicated models for 24-hour, 48-hour, and 72-hour forecasting rather than relying solely on recursive application of the next-hour model
- Incorporate additional historical data and more Karachi monitoring stations
- Add satellite-based air quality indicators and traffic or emissions-related features
- Produce prediction intervals rather than single-point forecasts to communicate uncertainty
- Build automated forecast-accuracy monitoring to track performance over time
- Strengthen incoming data validation and anomaly detection
- Improve long-horizon forecasting performance

---

## Internship Requirements

| Requirement | Status |
|-------------|--------|
| Historical data collection | Done |
| Data cleaning and preprocessing | Done |
| Feature engineering | Done |
| AQI forecasting | Done |
| Time-based features | Done |
| Lag and rolling features | Done |
| Feature Store integration | Done |
| Model training | Done |
| Model evaluation | Done |
| Model Registry integration | Done |
| Automated feature pipeline | Done |
| Automated training pipeline | Done |
| Automated 72-hour forecast pipeline | Done |
| Interactive dashboard | Done |
| SHAP explainability | Done |
| AQI health categories | Done |
| Hazardous AQI highlighting | Done |
| Forecast visualisation | Done |
| End-to-end ML workflow | Done |

---

## End-to-End Workflow

```
1.  Collect historical and current AQI and weather data
2.  Clean and preprocess environmental data
3.  Generate temporal, lag, rolling, pollutant and weather features
4.  Store engineered features in Hopsworks
5.  Train and evaluate the Random Forest model
6.  Register the production model in Hopsworks Model Registry
7.  Retrieve current AQI and weather data
8.  Construct live prediction features
9.  Generate next-hour AQI prediction through FastAPI
10. Generate recursive 72-hour forecast
11. Save forecast to forecast_72h.csv
12. Display current conditions, predictions and explanations
    through the Streamlit dashboard
```

---

## Author

**Umer Khan**
10Pearls SHINE Internship — Data Sciences Track

---

## License

This project was developed as part of an internship programme for educational and demonstration purposes.
