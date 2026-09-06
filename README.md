# Pearls AQI Predictor

A serverless machine learning system for forecasting **Karachi's Air Quality Index (AQI)**, combining automated data ingestion, feature engineering, a cloud-hosted feature store, model training, model registry management, and an interactive Streamlit dashboard.

Built for the **10Pearls SHINE Internship — Data Sciences Track**.

<img width="1880" height="798" alt="image" src="https://github.com/user-attachments/assets/12cc7b5e-de39-46fe-999c-5b2b246ca18c" />


**Live Dashboard:** https://pearls-aqi-predictor-karachicity.streamlit.app/

**GitHub Repository:** https://github.com/umerkhan791/Pearls-AQI-Predictor

---

## Overview

Pearls AQI Predictor is a complete, end-to-end machine learning pipeline designed to forecast Karachi's air quality and present the results through a publicly accessible dashboard. The project was built with a serverless architecture, meaning data ingestion, model training, and serving all operate without a dedicated application server.

The system brings together real-time and historical air quality data, automated feature engineering, a Hopsworks Feature Store, a trained Random Forest model, GitHub Actions automation, recursive 72-hour forecasting, and SHAP-based model explainability — all surfaced through an interactive Streamlit interface.

---

## System Architecture

```
                    ┌──────────────────────┐
                    │   External APIs      │
                    │                      │
                    │  Open-Meteo          │
                    │  AQI + Weather Data  │
                    │                      │
                    │  AQICN               │
                    │  Live Data Source    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Feature Pipeline     │
                    │                      │
                    │ Data collection      │
                    │ Cleaning             │
                    │ Feature engineering  │
                    │ AQI features         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Hopsworks Feature    │
                    │ Store                │
                    │                      │
                    │ Karachi AQI Features │
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
       ┌──────────────────┐       ┌──────────────────┐
       │ Training Pipeline│       │ Live API /       │
       │                  │       │ Dashboard        │
       │ Random Forest    │       │                  │
       │ Evaluation       │       │ Current AQI      │
       │ Model Selection  │       │ Next-hour AQI    │
       └────────┬─────────┘       │ 72-hour forecast │
                │                 └────────┬─────────┘
                ▼                          │
       ┌──────────────────┐                │
       │ Hopsworks Model  │                │
       │ Registry         │                │
       │                  │                │
       │ RF Model v2      │                │
       └──────────────────┘                │
                                           ▼
                                ┌────────────────────┐
                                │ Streamlit          │
                                │ Dashboard          │
                                │                    │
                                │ AQI Status         │
                                │ Forecast           │
                                │ Pollutants         │
                                │ Weather            │
                                │ Model Metrics      │
                                │ SHAP Explainability│
                                └────────────────────┘
```

---

## Key Features

### Historical AQI Backfill

Historical hourly air quality and weather data is collected through Open-Meteo's historical APIs. The local feature database currently holds over **2,000 hourly records** spanning multiple months of Karachi data. This backfill forms the foundation for feature engineering, model training, time-series validation, and forecast evaluation.

### Feature Engineering

The model draws on a range of time-based, historical, pollutant, and weather features.

**Calendar features** — hour, day, day of week, month, weekend indicator

**AQI lag features** — 1-hour, 3-hour, 6-hour, 12-hour, and 24-hour lags

**Rolling statistics** — 3, 6, 12, and 24-hour AQI means; 6 and 24-hour standard deviations

**AQI change features** — 1-hour, 6-hour, and 24-hour changes

**Pollutants** — PM2.5, PM10, ozone, nitrogen dioxide, sulfur dioxide, carbon monoxide

**Weather** — temperature, relative humidity, atmospheric pressure, wind speed

---

## Machine Learning Model

The production model is a **Random Forest Regressor** trained to predict the next-hour AQI from current conditions, historical AQI patterns, pollutant concentrations, weather variables, and temporal features.

```
Model:             RandomForestRegressor
Trees:             300
Random State:      42
Parallel Jobs:     Enabled
```

The 72-hour forecast is generated recursively — each predicted value is fed back into the model as an input for the next forecast step.

---

## Model Performance

Evaluation was performed using a chronological train/test split, preserving the temporal structure of the data rather than randomly shuffling it.

### Next-Hour Production Model

| Metric | Result |
|--------|-------:|
| MAE    | 0.530  |
| RMSE   | 0.714  |
| R²     | 0.994  |

The model produces strong next-hour predictions on the held-out test period. The most influential feature is the previous-hour AQI (`aqi_lag_1h`), which reflects how persistent air quality conditions tend to be over short time windows.

### Longer-Horizon Validation

Because the production model is trained for one-hour-ahead prediction, recursive forecasting becomes increasingly uncertain as the horizon extends. Each prediction step inherits the error from the step before it.

Independent validation at longer horizons produced the following results:

| Horizon   | MAE   | RMSE  | R²     |
|-----------|------:|------:|-------:|
| +24 hours | 6.10  | 7.20  | 0.549  |
| +48 hours | 10.00 | 11.97 | -0.175 |
| +72 hours | 11.29 | 12.93 | -0.286 |

These results are reported transparently rather than hidden. The dashboard clearly separates the 24/48/72-hour forecast from the validated next-hour performance so that users understand what each figure represents.

---

## Hopsworks Feature Store

Engineered features are stored and versioned in Hopsworks Feature Store, which provides the training dataset used by the machine learning pipeline.

```
Project:       pearls_aqi_predictors
Feature Group: karachi_aqi_features
Feature View:  karachi_aqi_fv
```

### Hopsworks Model Registry

The trained production model is registered in the Hopsworks Model Registry for versioned tracking and retrieval.

```
Model:   karachi_aqi_next_hour_rf
Version: 2
Type:    Random Forest Regressor
```

---

## Automated Pipelines

Two GitHub Actions workflows automate the data and model lifecycle.

**Hourly Feature Pipeline** — fetches new AQI and weather data, processes it, generates features, and updates the Feature Store.

**Daily Training Pipeline** — retrieves historical training data, trains the forecasting model, evaluates it, and registers the updated model in Hopsworks.

```
.github/workflows/
├── feature_pipeline.yml
└── training_pipeline.yml
```

---

## 72-Hour Forecasting

The system generates a 72-step recursive forecast sequence. At each step, the previous prediction becomes an input for the next:

```
Current AQI → Predict next hour → Use prediction as input → Predict next hour → Repeat × 72
```

The Streamlit dashboard summarises this into three daily windows:

- Day 1: 0–24 hours
- Day 2: 24–48 hours
- Day 3: 48–72 hours

---

## Interactive Dashboard

The dashboard is publicly available and provides a full view of the system's outputs.

**Current AQI** — shows the current AQI estimate, health category, pollutant breakdown, weather conditions, and next-hour prediction.

**72-Hour Forecast** — shows the 24, 48, and 72-hour forecast with a chart and hourly table.

**Model Evaluation** — displays MAE, RMSE, R², and the longer-horizon validation results.

**Explainability** — SHAP values are used to explain which features drive each prediction, providing a transparent view of why the model produces a given AQI estimate rather than treating it as a black box.

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

The dashboard highlights elevated and hazardous AQI conditions to make dangerous air quality situations easier to identify at a glance.

---

## Project Structure

```
Pearls-AQI-Predictor/
│
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml
│       └── training_pipeline.yml
│
├── data/
│   ├── feature_store.db
│   └── forecast_72h.csv
│
├── models/
│   └── aqi_model.pkl
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
├── requirements.txt
├── .gitignore
└── README.md
```

### Component Reference

| File | Purpose |
|------|---------|
| `api.py` | API endpoints, live AQI retrieval, feature construction and prediction |
| `dashboard.py` | Streamlit interactive dashboard |
| `backfill.py` | Historical AQI and weather data collection |
| `hopsworks_feature_store.py` | Feature Store integration |
| `create_feature_view.py` | Creates the Hopsworks Feature View |
| `train_model.py` | Model training and evaluation |
| `register_model.py` | Model Registry integration |
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

**Application** — FastAPI, Streamlit

**Automation** — GitHub Actions

**Storage** — Hopsworks Feature Store, local SQLite database

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

Do not commit API keys or secrets to version control.

---

## Running the Project

**Start the API:**

```bash
uvicorn api:app --reload
```

**Start the dashboard:**

```bash
streamlit run dashboard.py
```

**Generate a 72-hour forecast:**

```bash
python generate_72h_forecast.py
```

The forecast is saved to `data/forecast_72h.csv` and includes the forecast timestamp, predicted AQI, and AQI category for each hour.

---

## Known Limitations

**Forecast horizon** — The production model is optimised and strongly validated for next-hour prediction. Recursive forecasting over 24–72 hours introduces cumulative error. The longer-horizon results are reported separately and should not be treated as equivalent to the next-hour performance.

**Recursive error propagation** — Each future prediction depends on previous model outputs. Small errors accumulate and compound across later forecast steps.

**Live data availability** — The dashboard depends on third-party environmental APIs. Temporary outages, stale station data, or changes in API coverage can affect live predictions.

**Single monitoring station** — The current data reflects one primary Karachi data source. Coverage from additional monitoring stations would improve geographic accuracy.

---

## Potential Improvements

- Train dedicated 24-hour, 48-hour, and 72-hour models rather than relying solely on recursive forecasting
- Incorporate additional historical data and more Karachi monitoring stations
- Add satellite or traffic-related air quality indicators
- Produce prediction intervals rather than single-point forecasts
- Automate forecast regeneration on a scheduled basis
- Implement online monitoring to track forecast accuracy over time
- Improve incoming data validation and anomaly detection

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
| Interactive dashboard | Done |
| SHAP explainability | Done |
| AQI health categories | Done |
| Hazardous AQI alerting | Done |
| Forecast visualisation | Done |
| End-to-end ML workflow | Done |

---

## Author

**Umer Khan**
10Pearls SHINE Internship — Data Sciences Track

---

## License

This project was developed as part of an internship programme for educational and demonstration purposes.
