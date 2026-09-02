# 🌫️ AQI Predictor — Pearls Project

Predict the Air Quality Index (AQI) for the next 3 days using a fully automated,
serverless ML pipeline.

**City:** Karachi, Pakistan  
**Data Source:** AQICN API  
**Stack:** Python · SQLite · Scikit-learn · XGBoost · SHAP · Streamlit · GitHub Actions

---

## 📁 Project Structure

```
aqi_predictor/
├── config.py                    # All settings, API keys, paths
├── database.py                  # SQLite feature store (CRUD)
├── pipelines/
│   ├── feature_pipeline.py      # Fetch API → engineer features → store
│   ├── backfill.py              # Historical data loader
│   └── training_pipeline.py    # Train → evaluate → save model
├── models/                      # Saved .pkl model files
├── data/                        # SQLite DB lives here
├── app/
│   └── streamlit_app.py         # Interactive dashboard
├── .github/workflows/
│   ├── feature_pipeline.yml     # Runs every hour (CI/CD)
│   └── training_pipeline.yml    # Runs daily (CI/CD)
├── reports/                     # EDA plots, model reports
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/aqi-predictor.git
cd aqi-predictor
pip install -r requirements.txt
```

### 2. Set API Key
```bash
cp .env.example .env
# Edit .env and add your AQICN token
```

### 3. Initialize Database
```bash
python database.py
```

### 4. Backfill Historical Data
```bash
python pipelines/backfill.py
```

### 5. Train Models
```bash
python pipelines/training_pipeline.py
```

### 6. Launch Dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## 🔄 Pipeline Architecture

```
AQICN API (hourly)
     ↓
Feature Pipeline → SQLite Feature Store
     ↓
Training Pipeline → Model Registry (models/)
     ↓
Streamlit App → 3-Day AQI Forecast Dashboard
     ↓
GitHub Actions (automated scheduling)
```

---

## 📊 Models Used
- **Random Forest** — baseline ensemble model
- **XGBoost** — gradient boosting (usually best performer)
- **LSTM** — deep learning for sequence patterns (optional)

**Evaluation:** RMSE, MAE, R²  
**Explainability:** SHAP feature importance plots

---

## ⚠️ AQI Levels (US EPA)

| Level | AQI Range | Color |
|-------|-----------|-------|
| Good | 0–50 | 🟢 Green |
| Moderate | 51–100 | 🟡 Yellow |
| Unhealthy for Sensitive Groups | 101–150 | 🟠 Orange |
| Unhealthy | 151–200 | 🔴 Red |
| Very Unhealthy | 201–300 | 🟣 Purple |
| Hazardous | 301–500 | 🟤 Maroon |

Alerts are triggered automatically when predicted AQI > 150.

---

## 📋 Deliverables
1. End-to-end AQI prediction system
2. Scalable, automated pipeline (GitHub Actions)
3. Interactive Streamlit dashboard
4. Detailed project report (`reports/`)
