# Pearls AQI Predictor — Project Development Journal

## About This Document

This document records the development of the Pearls AQI Predictor from the initial project requirements through to deployment. It covers data collection, feature engineering, model development, Hopsworks integration, forecasting, automation, dashboard development, and the problems encountered along the way.

The purpose is not only to describe the final system, but to explain the reasoning behind technical decisions, the experiments performed, and the honest limitations discovered during evaluation.

This project was developed for the **10Pearls SHINE Internship — Data Sciences Track**.

---

## 1. Project Objective

The goal was to build an end-to-end machine learning system capable of forecasting Karachi's Air Quality Index (AQI), while demonstrating modern data science and MLOps practices across the full pipeline.

The internship requirements covered external environmental data, historical data processing, feature engineering, Feature Store integration, model training and evaluation, Model Registry, automated pipelines, an interactive dashboard, explainability, AQI health alerts, and multi-day forecasting.

Rather than treating this as a standalone notebook exercise, I designed it as a complete end-to-end machine learning pipeline:

```
Data Collection → Cleaning → Feature Engineering → Feature Store
→ Model Training → Evaluation → Model Registry → Prediction API
→ 72-Hour Forecast → Streamlit Dashboard
```

The architecture uses managed cloud services and serverless-style automation, with GitHub Actions for scheduled workflows, Hopsworks for feature and model management, Render for the FastAPI prediction service, and Streamlit Community Cloud for the dashboard.

---

## 2. Understanding the AQI Problem

Air quality is not determined by a single pollutant. The system uses AQI together with pollutant concentrations such as PM2.5, PM10, ozone, nitrogen dioxide, sulfur dioxide, and carbon monoxide. Weather conditions — temperature, humidity, atmospheric pressure, and wind speed — can also affect pollution levels significantly.

AQI also has strong temporal behavior. The air quality at a given hour is closely related to the air quality in the hours immediately before it. This persistence is one of the most important properties the model needs to capture.

Because of this, I chose not to train the model only on current pollutant values. The feature set was designed to combine historical AQI patterns, pollutant concentrations, weather conditions, time-based information, lag features, rolling statistics, and AQI change/trend features — giving the model context about both the current environment and recent behavior.

---

## 3. Choosing the Data Sources

The first decision was selecting suitable environmental APIs. The project explored AQICN and Open-Meteo as the primary sources.

AQICN provides AQI and station-based air quality information. Open-Meteo was particularly valuable for historical data, offering accessible hourly air quality and weather datasets for the Karachi geographic coordinate used by the project.

For the historical backfill, Open-Meteo was used to collect hourly environmental data for Karachi using the following coordinates:

```
Latitude:  24.8607
Longitude: 67.0011
```

An important design decision here was to convert incoming data into a consistent internal feature format before feeding it into the pipeline. For the final live prediction path, Open-Meteo is used because the AQICN feed available during development did not provide a sufficiently reliable current Karachi value for this application. Downstream components therefore work with a consistent internal representation rather than depending on the original source format.

---

## 4. Historical Data Collection

Machine learning requires enough historical data to identify meaningful patterns. I created a backfill process to collect hourly air quality and weather information for Karachi, covering AQI, PM2.5, PM10, ozone, nitrogen dioxide, sulfur dioxide, carbon monoxide, temperature, humidity, pressure, and wind speed.

The resulting local database contains more than **2,000 hourly records** spanning multiple months of Karachi data. This historical dataset forms the foundation for feature engineering, model training, time-series evaluation, and forecast validation.

---

## 5. Data Cleaning

Raw API data cannot be assumed to be complete or consistent. Before using the data for machine learning, the pipeline validates timestamps, checks for missing values, and confirms that the historical context required for feature construction is available.

This last point is particularly important. A 24-hour lag feature cannot be calculated for the first 24 rows of a dataset. A 24-hour rolling average similarly requires sufficient prior observations. Rows without enough historical context are excluded from training rather than filled with potentially misleading values. The same principle applies to the target: a training row must have a reliable future AQI value available, or it is excluded.

---

## 6. Avoiding Data Leakage

One of the most critical parts of working with time-series data is preventing the model from accidentally learning from the future. A standard random train/test split can easily create this problem — for instance, allowing March data into training while February is held back for testing.

The production model uses a strict chronological split instead. The oldest portion of the data is used for training and the most recent portion for testing, preserving temporal order throughout.

```
Training rows: 1,827
Testing rows:    457
```

---

## 7. Feature Engineering

Feature engineering was one of the most time-consuming and important parts of the project. The final model uses 29 engineered features across several categories.

**Calendar features** give the model awareness of time: hour, day, day of week, month, and a weekend indicator. These allow the model to learn recurring patterns — for example, higher AQI on weekday mornings due to traffic.

**AQI lag features** capture recent historical AQI values directly: 1-hour, 3-hour, 6-hour, 12-hour, and 24-hour lags. The 1-hour lag turned out to be the most important single feature in the model.

**Rolling statistics** give the model a sense of the broader recent trend: 3, 6, 12, and 24-hour AQI means, plus 6 and 24-hour standard deviations. The rolling standard deviation in particular tells the model whether recent AQI has been stable or volatile.

**AQI change features** capture how quickly AQI is moving: 1-hour, 6-hour, and 24-hour changes. These allow the model to distinguish between a stable AQI of 80 and an AQI of 80 that has been rising sharply over the past few hours — a meaningfully different situation.

**Pollutant and weather features** round out the feature set with PM2.5, PM10, ozone, NO2, SO2, CO, temperature, humidity, pressure, and wind speed.

---

## 8. Model Selection

Several machine learning approaches were considered during development, including Ridge Regression, Random Forest, Gradient Boosting, and XGBoost. Deep learning approaches were also explored as part of the broader experimentation required by the internship brief.

The evaluation criteria were not purely metric-based. The final model also needed to be reliable, fast enough for daily automated retraining, easy to deploy, explainable, and suitable for unattended execution in a scheduled pipeline.

The final production model is a **Random Forest Regressor** with 300 trees. It was selected as the production model based on its strong next-hour predictive performance, practical training and deployment requirements, and compatibility with the project's explainability workflow.

The trained model is registered in Hopsworks Model Registry as `karachi_aqi_next_hour_rf`, Version 2.

---

## 9. Production Model Performance

The production model was evaluated on the chronological held-out test set.

| Metric | Result |
|--------|-------:|
| MAE    | 0.530  |
| RMSE   | 0.714  |
| R²     | 0.994  |

These results reflect performance on the specific task the model was trained for: **next-hour AQI prediction**. They should not be interpreted as representing 72-hour forecasting accuracy, which is evaluated separately.

**On the high R²:** An R² of 0.994 is genuinely strong, but it makes sense in this context. AQI is persistent over short time windows — the previous hour's AQI is a very strong predictor of the next hour's AQI. Feature importance analysis confirmed this; `aqi_lag_1h` dominates the model. This is not a sign that the model is trivial, but it does mean the longer-horizon performance must be understood on its own terms.

---

## 10. The 72-Hour Forecast Challenge

The internship project requires a three-day forecast, but the production model is trained for one-hour-ahead prediction. The approach used is **recursive forecasting**: the model's prediction for hour 1 becomes an input for predicting hour 2, and so on up to hour 72.

This is a workable strategy, but it has an important limitation. Each prediction step inherits the error from the step before it. Small errors can accumulate across 72 steps, and because the Random Forest relies heavily on the previous AQI value, the recursive forecast can also tend toward a relatively stable AQI range rather than capturing real atmospheric variability.

---

## 11. Longer-Horizon Validation

To measure how the model actually performs at extended horizons, a separate horizon evaluation was run.

| Horizon   | MAE   | RMSE  | R²     |
|-----------|------:|------:|-------:|
| +24 hours | 6.10  | 7.20  | 0.549  |
| +48 hours | 10.00 | 11.97 | -0.175 |
| +72 hours | 11.29 | 12.93 | -0.286 |

Performance degrades substantially compared to the next-hour model. Rather than suppressing this result, I included it in the project documentation and in the dashboard itself. Presenting both the next-hour performance and the longer-horizon validation gives an accurate picture of what the system can and cannot currently do.

---

## 12. Hopsworks Feature Store and Model Registry

The project uses **Hopsworks Feature Store** as the cloud feature management layer.

```
Feature Group: karachi_aqi_features
Feature View:  karachi_aqi_fv
Project:       pearls_aqi_predictors
```

The Feature View provides the training dataset consumed by the machine learning pipeline. This demonstrates a core MLOps principle: separating raw data from engineered features and making features reproducible and versioned rather than ad hoc.

The trained production model is registered in the **Hopsworks Model Registry** under `karachi_aqi_next_hour_rf` (Version 2), with associated evaluation metrics and training data linkage. This makes it possible to track model versions over time and roll back if needed.

---

## 13. GitHub Actions Automation

Three GitHub Actions workflows automate the data, model, and forecast lifecycle.

**Hourly Feature Pipeline** — runs every hour, fetches new AQI and weather data, processes it, generates features, and updates the Feature Store.

**Daily Training Pipeline** — runs once per day, retrieves the historical training dataset, trains and evaluates the model, and registers the updated version in Hopsworks.

**Hourly 72-Hour Forecast Pipeline** — runs every hour, regenerates the recursive 72-hour AQI forecast, and commits the updated forecast data for the public dashboard.

Together, these workflows remove the need for manual feature updates, retraining, and forecast regeneration.

---

## 14. Engineering Problems Encountered

### Hopsworks on Windows

Getting the Hopsworks environment working on Windows was one of the more difficult parts of the project. The Hopsworks dependency stack introduced compatibility and installation conflicts. The solution was to use a dedicated Conda environment (`hopsenv`) for Hopsworks-related operations, separated from the main application environment.

### Feature Store Materialization Delay

A more significant problem was discovered during development: very recently inserted Feature Store rows were not always immediately available when queried. Hopsworks Feature Store has a materialization process, and data inserted by the hourly pipeline could be temporarily unavailable for live prediction.

The solution was to separate training and live serving responsibilities. Hopsworks remains the system of record for training data. The live prediction path uses locally available historical data to construct the current feature row, making live prediction independent of materialization delays.

### Deployment Dependency Conflicts

During Streamlit Cloud deployment, the initial dashboard dependency stack was too heavy and caused installation and resource problems. The solution was to separate application dependencies from the Hopsworks and CI workflow dependencies, resulting in a lighter and cleaner deployment environment.

---

## 15. Prediction API

A **FastAPI** backend handles live prediction. The API fetches current environmental data, loads recent historical AQI values, constructs the feature row, loads the production Random Forest, and returns the next-hour AQI prediction along with the current AQI, health category, pollutant readings, weather conditions, and model metrics.

---

## 16. Dashboard

The **Streamlit dashboard** provides a publicly accessible interface to the full system without requiring any direct Python interaction.

It covers current AQI and health category, next-hour forecast, 72-hour forecast with chart and hourly table, pollutant and weather readings, model evaluation metrics (including longer-horizon validation), SHAP explainability, AQI health reference, methodology documentation, and hazardous AQI highlighting.

The dashboard converts raw AQI values into standard health categories:

| AQI Range | Category                       |
|----------:|-------------------------------|
| 0–50      | Good                           |
| 51–100    | Moderate                       |
| 101–150   | Unhealthy for Sensitive Groups |
| 151–200   | Unhealthy                      |
| 201–300   | Very Unhealthy                 |
| 301+      | Hazardous                      |

---

## 17. SHAP Explainability

SHAP was incorporated to provide feature-level explanations of individual predictions — answering the question of why the model produced a given AQI estimate rather than treating it as a black box. The SHAP analysis shows the contribution of historical AQI features, pollutants, weather, and temporal features for each prediction, making the model's reasoning interpretable.

---

## 18. Known Limitations

**Recursive forecasting accuracy** — The production model is optimised and validated for next-hour prediction. Recursive application over 72 hours introduces error accumulation. The longer-horizon results are reported separately and honestly.

**Forecast stability** — Because the model relies heavily on the previous AQI value, recursive forecasts can converge toward a relatively stable range rather than capturing real atmospheric variability. A future version should investigate dedicated multi-horizon models.

**Geographic coverage** — The current system represents Karachi using a single geographic coordinate rather than a network of physical monitoring stations. Air quality can vary substantially across the city, so additional monitoring locations would improve geographic accuracy.

**AQI data representation** — The live AQI displayed by the prediction API is an Open-Meteo AQI estimate rather than a direct measurement from a physical Karachi monitoring station.

**External API dependency** — The system depends on third-party environmental data services. API downtime, stale station data, or changes in API coverage can affect live predictions.

---

## 19. Potential Improvements

- Train dedicated 24-hour, 48-hour, and 72-hour models rather than relying on recursive application of the next-hour model
- Incorporate more historical data and additional Karachi monitoring stations
- Add prediction intervals to communicate forecast uncertainty rather than returning only point estimates
- Incorporate additional environmental variables such as traffic, satellite observations, or fire/smoke data
- Build automated forecast accuracy monitoring to track model performance over time against actual observed AQI values

---

## 20. Final Reflection

The most important lesson from this project was that model performance is only one part of a real-world machine learning system. A model can achieve an excellent evaluation score and still encounter serious problems involving data freshness, feature availability, deployment dependencies, forecast horizons, and API reliability.

Building the full pipeline — rather than just a training notebook — exposed all of these issues in a way that a standalone model evaluation could not.

The project also reinforced why honest evaluation matters. The next-hour Random Forest model achieved very strong results. The same model's recursive 72-hour performance degraded substantially. Reporting both results, and explaining why they differ, gives a more accurate picture of what the system currently does well and where further work is needed.

---

## 21. Conclusion

The Pearls AQI Predictor evolved from a machine learning forecasting task into a complete data science and MLOps project covering the full lifecycle from raw data to deployed dashboard.

The final system demonstrates automated environmental data collection, historical backfilling, feature engineering, Hopsworks Feature Store and Model Registry integration, Random Forest modelling, GitHub Actions automation, recursive 72-hour forecasting, SHAP explainability, AQI health categorisation, and a publicly deployed interactive dashboard.

The current system provides a solid foundation for further development toward dedicated multi-horizon AQI forecasting and more geographically comprehensive air quality prediction for Karachi.
