"""
Pearls AQI Predictor - Complete Training Pipeline

Models:
    1. Ridge Regression
    2. Random Forest
    3. XGBoost
    4. TensorFlow LSTM

Pipeline:
    Feature Store/SQLite
        ↓
    Dataset preparation
        ↓
    Chronological train/test split
        ↓
    Model training
        ↓
    RMSE / MAE / R² evaluation
        ↓
    SHAP explainability
        ↓
    Model artifacts / local registry
        ↓
    72-hour forecast
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from loguru import logger

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb
import shap

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# PROJECT IMPORTS
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    CITY,
    MODELS_DIR,
    REPORTS_DIR,
    RANDOM_SEED,
    TRAIN_TEST_SPLIT,
    MODEL_REGISTRY,
    ALERT_THRESHOLD,
    get_aqi_level,
    ensure_dirs,
)

from database import (
    init_db,
    load_features,
    save_model_metrics,
    save_predictions,
)

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ---------------------------------------------------------------------
# FEATURES
# ---------------------------------------------------------------------

FEATURE_COLS = [
    # Time
    "hour",
    "day",
    "month",
    "day_of_week",
    "is_weekend",

    # AQI lags
    "aqi_lag_1h",
    "aqi_lag_3h",
    "aqi_lag_6h",
    "aqi_lag_12h",
    "aqi_lag_24h",

    # Rolling features
    "aqi_roll_mean_3h",
    "aqi_roll_mean_6h",
    "aqi_roll_mean_12h",
    "aqi_roll_mean_24h",
    "aqi_roll_std_6h",
    "aqi_roll_std_24h",

    # AQI changes
    "aqi_change_1h",
    "aqi_change_6h",
    "aqi_change_24h",

    # Pollutants
    "pm25",
    "pm10",
    "o3",
    "no2",
    "so2",
    "co",

    # Weather
    "temperature",
    "humidity",
    "pressure",
    "wind",
]

TARGET_COL = "aqi"


# ---------------------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------------------

def prepare_dataset(
    df: pd.DataFrame,
    horizon_hours: int = 24,
):
    """
    Create features and a future AQI target.

    Example:
        horizon=24

    means:

        current features → AQI 24 hours later
    """

    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = (
        df.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .reset_index(drop=True)
    )

    available_features = [
        col for col in FEATURE_COLS
        if col in df.columns
    ]

    missing_features = [
        col for col in FEATURE_COLS
        if col not in df.columns
    ]

    if missing_features:
        logger.warning(
            f"Missing feature columns: {missing_features}"
        )

    # Future target
    df["target"] = df[TARGET_COL].shift(-horizon_hours)

    # Remove rows without target/features
    df = df.dropna(
        subset=["target"] + available_features
    )

    X = df[available_features].copy()
    y = df["target"].copy()

    logger.info(
        f"Prepared {len(X)} samples | "
        f"{len(available_features)} features | "
        f"horizon={horizon_hours}h"
    )

    return X, y, available_features


# ---------------------------------------------------------------------
# TIME SERIES SPLIT
# ---------------------------------------------------------------------

def chronological_split(X, y):
    """
    Chronological 80/20 split.

    We NEVER shuffle time-series data.
    """

    split_index = int(len(X) * TRAIN_TEST_SPLIT)

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()

    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    logger.info(
        f"Train samples: {len(X_train)}"
    )

    logger.info(
        f"Test samples: {len(X_test)}"
    )

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------
# CLASSICAL MODELS
# ---------------------------------------------------------------------

def create_classical_models():

    return {

        "ridge_regression": Ridge(
            alpha=1.0
        ),

        "random_forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),

        "xgboost": xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            objective="reg:squarederror",
        ),
    }


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------

def calculate_metrics(y_true, y_pred):

    return {
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred
                )
            )
        ),

        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred
            )
        ),

        "r2": float(
            r2_score(
                y_true,
                y_pred
            )
        ),
    }


# ---------------------------------------------------------------------
# CLASSICAL MODEL TRAINING
# ---------------------------------------------------------------------

def train_classical_models(
    X_train,
    X_test,
    y_train,
    y_test,
):

    scaler = StandardScaler()

    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = create_classical_models()

    results = {}
    predictions = {}

    trained_models = {}

    for name, model in models.items():

        logger.info(
            f"Training {name}..."
        )

        model.fit(
            X_train_scaled,
            y_train
        )

        pred = model.predict(
            X_test_scaled
        )

        metrics = calculate_metrics(
            y_test,
            pred
        )

        results[name] = metrics
        predictions[name] = pred
        trained_models[name] = model

        logger.info(
            f"{name}: "
            f"RMSE={metrics['rmse']:.2f} | "
            f"MAE={metrics['mae']:.2f} | "
            f"R²={metrics['r2']:.4f}"
        )

    return (
        trained_models,
        results,
        predictions,
        scaler,
        X_train_scaled,
        X_test_scaled,
    )


# ---------------------------------------------------------------------
# TIME SERIES CROSS VALIDATION
# ---------------------------------------------------------------------

def calculate_cv_rmse(
    model,
    X_train,
    y_train,
):

    tscv = TimeSeriesSplit(
        n_splits=5
    )

    scores = []

    for train_idx, val_idx in tscv.split(X_train):

        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]

        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]

        scaler = StandardScaler()

        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        cloned_model = model.__class__(
            **model.get_params()
        )

        cloned_model.fit(
            X_tr_scaled,
            y_tr
        )

        pred = cloned_model.predict(
            X_val_scaled
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_val,
                pred
            )
        )

        scores.append(rmse)

    return float(np.mean(scores))


# ---------------------------------------------------------------------
# LSTM DATASET
# ---------------------------------------------------------------------

def create_sequences(
    X,
    y,
    sequence_length=24,
):

    X_values = np.asarray(X)
    y_values = np.asarray(y)

    X_seq = []
    y_seq = []

    for i in range(
        sequence_length,
        len(X_values)
    ):

        X_seq.append(
            X_values[
                i - sequence_length:i
            ]
        )

        y_seq.append(
            y_values[i]
        )

    return (
        np.asarray(X_seq),
        np.asarray(y_seq),
    )


# ---------------------------------------------------------------------
# LSTM MODEL
# ---------------------------------------------------------------------

def build_lstm(
    sequence_length,
    n_features,
):

    model = Sequential([

        LSTM(
            64,
            input_shape=(
                sequence_length,
                n_features
            ),
            return_sequences=False,
        ),

        Dropout(0.2),

        Dense(32, activation="relu"),

        Dense(1),
    ])

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"],
    )

    return model


# ---------------------------------------------------------------------
# LSTM TRAINING
# ---------------------------------------------------------------------

def train_lstm(
    X_train,
    X_test,
    y_train,
    y_test,
):

    logger.info(
        "Preparing TensorFlow LSTM sequences..."
    )

    sequence_length = 24

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    X_train_seq, y_train_seq = create_sequences(
        X_train_scaled,
        y_train.values,
        sequence_length,
    )

    # Include the last 24 training observations
    # so the test sequence has historical context.
    X_test_with_context = np.vstack([
        X_train_scaled[-sequence_length:],
        X_test_scaled,
    ])

    y_test_with_context = np.concatenate([
        y_train.values[-sequence_length:],
        y_test.values,
    ])

    X_test_seq, y_test_seq = create_sequences(
        X_test_with_context,
        y_test_with_context,
        sequence_length,
    )

    logger.info(
        f"LSTM train sequences: {len(X_train_seq)}"
    )

    logger.info(
        f"LSTM test sequences: {len(X_test_seq)}"
    )

    model = build_lstm(
        sequence_length,
        X_train.shape[1],
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
    )

    logger.info(
        "Training TensorFlow LSTM..."
    )

    history = model.fit(
        X_train_seq,
        y_train_seq,
        validation_split=0.15,
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1,
    )

    pred = model.predict(
        X_test_seq,
        verbose=0
    ).flatten()

    metrics = calculate_metrics(
        y_test_seq,
        pred
    )

    logger.info(
        f"LSTM: "
        f"RMSE={metrics['rmse']:.2f} | "
        f"MAE={metrics['mae']:.2f} | "
        f"R²={metrics['r2']:.4f}"
    )

    return (
        model,
        scaler,
        metrics,
        pred,
        history,
    )


# ---------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------

def run_shap(
    model,
    X_sample,
    feature_names,
    model_name,
):

    try:

        logger.info(
            f"Running SHAP for {model_name}..."
        )

        X_sample_df = pd.DataFrame(
            X_sample,
            columns=feature_names
        )

        if hasattr(
            model,
            "feature_importances_"
        ):

            explainer = shap.TreeExplainer(
                model
            )

        else:

            explainer = shap.LinearExplainer(
                model,
                X_sample_df
            )

        shap_values = explainer(
            X_sample_df
        )

        mean_abs = (
            np.abs(
                shap_values.values
            )
            .mean(axis=0)
        )

        importance = pd.Series(
            mean_abs,
            index=feature_names
        ).sort_values(
            ascending=False
        )

        # Bar plot
        top = importance.head(15).sort_values()

        plt.figure(
            figsize=(10, 7)
        )

        plt.barh(
            top.index,
            top.values
        )

        plt.xlabel(
            "Mean |SHAP value|"
        )

        plt.title(
            f"SHAP Feature Importance - {model_name}"
        )

        plt.tight_layout()

        output_path = (
            REPORTS_DIR
            / f"shap_bar_{model_name}.png"
        )

        plt.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        # JSON
        json_path = (
            REPORTS_DIR
            / f"shap_importance_{model_name}.json"
        )

        with open(
            json_path,
            "w"
        ) as f:

            json.dump(
                importance.round(6).to_dict(),
                f,
                indent=2
            )

        logger.info(
            f"Top SHAP features: "
            f"{list(importance.head(5).index)}"
        )

        return importance

    except Exception as e:

        logger.warning(
            f"SHAP failed for "
            f"{model_name}: {e}"
        )

        return None


# ---------------------------------------------------------------------
# EVALUATION PLOTS
# ---------------------------------------------------------------------

def save_evaluation_plots(
    y_test,
    predictions,
    best_model_name,
):

    ensure_dirs()

    best_pred = predictions[
        best_model_name
    ]

    # Actual vs predicted
    plt.figure(
        figsize=(14, 5)
    )

    plt.plot(
        y_test.values,
        label="Actual AQI"
    )

    plt.plot(
        best_pred,
        label="Predicted AQI",
        alpha=0.8
    )

    plt.title(
        f"Actual vs Predicted AQI - {best_model_name}"
    )

    plt.xlabel(
        "Test Time Step"
    )

    plt.ylabel(
        "AQI"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR
        / "actual_vs_predicted.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    # Residuals
    residuals = (
        y_test.values - best_pred
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.hist(
        residuals,
        bins=40
    )

    plt.axvline(
        0,
        linestyle="--"
    )

    plt.title(
        f"Residual Distribution - {best_model_name}"
    )

    plt.xlabel(
        "Actual - Predicted"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR
        / "residuals.png",
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


# ---------------------------------------------------------------------
# 72-HOUR FORECAST
# ---------------------------------------------------------------------

def generate_72_hour_forecast(
    model,
    scaler,
    last_features,
    feature_names,
):

    current = (
        last_features[
            feature_names
        ]
        .astype(float)
        .values
    )

    now = datetime.now().replace(
        minute=0,
        second=0,
        microsecond=0,
    )

    forecasts = []

    for hour in range(
        1,
        73
    ):

        X = scaler.transform(
            current.reshape(1, -1)
        )

        prediction = float(
            model.predict(X)[0]
        )

        prediction = max(
            0,
            min(500, prediction)
        )

        timestamp = (
            pd.Timestamp(now)
            + pd.Timedelta(hours=hour)
        )

        level, _ = get_aqi_level(
            prediction
        )

        forecasts.append({

            "predicted_for":
                timestamp.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "predicted_at":
                now.strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "city":
                CITY,

            "model_name":
                "best_model",

            "predicted_aqi":
                round(prediction, 2),

            "aqi_level":
                level,

            "is_hazardous":
                int(
                    prediction
                    >= ALERT_THRESHOLD
                ),
        })

        # -------------------------------------------------------------
        # Update AQI lag features
        # -------------------------------------------------------------

        feature_index = {
            name: i
            for i, name
            in enumerate(feature_names)
        }

        def idx(name):
            return feature_index.get(name)

        # Shift long lags first
        if idx("aqi_lag_24h") is not None:
            if idx("aqi_lag_12h") is not None:
                current[idx("aqi_lag_24h")] = (
                    current[idx("aqi_lag_12h")]
                )

        if idx("aqi_lag_12h") is not None:
            if idx("aqi_lag_6h") is not None:
                current[idx("aqi_lag_12h")] = (
                    current[idx("aqi_lag_6h")]
                )

        if idx("aqi_lag_6h") is not None:
            if idx("aqi_lag_3h") is not None:
                current[idx("aqi_lag_6h")] = (
                    current[idx("aqi_lag_3h")]
                )

        if idx("aqi_lag_3h") is not None:
            if idx("aqi_lag_1h") is not None:
                current[idx("aqi_lag_3h")] = (
                    current[idx("aqi_lag_1h")]
                )

        previous_aqi = current[
            idx("aqi_lag_1h")
        ] if idx("aqi_lag_1h") is not None else prediction

        if idx("aqi_lag_1h") is not None:
            current[idx("aqi_lag_1h")] = prediction

        # AQI change
        if idx("aqi_change_1h") is not None:
            current[idx("aqi_change_1h")] = (
                prediction - previous_aqi
            )

        # Rolling means
        for column in [
            "aqi_roll_mean_3h",
            "aqi_roll_mean_6h",
            "aqi_roll_mean_12h",
            "aqi_roll_mean_24h",
        ]:

            if idx(column) is not None:

                current[idx(column)] = (
                    0.8
                    * current[idx(column)]
                    + 0.2
                    * prediction
                )

        # Time features
        if idx("hour") is not None:
            current[idx("hour")] = timestamp.hour

        if idx("day") is not None:
            current[idx("day")] = timestamp.day

        if idx("month") is not None:
            current[idx("month")] = timestamp.month

        if idx("day_of_week") is not None:
            current[idx("day_of_week")] = (
                timestamp.weekday()
            )

        if idx("is_weekend") is not None:
            current[idx("is_weekend")] = int(
                timestamp.weekday() >= 5
            )

    return pd.DataFrame(
        forecasts
    )


# ---------------------------------------------------------------------
# SAVE MODEL ARTIFACTS
# ---------------------------------------------------------------------

def save_classical_models(
    models,
    scaler,
):

    ensure_dirs()

    joblib.dump(
        scaler,
        MODEL_REGISTRY["scaler"]
    )

    for name, model in models.items():

        if name in MODEL_REGISTRY:

            joblib.dump(
                model,
                MODEL_REGISTRY[name]
            )


# ---------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------

def run_training_pipeline(
    city=CITY,
    horizon_hours=24,
):

    ensure_dirs()

    logger.info("=" * 70)
    logger.info(
        "AQI TRAINING PIPELINE"
    )
    logger.info("=" * 70)

    # -------------------------------------------------------------
    # Load real feature data
    # -------------------------------------------------------------

    df = load_features(
        city,
        limit=10000
    )

    if df.empty:

        raise RuntimeError(
            "No feature data found. "
            "Run the real-data backfill first."
        )

    logger.info(
        f"Loaded {len(df)} feature rows"
    )

    # -------------------------------------------------------------
    # Prepare data
    # -------------------------------------------------------------

    X, y, feature_names = (
        prepare_dataset(
            df,
            horizon_hours
        )
    )

    # -------------------------------------------------------------
    # Chronological split
    # -------------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = chronological_split(
        X,
        y
    )

    # -------------------------------------------------------------
    # Classical models
    # -------------------------------------------------------------

    (
        classical_models,
        classical_results,
        classical_predictions,
        scaler,
        X_train_scaled,
        X_test_scaled,
    ) = train_classical_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    # -------------------------------------------------------------
    # Cross-validation
    # -------------------------------------------------------------

    for name, model in classical_models.items():

        cv_rmse = calculate_cv_rmse(
            model,
            X_train,
            y_train,
        )

        classical_results[name][
            "cv_rmse"
        ] = cv_rmse

    # -------------------------------------------------------------
    # TensorFlow LSTM
    # -------------------------------------------------------------

    (
        lstm_model,
        lstm_scaler,
        lstm_metrics,
        lstm_predictions,
        lstm_history,
    ) = train_lstm(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    classical_results[
        "tensorflow_lstm"
    ] = lstm_metrics

    # Keep predictions for evaluation
    classical_predictions[
        "tensorflow_lstm"
    ] = lstm_predictions

    # -------------------------------------------------------------
    # Find best classical model
    #
    # LSTM test target is sequence-aligned and therefore has a
    # slightly different length. Compare it using its own metrics.
    # -------------------------------------------------------------

    best_model_name = min(
        classical_results,
        key=lambda name:
            classical_results[name]["rmse"]
    )

    logger.info(
        f"BEST MODEL: {best_model_name}"
    )

    # -------------------------------------------------------------
    # SHAP
    # -------------------------------------------------------------

    shap_sample_size = min(
        200,
        len(X_test_scaled)
    )

    X_shap = X_test_scaled[
        :shap_sample_size
    ]

    # SHAP is especially useful for tree models.
    for model_name in [
        "random_forest",
        "xgboost",
    ]:

        if model_name in classical_models:

            run_shap(
                classical_models[model_name],
                X_shap,
                feature_names,
                model_name,
            )

    # Also explain Ridge
    run_shap(
        classical_models[
            "ridge_regression"
        ],
        X_shap,
        feature_names,
        "ridge_regression",
    )

    # -------------------------------------------------------------
    # Evaluation plots
    # -------------------------------------------------------------

    # Plot only classical predictions because they align exactly
    # with y_test.
    classical_plot_predictions = {
        name: pred
        for name, pred
        in classical_predictions.items()
        if name != "tensorflow_lstm"
    }

    classical_best_name = min(
        classical_plot_predictions,
        key=lambda name:
            classical_results[name]["rmse"]
    )

    save_evaluation_plots(
        y_test,
        classical_plot_predictions,
        classical_best_name,
    )

    # -------------------------------------------------------------
    # Save classical models
    # -------------------------------------------------------------

    save_classical_models(
        classical_models,
        scaler,
    )

    # Save LSTM
    lstm_path = (
        MODELS_DIR
        / "lstm_model.keras"
    )

    lstm_model.save(
        lstm_path
    )

    joblib.dump(
        lstm_scaler,
        MODELS_DIR
        / "lstm_scaler.pkl"
    )

    # -------------------------------------------------------------
    # Save metadata
    # -------------------------------------------------------------

    metadata = {

        "city":
            city,

        "best_model":
            best_model_name,

        "horizon_hours":
            horizon_hours,

        "feature_names":
            feature_names,

        "n_features":
            len(feature_names),

        "train_samples":
            len(X_train),

        "test_samples":
            len(X_test),

        "trained_at":
            datetime.now().isoformat(),

        "models":
            classical_results,
    }

    metadata_path = (
        MODELS_DIR
        / "model_metadata.json"
    )

    with open(
        metadata_path,
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    # -------------------------------------------------------------
    # Save metrics to local database
    # -------------------------------------------------------------

    for name, metrics in classical_results.items():

        metrics_record = {
            "model_name": name,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "r2": metrics["r2"],
            "cv_rmse": metrics.get(
                "cv_rmse",
                None
            ),
            "n_samples":
                len(X_train) + len(X_test),
            "trained_at":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            "is_best":
                int(
                    name == best_model_name
                ),
        }

        try:

            save_model_metrics(
                metrics_record
            )

        except Exception as e:

            logger.warning(
                f"Could not save metrics: {e}"
            )

    # -------------------------------------------------------------
    # 72-hour forecast
    #
    # Use the best CLASSICAL model because the generic iterative
    # forecasting function works directly with the tabular feature
    # representation.
    # -------------------------------------------------------------

    best_classical_model_name = min(
        classical_models,
        key=lambda name:
            classical_results[name]["rmse"]
    )

    best_classical_model = (
        classical_models[
            best_classical_model_name
        ]
    )

    last_features = (
        X.iloc[-1]
    )

    forecast_df = (
        generate_72_hour_forecast(
            best_classical_model,
            scaler,
            last_features,
            feature_names,
        )
    )

    # Save predictions
    try:

        save_predictions(
            forecast_df
        )

        logger.info(
            f"Saved {len(forecast_df)} forecast rows"
        )

    except Exception as e:

        logger.warning(
            f"Could not save forecast: {e}"
        )

    # -------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 70)

    print(
        f"{'Model':<22}"
        f"{'RMSE':>10}"
        f"{'MAE':>10}"
        f"{'R²':>10}"
    )

    print("-" * 52)

    for name, metrics in sorted(
        classical_results.items(),
        key=lambda item:
            item[1]["rmse"]
    ):

        marker = (
            "  <-- BEST"
            if name == best_model_name
            else ""
        )

        print(
            f"{name:<22}"
            f"{metrics['rmse']:>10.2f}"
            f"{metrics['mae']:>10.2f}"
            f"{metrics['r2']:>10.4f}"
            f"{marker}"
        )

    print()

    print(
        f"Best model: {best_model_name}"
    )

    print(
        f"72-hour forecast: "
        f"{len(forecast_df)} rows"
    )

    print(
        f"Models directory: "
        f"{MODELS_DIR}"
    )

    print(
        f"Reports directory: "
        f"{REPORTS_DIR}"
    )

    print("=" * 70)

    return {
        "best_model":
            best_model_name,

        "metrics":
            classical_results,

        "forecast":
            forecast_df,
    }


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=
        "Pearls AQI Predictor Training Pipeline"
    )

    parser.add_argument(
        "--city",
        default=CITY
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=24,
        help=
        "Prediction horizon in hours"
    )

    args = parser.parse_args()

    init_db()

    run_training_pipeline(
        city=args.city,
        horizon_hours=args.horizon,
    )