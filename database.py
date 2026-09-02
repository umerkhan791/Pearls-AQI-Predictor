"""
database.py — SQLite Feature Store
Acts as our local replacement for Hopsworks/Vertex AI.
Stores raw readings, engineered features, predictions, and model metrics.
"""

import sqlite3
import pandas as pd
from datetime import datetime
from loguru import logger
from config import DB_PATH, ensure_dirs


def get_connection() -> sqlite3.Connection:
    ensure_dirs()
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create all tables if they don't exist."""
    conn = get_connection()
    cur = conn.cursor()

    # ── Raw readings from AQICN API ──────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS raw_readings (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT NOT NULL,
            city        TEXT NOT NULL,
            aqi         REAL,
            pm25        REAL,
            pm10        REAL,
            o3          REAL,
            no2         REAL,
            so2         REAL,
            co          REAL,
            temperature REAL,
            humidity    REAL,
            pressure    REAL,
            wind        REAL,
            dominant_pol TEXT,
            fetched_at  TEXT DEFAULT (datetime('now')),
            UNIQUE(timestamp, city)
        )
    """)

    # ── Engineered features (inputs to ML model) ─────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS features (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            city            TEXT NOT NULL,
            aqi             REAL,
            -- Time features
            hour            INTEGER,
            day             INTEGER,
            month           INTEGER,
            day_of_week     INTEGER,
            is_weekend      INTEGER,
            -- Lag features
            aqi_lag_1h      REAL,
            aqi_lag_3h      REAL,
            aqi_lag_6h      REAL,
            aqi_lag_12h     REAL,
            aqi_lag_24h     REAL,
            -- Rolling stats
            aqi_roll_mean_3h  REAL,
            aqi_roll_mean_6h  REAL,
            aqi_roll_mean_12h REAL,
            aqi_roll_mean_24h REAL,
            aqi_roll_std_6h   REAL,
            aqi_roll_std_24h  REAL,
            -- AQI change rate
            aqi_change_1h   REAL,
            aqi_change_6h   REAL,
            aqi_change_24h  REAL,
            -- Pollutants (normalized)
            pm25            REAL,
            pm10            REAL,
            o3              REAL,
            no2             REAL,
            so2             REAL,
            co              REAL,
            -- Weather
            temperature     REAL,
            humidity        REAL,
            pressure        REAL,
            wind            REAL,
            created_at      TEXT DEFAULT (datetime('now')),
            UNIQUE(timestamp, city)
        )
    """)

    # ── Model predictions ─────────────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            predicted_for   TEXT NOT NULL,   -- future timestamp
            predicted_at    TEXT NOT NULL,   -- when prediction was made
            city            TEXT NOT NULL,
            model_name      TEXT NOT NULL,
            predicted_aqi   REAL NOT NULL,
            aqi_level       TEXT,
            is_hazardous    INTEGER DEFAULT 0,
            UNIQUE(predicted_for, city, model_name)
        )
    """)

    # ── Model performance registry ────────────────────────────────────────────
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name  TEXT NOT NULL,
            trained_at  TEXT NOT NULL,
            rmse        REAL,
            mae         REAL,
            r2          REAL,
            n_samples   INTEGER,
            is_best     INTEGER DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {DB_PATH}")


def save_raw_reading(data: dict):
    """Insert one raw API reading. Ignores duplicates (same timestamp+city)."""
    conn = get_connection()
    try:
        conn.execute("""
            INSERT OR IGNORE INTO raw_readings
            (timestamp, city, aqi, pm25, pm10, o3, no2, so2, co,
             temperature, humidity, pressure, wind, dominant_pol)
            VALUES
            (:timestamp, :city, :aqi, :pm25, :pm10, :o3, :no2, :so2, :co,
             :temperature, :humidity, :pressure, :wind, :dominant_pol)
        """, data)
        conn.commit()
    finally:
        conn.close()


def save_features(df: pd.DataFrame):
    """Write a features DataFrame to the features table, skipping duplicates."""
    conn = get_connection()
    try:
        df.to_sql("features_tmp", conn, if_exists="replace", index=False)
        # Get column names that exist in the features table
        cols = [row[1] for row in conn.execute("PRAGMA table_info(features)").fetchall()
                if row[1] not in ("id", "created_at")]
        available = [c for c in cols if c in df.columns]
        col_str = ", ".join(available)
        conn.execute(f"""
            INSERT OR IGNORE INTO features ({col_str})
            SELECT {col_str} FROM features_tmp
        """)
        conn.execute("DROP TABLE IF EXISTS features_tmp")
        conn.commit()
    except Exception as e:
        logger.warning(f"Feature save warning: {e}")
    finally:
        conn.close()


def load_features(city: str, limit: int = 5000) -> pd.DataFrame:
    """Load features for a city, most recent first."""
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM features WHERE city=? ORDER BY timestamp DESC LIMIT ?",
        conn, params=(city, limit)
    )
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp")


def load_raw_readings(city: str, limit: int = 5000) -> pd.DataFrame:
    """Load raw readings for a city."""
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM raw_readings WHERE city=? ORDER BY timestamp DESC LIMIT ?",
        conn, params=(city, limit)
    )
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp")


def save_predictions(df: pd.DataFrame):
    """Save model predictions to the predictions table."""
    conn = get_connection()
    try:
        df.to_sql("predictions", conn, if_exists="append", index=False,
                  )
    except Exception as e:
        logger.warning(f"Prediction save warning: {e}")
    finally:
        conn.close()


def load_predictions(city: str, model_name: str = "best_model") -> pd.DataFrame:
    """Load latest predictions for a city."""
    conn = get_connection()
    df = pd.read_sql("""
        SELECT * FROM predictions
        WHERE city=? AND model_name=?
        ORDER BY predicted_for ASC
    """, conn, params=(city, model_name))
    conn.close()
    return df


def save_model_metrics(metrics: dict):
    """Log model training metrics."""
    conn = get_connection()
    conn.execute("""
        INSERT INTO model_metrics (model_name, trained_at, rmse, mae, r2, n_samples, is_best)
        VALUES (:model_name, :trained_at, :rmse, :mae, :r2, :n_samples, :is_best)
    """, metrics)
    conn.commit()
    conn.close()


def get_feature_count(city: str) -> int:
    """Return how many feature rows exist for a city."""
    conn = get_connection()
    count = conn.execute(
        "SELECT COUNT(*) FROM features WHERE city=?", (city,)
    ).fetchone()[0]
    conn.close()
    return count


if __name__ == "__main__":
    init_db()
    print(f"✅ Feature store ready at: {DB_PATH}")
