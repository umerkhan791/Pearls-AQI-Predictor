import streamlit as st
import requests
from datetime import datetime, timezone
import math
import os
import pandas as pd

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_URL = "http://127.0.0.1:8000/predict"
FORECAST_PATH = os.path.join("data", "forecast_72h.csv")

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #f0f2f8;
    color: #1a1f36;
}

.block-container {
    max-width: 1160px !important;
    padding: 0 24px 80px !important;
    margin: 0 auto;
}

#MainMenu, footer, header { visibility: hidden; }

/* ── NAV BAR ─────────────────────────────────────── */
.nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 22px 0 18px;
    margin-bottom: 8px;
    border-bottom: 1px solid rgba(26,31,54,0.08);
}
.nav-brand {
    display: flex;
    align-items: center;
    gap: 10px;
}
.nav-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 0 3px rgba(34,197,94,0.2);
    animation: pulse 2.4s ease-in-out infinite;
}
@keyframes pulse {
    0%,100% { box-shadow: 0 0 0 3px rgba(34,197,94,0.2); }
    50%      { box-shadow: 0 0 0 7px rgba(34,197,94,0.06); }
}
.nav-name {
    font-size: 15px;
    font-weight: 700;
    color: #1a1f36;
    letter-spacing: -0.3px;
}
.nav-right {
    display: flex;
    align-items: center;
    gap: 8px;
}
.nav-pill {
    background: white;
    border: 1px solid rgba(26,31,54,0.1);
    border-radius: 999px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 600;
    color: #64748b;
    letter-spacing: 0.2px;
}
.nav-pill.live {
    background: #f0fdf4;
    border-color: rgba(34,197,94,0.3);
    color: #16a34a;
}

/* ── HERO ─────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 45%, #0d3d37 100%);
    border-radius: 24px;
    padding: 52px 52px 48px;
    margin: 20px 0 32px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 360px; height: 360px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(34,197,94,0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 30%;
    width: 280px; height: 280px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #4ade80;
    margin-bottom: 16px;
}
.hero-title {
    font-size: 52px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -2px;
    line-height: 1.06;
    margin-bottom: 16px;
}
.hero-title span { color: #4ade80; }
.hero-sub {
    font-size: 17px;
    color: rgba(255,255,255,0.6);
    line-height: 1.65;
    max-width: 560px;
    font-weight: 400;
}
.hero-meta {
    display: flex;
    align-items: center;
    gap: 20px;
    margin-top: 30px;
    flex-wrap: wrap;
}
.hero-tag {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 600;
    color: rgba(255,255,255,0.55);
}
.hero-tag-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #4ade80;
    opacity: 0.7;
}

/* ── AQI STATUS BADGE ─────────────────────────────── */
.aqi-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 5px 13px 5px 8px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.2px;
}
.aqi-badge-dot { width: 8px; height: 8px; border-radius: 50%; }

/* ── MAIN AQI CARDS ──────────────────────────────── */
.aqi-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }

.aqi-main-card {
    background: white;
    border-radius: 20px;
    padding: 30px 30px 26px;
    border: 1px solid rgba(26,31,54,0.06);
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.04), 0 2px 4px -1px rgba(0,0,0,0.03);
    position: relative;
    overflow: hidden;
}
.aqi-main-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 20px 20px 0 0;
}
.aqi-main-card.good::after   { background: linear-gradient(90deg, #22c55e, #4ade80); }
.aqi-main-card.mod::after    { background: linear-gradient(90deg, #eab308, #facc15); }
.aqi-main-card.usg::after    { background: linear-gradient(90deg, #f97316, #fb923c); }
.aqi-main-card.bad::after    { background: linear-gradient(90deg, #ef4444, #f87171); }
.aqi-main-card.vbad::after   { background: linear-gradient(90deg, #8b5cf6, #a78bfa); }
.aqi-main-card.haz::after    { background: linear-gradient(90deg, #7f1d1d, #b91c1c); }

.card-eyebrow {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 18px;
}
.card-aqi-number {
    font-size: 76px;
    font-weight: 800;
    letter-spacing: -4px;
    line-height: 1;
    color: #0f172a;
    margin-bottom: 14px;
}
.card-footer {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    margin-top: 20px;
    padding-top: 16px;
    border-top: 1px solid #f1f5f9;
}
.card-time {
    font-size: 12px;
    color: #94a3b8;
    font-weight: 500;
}
.card-city {
    font-size: 12px;
    font-weight: 600;
    color: #64748b;
}

/* ── STAT STRIP ──────────────────────────────────── */
.stat-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 28px;
}
.stat-card {
    background: white;
    border-radius: 16px;
    padding: 20px 22px;
    border: 1px solid rgba(26,31,54,0.06);
    box-shadow: 0 2px 4px rgba(0,0,0,0.03);
}
.stat-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 8px;
}
.stat-value {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -1px;
    color: #0f172a;
    line-height: 1;
}
.stat-sub {
    font-size: 12px;
    color: #94a3b8;
    margin-top: 5px;
    font-weight: 500;
}
.stat-up   { color: #ef4444; }
.stat-down { color: #22c55e; }
.stat-flat { color: #64748b; }

/* ── SECTION LABEL ───────────────────────────────── */
.section-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #94a3b8;
    margin: 36px 0 14px;
}

/* ── DETAIL CARD ─────────────────────────────────── */
.detail-card {
    background: white;
    border-radius: 18px;
    padding: 0;
    border: 1px solid rgba(26,31,54,0.06);
    box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    overflow: hidden;
}
.detail-card-header {
    padding: 18px 24px;
    border-bottom: 1px solid #f1f5f9;
    font-size: 13px;
    font-weight: 700;
    color: #1a1f36;
    letter-spacing: -0.2px;
}
.detail-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 13px 24px;
    border-bottom: 1px solid #f8fafc;
}
.detail-row:last-child { border-bottom: none; }
.detail-key {
    font-size: 13px;
    color: #64748b;
    font-weight: 500;
}
.detail-val {
    font-size: 13px;
    font-weight: 700;
    color: #1a1f36;
    text-align: right;
}

/* ── POLLUTANT BARS ──────────────────────────────── */
.pol-card {
    background: white;
    border-radius: 18px;
    padding: 22px 24px;
    border: 1px solid rgba(26,31,54,0.06);
    box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    margin-bottom: 16px;
}
.pol-card-title {
    font-size: 13px;
    font-weight: 700;
    color: #1a1f36;
    margin-bottom: 18px;
}
.pol-row { margin-bottom: 14px; }
.pol-row:last-child { margin-bottom: 0; }
.pol-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 6px;
}
.pol-name { font-size: 12px; font-weight: 600; color: #475569; }
.pol-val  { font-size: 12px; font-weight: 700; color: #1a1f36; }
.pol-track {
    height: 5px;
    background: #f1f5f9;
    border-radius: 999px;
    overflow: hidden;
}
.pol-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 0.6s ease;
}

/* ── MODEL CARD ──────────────────────────────────── */
.model-card {
    background: white;
    border-radius: 18px;
    padding: 26px 28px;
    border: 1px solid rgba(26,31,54,0.06);
    box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    margin-bottom: 16px;
}
.model-name {
    font-size: 16px;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.4px;
    margin-bottom: 8px;
}
.model-desc {
    font-size: 13px;
    color: #64748b;
    line-height: 1.7;
    margin-bottom: 20px;
}
.model-metrics {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
}
.model-metric {
    background: #f8fafc;
    border-radius: 12px;
    padding: 14px 16px;
    border: 1px solid #f1f5f9;
}
.model-metric-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 6px;
}
.model-metric-value {
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.8px;
    color: #0f172a;
}

/* ── FEATURE TAGS ────────────────────────────────── */
.feature-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
    margin-top: 16px;
    padding-top: 16px;
    border-top: 1px solid #f1f5f9;
}
.feature-tag {
    background: #f0f9ff;
    border: 1px solid #e0f2fe;
    border-radius: 999px;
    padding: 4px 11px;
    font-size: 11px;
    font-weight: 600;
    color: #0369a1;
}

/* ── ALERT ───────────────────────────────────────── */
.alert-bar {
    border-radius: 14px;
    padding: 14px 20px;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 20px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
    line-height: 1.5;
}
.alert-bar.warn { background: #fffbeb; border: 1px solid #fcd34d; color: #92400e; }
.alert-bar.bad  { background: #fef2f2; border: 1px solid #fca5a5; color: #991b1b; }
.alert-icon { font-size: 16px; flex-shrink: 0; margin-top: 1px; }

/* ── FOOTER ──────────────────────────────────────── */
.site-footer {
    text-align: center;
    padding: 32px 0 0;
    margin-top: 48px;
    border-top: 1px solid rgba(26,31,54,0.08);
}
.site-footer p {
    font-size: 12px;
    color: #94a3b8;
    line-height: 2;
    font-weight: 500;
}

/* ── STREAMLIT OVERRIDES ─────────────────────────── */
[data-testid="stVerticalBlock"] > div { gap: 0 !important; }
.stButton > button {
    width: 100%;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    background: white;
    color: #334155;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 13px;
    height: 44px;
    transition: all 0.15s;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.stButton > button:hover {
    border-color: #22c55e;
    color: #15803d;
    background: #f0fdf4;
    box-shadow: 0 4px 12px rgba(34,197,94,0.12);
}

div[data-testid="column"] { padding: 0 6px !important; }
div[data-testid="column"]:first-child { padding-left: 0 !important; }
div[data-testid="column"]:last-child  { padding-right: 0 !important; }

/* hide streamlit metric widget — we draw our own */
[data-testid="stMetric"] { display: none !important; }

@media (max-width: 768px) {
    .hero { padding: 34px 28px; border-radius: 18px; }
    .hero-title { font-size: 36px; }
    .aqi-grid { grid-template-columns: 1fr; }
    .stat-strip { grid-template-columns: 1fr 1fr; }
    .model-metrics { grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

def fmt_time(value):
    if not value:
        return "—"
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.strftime("%d %b %Y, %H:%M UTC")
    except Exception:
        return str(value)


def get_category(aqi):
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Moderate"
    if aqi <= 150:  return "Unhealthy for SG"
    if aqi <= 200:  return "Unhealthy"
    if aqi <= 300:  return "Very Unhealthy"
    return "Hazardous"


AQI_STYLES = {
    "Good":            ("good",  "#22c55e", "#f0fdf4", "#166534"),
    "Moderate":        ("mod",   "#eab308", "#fefce8", "#854d0e"),
    "Unhealthy for SG":("usg",   "#f97316", "#fff7ed", "#9a3412"),
    "Unhealthy":       ("bad",   "#ef4444", "#fef2f2", "#991b1b"),
    "Very Unhealthy":  ("vbad",  "#8b5cf6", "#f5f3ff", "#5b21b6"),
    "Hazardous":       ("haz",   "#dc2626", "#fef2f2", "#7f1d1d"),
}

def aqi_style(cat):
    return AQI_STYLES.get(cat, ("mod", "#eab308", "#fefce8", "#854d0e"))


HEALTH_TIPS = {
    "Good":             ("✅", "Air quality is satisfactory. Great day for outdoor activity.", ""),
    "Moderate":         ("🟡", "Acceptable air quality. Unusually sensitive people should consider limiting prolonged outdoor exertion.", "warn"),
    "Unhealthy for SG": ("🟠", "Sensitive groups — children, elderly, those with respiratory conditions — should limit prolonged outdoor activity.", "warn"),
    "Unhealthy":        ("🔴", "Everyone may begin experiencing health effects. Limit prolonged outdoor exertion.", "bad"),
    "Very Unhealthy":   ("🔴", "Health alert: everyone should avoid prolonged outdoor activities. Wear an N95 mask if you must go out.", "bad"),
    "Hazardous":        ("🚨", "Emergency conditions. Stay indoors, seal windows, run air purifiers on maximum. Avoid all outdoor exposure.", "bad"),
}


def pol_bar(name, value, max_val, color):
    pct = min(100, round(value / max_val * 100)) if max_val else 0
    return f"""
<div class="pol-row">
    <div class="pol-header">
        <span class="pol-name">{name}</span>
        <span class="pol-val">{value:.1f} µg/m³</span>
    </div>
    <div class="pol-track">
        <div class="pol-fill" style="width:{pct}%;background:{color};"></div>
    </div>
</div>"""


def fetch_prediction():
    r = requests.get(API_URL, timeout=15)
    r.raise_for_status()
    return r.json()


def load_72h_forecast():
    """Load and validate the locally generated 72-hour recursive forecast."""
    if not os.path.exists(FORECAST_PATH):
        return pd.DataFrame()

    try:
        df = pd.read_csv(FORECAST_PATH)
    except Exception:
        return pd.DataFrame()

    if df.empty or "predicted_aqi" not in df.columns:
        return pd.DataFrame()

    # The generator writes "predicted_for". Accept "timestamp" too
    # so the dashboard remains compatible with older forecast files.
    if "predicted_for" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["predicted_for"], errors="coerce"
        )
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(
            df["timestamp"], errors="coerce"
        )
    else:
        return pd.DataFrame()

    df["predicted_aqi"] = pd.to_numeric(
        df["predicted_aqi"], errors="coerce"
    )

    df = (
        df.dropna(subset=["timestamp", "predicted_aqi"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Only display a valid 72-hour sequence.
    if len(df) != 72:
        return pd.DataFrame()

    expected = pd.date_range(
        start=df["timestamp"].iloc[0],
        periods=72,
        freq="h",
    )

    if df["timestamp"].tolist() != expected.tolist():
        return pd.DataFrame()

    return df


# ============================================================
# NAV
# ============================================================

now_str = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")

st.markdown(f"""
<div class="nav">
    <div class="nav-brand">
        <div class="nav-dot"></div>
        <span class="nav-name">Pearls AQI Predictor</span>
    </div>
    <div class="nav-right">
        <span class="nav-pill live">● Model Online</span>
        <span class="nav-pill">📍 Karachi, PK</span>
        <span class="nav-pill">{now_str}</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# HERO
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Air Quality Intelligence</div>
    <div class="hero-title">Karachi AQI<br><span>Forecast.</span></div>
    <div class="hero-sub">
        Air quality monitoring and next-hour prediction powered by
        machine learning, engineered features, and the Hopsworks
        Feature Store.
    </div>
    <div class="hero-meta">
        <div class="hero-tag"><div class="hero-tag-dot"></div>Random Forest · R² 0.994</div>
        <div class="hero-tag"><div class="hero-tag-dot"></div>29 engineered features</div>
        <div class="hero-tag"><div class="hero-tag-dot"></div>Hopsworks Feature Store</div>
        <div class="hero-tag"><div class="hero-tag-dot"></div>FastAPI backend</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

# Fetch the current API result on every Streamlit run.
# This prevents an old prediction from remaining in session_state after
# the model/API has been retrained or updated.
try:
    with st.spinner("Connecting to prediction service…"):
        data = fetch_prediction()
except requests.exceptions.ConnectionError:
    st.error("FastAPI server is not running. Start it on 127.0.0.1:8000 and refresh.")
    st.stop()
except requests.exceptions.Timeout:
    st.error("Request timed out (15 s). Make sure the FastAPI server is running.")
    st.stop()
except Exception as e:
    st.error(f"Error: {e}")
    st.stop()
current_aqi = float(data.get("current_aqi", 0))
predicted_aqi = float(data.get("predicted_aqi", 0))
current_time = data.get("timestamp")
pred_time = data.get("prediction_for")

current_cat = get_category(current_aqi)
pred_cat = data.get("category", get_category(predicted_aqi))

change = predicted_aqi - current_aqi

if change > 0.05:
    trend, trend_sym, trend_cls = "Increasing", "^", "stat-up"
elif change < -0.05:
    trend, trend_sym, trend_cls = "Decreasing", "v", "stat-down"
else:
    trend, trend_sym, trend_cls = "Stable", "-", "stat-flat"

metrics = data.get("model_metrics", {})

mae = metrics.get("mae")
rmse = metrics.get("rmse")
r2 = metrics.get("r2")

model_name = data.get("model", "RandomForestRegressor")
input_features = 29

pollutants = data.get("pollutants", {})
required_pollutants = ["pm25", "pm10", "o3", "no2", "so2", "co"]

missing_pollutants = [
    p for p in required_pollutants if p not in pollutants
]

if missing_pollutants:
    st.error(
        "API response is missing real pollutant values: "
        + ", ".join(missing_pollutants)
    )
    st.stop()

pm25 = float(pollutants["pm25"])
pm10 = float(pollutants["pm10"])
o3   = float(pollutants["o3"])
no2  = float(pollutants["no2"])
so2  = float(pollutants["so2"])
co   = float(pollutants["co"])


# ============================================================
# ALERT BANNER
# ============================================================

icon, tip, alert_cls = HEALTH_TIPS.get(current_cat, ("🟡", "", "warn"))
if alert_cls:
    st.markdown(f"""
<div class="alert-bar {alert_cls}">
    <div class="alert-icon">{icon}</div>
    <div><strong>{current_cat}:</strong> {tip}</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# CURRENT + FORECAST CARDS
# ============================================================

cc, pc = aqi_style(current_cat), aqi_style(pred_cat)

left, right = st.columns(2, gap="medium")

with left:
    st.markdown(f"""
<div class="aqi-main-card {cc[0]}">
    <div class="card-eyebrow">Current Air Quality</div>
    <div class="card-aqi-number">{current_aqi:.0f}</div>
    <div class="aqi-badge" style="background:{cc[2]};color:{cc[3]};">
        <div class="aqi-badge-dot" style="background:{cc[1]};"></div>
        {current_cat}
    </div>
    <div class="card-footer">
        <span class="card-time">{fmt_time(current_time)}</span>
        <span class="card-city">Karachi, PK</span>
    </div>
</div>
""", unsafe_allow_html=True)

with right:
    sign = "+" if change > 0.05 else ("" if abs(change) <= 0.05 else "")
    st.markdown(f"""
<div class="aqi-main-card {pc[0]}">
    <div class="card-eyebrow">Next-Hour Forecast</div>
    <div class="card-aqi-number">{predicted_aqi:.1f}</div>
    <div class="aqi-badge" style="background:{pc[2]};color:{pc[3]};">
        <div class="aqi-badge-dot" style="background:{pc[1]};"></div>
        {pred_cat}
    </div>
    <div class="card-footer">
        <span class="card-time">{fmt_time(pred_time)}</span>
        <span class="card-city">{sign}{change:.2f} from current</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# STAT STRIP
# ============================================================

st.markdown('<div class="section-label">Forecast Analysis</div>', unsafe_allow_html=True)

s1, s2, s3, s4 = st.columns(4, gap="medium")

with s1:
    st.markdown(f"""
<div class="stat-card">
    <div class="stat-label">Current AQI</div>
    <div class="stat-value">{current_aqi:.0f}</div>
    <div class="stat-sub">{current_cat}</div>
</div>
""", unsafe_allow_html=True)

with s2:
    st.markdown(f"""
<div class="stat-card">
    <div class="stat-label">Predicted AQI</div>
    <div class="stat-value">{predicted_aqi:.1f}</div>
    <div class="stat-sub">{pred_cat}</div>
</div>
""", unsafe_allow_html=True)

with s3:
    st.markdown(f"""
<div class="stat-card">
    <div class="stat-label">AQI Change</div>
    <div class="stat-value {trend_cls}">{change:+.2f}</div>
    <div class="stat-sub">next hour</div>
</div>
""", unsafe_allow_html=True)

with s4:
    st.markdown(f"""
<div class="stat-card">
    <div class="stat-label">Trend</div>
    <div class="stat-value {trend_cls}" style="font-size:22px;letter-spacing:-0.5px;">{trend_sym} {trend}</div>
    <div class="stat-sub">direction</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 72-HOUR FORECAST
# ============================================================

forecast_df = load_72h_forecast()

if not forecast_df.empty:
    st.markdown(
        '<div class="section-label">72-Hour Forecast</div>',
        unsafe_allow_html=True
    )

    # IMPORTANT:
    # Use the real timestamp column as X.
    # Do not use the dataframe row index, otherwise Streamlit
    # displays -5, 0, 5, 10... instead of forecast dates/times.
    chart_df = forecast_df[
        ["timestamp", "predicted_aqi"]
    ].copy()

    st.line_chart(
        chart_df,
        x="timestamp",
        y="predicted_aqi",
        height=330,
        use_container_width=True,
    )

    # Daily forecast averages
    daily = forecast_df.copy()
    daily["date"] = daily["timestamp"].dt.strftime("%d %b")

    daily_avg = (
        daily.groupby("date", sort=False)["predicted_aqi"]
        .mean()
        .round(1)
    )

    cols = st.columns(3, gap="medium")

    for i in range(3):
        if i < len(daily_avg):
            day = daily_avg.index[i]
            avg = daily_avg.iloc[i]

            with cols[i]:
                st.markdown(
                    f"""
                    <div class="stat-card">
                        <div class="stat-label">
                            Day {i + 1} · {day}
                        </div>
                        <div class="stat-value">
                            {avg:.1f}
                        </div>
                        <div class="stat-sub">
                            average forecast AQI
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    first_forecast = float(
        forecast_df.iloc[0]["predicted_aqi"]
    )
    last_forecast = float(
        forecast_df.iloc[-1]["predicted_aqi"]
    )

    start_time = forecast_df.iloc[0]["timestamp"]
    end_time = forecast_df.iloc[-1]["timestamp"]

    st.caption(
        f"Forecast period: {start_time.strftime('%d %b %Y, %H:%M')} "
        f"→ {end_time.strftime('%d %b %Y, %H:%M')}. "
        f"First hour: {first_forecast:.2f} AQI · "
        f"Final hour: {last_forecast:.2f} AQI. "
        "The first hour uses the latest observed feature row; later "
        "hours are generated recursively. Future pollutant and weather "
        "inputs are carried forward from the latest observation, so "
        "this is a model forecast, not future measured data."
    )
else:
    st.info(
        "A valid 72-hour forecast was not found. "
        "Run generate_72h_forecast.py to create forecast_72h.csv."
    )


# ============================================================
# DETAIL ROWS + POLLUTANTS
# ============================================================

st.markdown('<div class="section-label">Observation Details</div>', unsafe_allow_html=True)

dl, dr = st.columns(2, gap="medium")

with dl:
    st.markdown(f"""
<div class="detail-card">
    <div class="detail-card-header">📊 Measurement</div>
    <div class="detail-row">
        <span class="detail-key">Location</span>
        <span class="detail-val">Karachi, Pakistan</span>
    </div>
    <div class="detail-row">
        <span class="detail-key">Observation time</span>
        <span class="detail-val">{fmt_time(current_time)}</span>
    </div>
    <div class="detail-row">
        <span class="detail-key">Current AQI</span>
        <span class="detail-val">{current_aqi:.2f}</span>
    </div>
    <div class="detail-row">
        <span class="detail-key">Current category</span>
        <span class="detail-val">{current_cat}</span>
    </div>
    <div class="detail-row">
        <span class="detail-key">Predicted AQI</span>
        <span class="detail-val">{predicted_aqi:.2f}</span>
    </div>
    <div class="detail-row">
        <span class="detail-key">Forecast time</span>
        <span class="detail-val">{fmt_time(pred_time)}</span>
    </div>
    <div class="detail-row">
        <span class="detail-key">Expected change</span>
        <span class="detail-val">{change:+.2f} AQI points</span>
    </div>
    <div class="detail-row">
        <span class="detail-key">Horizon</span>
        <span class="detail-val">1 hour</span>
    </div>
</div>
""", unsafe_allow_html=True)

with dr:
    pm25_bar = pol_bar("PM 2.5",  pm25, 250, "#ef4444")
    pm10_bar = pol_bar("PM 10",   pm10, 350, "#f97316")
    o3_bar   = pol_bar("O₃",      o3,   200, "#8b5cf6")
    no2_bar  = pol_bar("NO₂",     no2,  200, "#3b82f6")
    so2_bar  = pol_bar("SO₂",     so2,  150, "#eab308")
    co_bar   = pol_bar("CO",      co,   50,  "#6b7280")

    st.markdown(f"""
<div class="pol-card">
    <div class="pol-card-title">🧪 Pollutant Concentrations</div>
    {pm25_bar}
    {pm10_bar}
    {o3_bar}
    {no2_bar}
    {so2_bar}
    {co_bar}
</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL CARD
# ============================================================

st.markdown('<div class="section-label">Machine Learning Model</div>', unsafe_allow_html=True)

mae_str  = f"{float(mae):.4f}"  if mae  is not None else "—"
rmse_str = f"{float(rmse):.4f}" if rmse is not None else "—"
r2_str   = f"{float(r2):.4f}"   if r2   is not None else "—"

features_list = [
    "hour","day","month","day_of_week","is_weekend",
    "aqi_lag_1h","aqi_lag_3h","aqi_lag_6h","aqi_lag_12h","aqi_lag_24h",
    "aqi_roll_mean_3h","aqi_roll_mean_6h","aqi_roll_mean_12h","aqi_roll_mean_24h",
    "aqi_roll_std_6h","aqi_roll_std_24h","aqi_change_1h","aqi_change_6h","aqi_change_24h",
    "pm25","pm10","o3","no2","so2","co","temperature","humidity","pressure","wind",
]
tags = "".join(f'<span class="feature-tag">{f}</span>' for f in features_list)

st.markdown(f"""
<div class="model-card">
    <div class="model-name">{model_name}</div>
    <div class="model-desc">
        Trained on {input_features} engineered features sourced from the Hopsworks Feature Store.
        The FastAPI service uses a local SQLite feature-store cache for fast inference and reliability.
        Uses chronological train/test splitting to prevent data leakage and predicts one hour ahead
        from the latest available observation.
    </div>
    <div class="model-metrics">
        <div class="model-metric">
            <div class="model-metric-label">MAE</div>
            <div class="model-metric-value">{mae_str}</div>
        </div>
        <div class="model-metric">
            <div class="model-metric-label">RMSE</div>
            <div class="model-metric-value">{rmse_str}</div>
        </div>
        <div class="model-metric">
            <div class="model-metric-label">R²</div>
            <div class="model-metric-value">{r2_str}</div>
        </div>
    </div>
    <div class="feature-tags">{tags}</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# REFRESH
# ============================================================

st.markdown('<div class="section-label">Controls</div>', unsafe_allow_html=True)

col_btn, col_pad = st.columns([1, 3])
with col_btn:
    if st.button("↻  Refresh prediction"):
        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown(f"""
<div class="site-footer">
    <p>
        <strong>Pearls AQI Predictor</strong> — Karachi Air Quality Forecasting<br>
        Hopsworks Feature Store &nbsp;·&nbsp; Random Forest Regression
        &nbsp;·&nbsp; FastAPI &nbsp;·&nbsp; Streamlit<br>
        Last refreshed: {now_str}
    </p>
</div>
""", unsafe_allow_html=True)
