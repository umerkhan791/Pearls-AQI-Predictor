import streamlit as st
import altair as alt
import requests
from datetime import datetime, timezone
import math
import json
import os
import pandas as pd
from zoneinfo import ZoneInfo

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Pearls AQI Predictor",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_URL = "https://pearls-aqi-api-xc81.onrender.com/predict"
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
    max-width: 1200px !important;
    padding: 0 28px 80px !important;
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
    flex-wrap: wrap;
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
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.2px;
    color: #1a1f36;
    margin: 36px 0 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(26,31,54,0.06);
}
.section-label-sm {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #94a3b8;
    margin: 28px 0 12px;
}

/* ── DETAIL CARD ─────────────────────────────────── */
.detail-card {
    background: white;
    border-radius: 18px;
    padding: 0;
    border: 1px solid rgba(26,31,54,0.06);
    box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    overflow: hidden;
    height: 100%;
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

/* ── WEATHER CARD ────────────────────────────────── */
.weather-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 16px;
}
.weather-card {
    background: white;
    border-radius: 16px;
    padding: 20px 20px 18px;
    border: 1px solid rgba(26,31,54,0.06);
    box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    text-align: center;
}
.weather-icon { font-size: 26px; margin-bottom: 10px; display: block; }
.weather-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 6px;
}
.weather-value {
    font-size: 22px;
    font-weight: 800;
    letter-spacing: -0.8px;
    color: #0f172a;
    line-height: 1;
}
.weather-unit {
    font-size: 11px;
    font-weight: 600;
    color: #94a3b8;
    margin-top: 3px;
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
    align-items: center;
}
.pol-name { font-size: 12px; font-weight: 600; color: #475569; }
.pol-val  { font-size: 12px; font-weight: 700; color: #1a1f36; }
.pol-status {
    font-size: 10px;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 999px;
    letter-spacing: 0.3px;
}
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

/* ── AQI SCALE ───────────────────────────────────── */
.aqi-scale-wrap {
    background: white;
    border-radius: 18px;
    padding: 22px 24px;
    border: 1px solid rgba(26,31,54,0.06);
    box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    margin-bottom: 16px;
}
.aqi-scale-title {
    font-size: 13px;
    font-weight: 700;
    color: #1a1f36;
    margin-bottom: 16px;
}
.aqi-scale-bar {
    display: flex;
    border-radius: 8px;
    overflow: hidden;
    height: 10px;
    margin-bottom: 8px;
}
.aqi-seg {
    flex: 1;
}
.aqi-scale-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 4px;
}
.aqi-scale-label {
    font-size: 10px;
    font-weight: 600;
    color: #94a3b8;
    text-align: center;
    flex: 1;
    line-height: 1.3;
}
.aqi-scale-pointer {
    font-size: 11px;
    font-weight: 700;
    color: #0f172a;
    margin-top: 8px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.aqi-pointer-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
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
    margin-bottom: 6px;
}
.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: #f0fdf4;
    border: 1px solid rgba(34,197,94,0.25);
    border-radius: 999px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 700;
    color: #15803d;
    margin-bottom: 14px;
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
    margin-bottom: 16px;
}
.model-metric {
    background: #f8fafc;
    border-radius: 12px;
    padding: 16px 18px;
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
.model-metric-note {
    font-size: 10px;
    color: #94a3b8;
    margin-top: 4px;
    font-weight: 500;
}
.model-metrics-disclaimer {
    font-size: 11px;
    color: #94a3b8;
    margin-bottom: 16px;
    line-height: 1.5;
    padding: 10px 14px;
    background: #f8fafc;
    border-radius: 8px;
    border-left: 3px solid #e2e8f0;
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
.alert-bar.haz  { background: #7f1d1d; border: 1px solid #991b1b; color: #fef2f2; }
.alert-icon { font-size: 16px; flex-shrink: 0; margin-top: 1px; }

/* ── ARCHITECTURE ────────────────────────────────── */
.arch-card {
    background: white;
    border-radius: 18px;
    padding: 26px 28px;
    border: 1px solid rgba(26,31,54,0.06);
    box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    margin-bottom: 16px;
}
.arch-flow {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 16px;
}
.arch-node {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 12px;
    font-weight: 700;
    color: #334155;
    white-space: nowrap;
}
.arch-node.highlight {
    background: #f0fdf4;
    border-color: rgba(34,197,94,0.3);
    color: #15803d;
}
.arch-arrow {
    font-size: 14px;
    color: #cbd5e1;
    font-weight: 700;
}

/* ── DATA SOURCES ────────────────────────────────── */
.source-card {
    background: white;
    border-radius: 18px;
    padding: 22px 24px;
    border: 1px solid rgba(26,31,54,0.06);
    box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    margin-bottom: 16px;
}
.source-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 12px 0;
    border-bottom: 1px solid #f1f5f9;
}
.source-item:last-child { border-bottom: none; }
.source-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #3b82f6;
    margin-top: 4px;
    flex-shrink: 0;
}
.source-title {
    font-size: 13px;
    font-weight: 700;
    color: #1a1f36;
    margin-bottom: 3px;
}
.source-desc {
    font-size: 12px;
    color: #64748b;
    line-height: 1.5;
}

/* ── METHODOLOGY ─────────────────────────────────── */
.method-card {
    background: white;
    border-radius: 18px;
    padding: 26px 28px;
    border: 1px solid rgba(26,31,54,0.06);
    box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    margin-bottom: 16px;
}
.method-step {
    display: flex;
    gap: 14px;
    margin-bottom: 16px;
}
.method-step:last-child { margin-bottom: 0; }
.method-num {
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: #f0f9ff;
    border: 1px solid #e0f2fe;
    font-size: 11px;
    font-weight: 800;
    color: #0369a1;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
}
.method-text {
    font-size: 13px;
    color: #475569;
    line-height: 1.6;
    font-weight: 500;
}
.method-text strong {
    color: #1a1f36;
    font-weight: 700;
}

/* ── FORECAST TABLE ──────────────────────────────── */
.forecast-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
}
.forecast-table th {
    text-align: left;
    padding: 10px 14px;
    background: #f8fafc;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #94a3b8;
    border-bottom: 1px solid #f1f5f9;
}
.forecast-table td {
    padding: 10px 14px;
    border-bottom: 1px solid #f8fafc;
    font-weight: 600;
    color: #1a1f36;
}
.forecast-table tr:last-child td { border-bottom: none; }
.forecast-table tr:hover td { background: #fafbfc; }

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
    .weather-grid { grid-template-columns: 1fr 1fr; }
    .arch-flow { flex-direction: column; align-items: flex-start; }
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
        return dt.strftime("%d %b %Y, %H:%M PKT")
    except Exception:
        return str(value)


def get_category(aqi):
    if aqi <= 50:   return "Good"
    if aqi <= 100:  return "Moderate"
    if aqi <= 150:  return "Unhealthy for Sensitive Groups"
    if aqi <= 200:  return "Unhealthy"
    if aqi <= 300:  return "Very Unhealthy"
    return "Hazardous"


AQI_STYLES = {
    "Good":             ("good", "#22c55e", "#f0fdf4", "#166534"),
    "Moderate":         ("mod", "#eab308", "#fefce8", "#854d0e"),
    "Unhealthy for Sensitive Groups": ("usg", "#f97316", "#fff7ed", "#9a3412"),
    "Unhealthy":        ("bad", "#ef4444", "#fef2f2", "#991b1b"),
    "Very Unhealthy":   ("vbad", "#dc2626", "#fef2f2", "#991b1b"),
    "Hazardous":        ("haz", "#7f1d1d", "#fef2f2", "#7f1d1d"),
}

def aqi_style(cat):
    return AQI_STYLES.get(cat, ("mod", "#eab308", "#fefce8", "#854d0e"))


HEALTH_TIPS = {
    "Good":             ("✓", "Air quality is satisfactory. Great day for outdoor activity.", ""),
    "Moderate":         ("◉", "Acceptable air quality. Unusually sensitive people should consider limiting prolonged outdoor exertion.", "warn"),
    "Unhealthy for Sensitive Groups": ("⚠", "Sensitive groups — children, elderly, and those with respiratory conditions — should limit prolonged outdoor activity.", "warn"),
    "Unhealthy":        ("⚠", "Everyone may begin experiencing health effects. Limit prolonged outdoor exertion.", "bad"),
    "Very Unhealthy":   ("⛔", "Health alert: everyone should avoid prolonged outdoor activities. Wear an N95 mask if you must go out.", "bad"),
    "Hazardous":        ("☣", "Emergency conditions. Stay indoors, seal windows, run air purifiers on maximum. Avoid all outdoor exposure.", "haz"),
}


def pol_bar(name, value, max_val, color, unit="µg/m³"):
    pct = min(100, round(value / max_val * 100)) if max_val else 0
    # status label based on % of reference
    if pct < 30:
        status_bg, status_color, status_txt = "#f0fdf4", "#15803d", "Low"
    elif pct < 65:
        status_bg, status_color, status_txt = "#fefce8", "#854d0e", "Moderate"
    else:
        status_bg, status_color, status_txt = "#fef2f2", "#991b1b", "Elevated"
    return f"""
<div class="pol-row">
    <div class="pol-header">
        <span class="pol-name">{name}</span>
        <span style="display:flex;gap:8px;align-items:center;">
            <span class="pol-status" style="background:{status_bg};color:{status_color};">{status_txt}</span>
            <span class="pol-val">{value:.1f} {unit}</span>
        </span>
    </div>
    <div class="pol-track">
        <div class="pol-fill" style="width:{pct}%;background:{color};"></div>
    </div>
</div>"""


def fetch_prediction():
    r = requests.get(API_URL, timeout=15)
    r.raise_for_status()
    return r.json()


def load_actual_history(hours=168):
    """Load recent historical AQI data from the local feature-store cache."""
    import sqlite3

    db_path = "data/feature_store.db"
    if not os.path.exists(db_path):
        return pd.DataFrame()

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            """SELECT timestamp, aqi FROM raw_readings
               WHERE city = 'karachi' AND aqi IS NOT NULL
               ORDER BY timestamp DESC LIMIT ?""",
            conn,
            params=(hours,),
        )

        if df.empty:
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["aqi"] = pd.to_numeric(df["aqi"], errors="coerce")

        return (
            df.dropna(subset=["timestamp", "aqi"])
              .sort_values("timestamp")
              .reset_index(drop=True)
        )

    except Exception:
        return pd.DataFrame()

    finally:
        if conn is not None:
            conn.close()


def load_72h_forecast(expected_pred_time=None, expected_predicted_aqi=None):
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

    # The first 72h forecast must match the live FastAPI
    # next-hour prediction. This prevents stale or mismatched
    # forecast files from being displayed.
    if expected_pred_time is not None:
        expected_time = pd.to_datetime(expected_pred_time, errors="coerce")
        if pd.isna(expected_time) or df.iloc[0]["timestamp"] != expected_time:
            return pd.DataFrame()

    if expected_predicted_aqi is not None:
        first_value = float(df.iloc[0]["predicted_aqi"])
        if abs(first_value - float(expected_predicted_aqi)) > 0.05:
            return pd.DataFrame()

    return df



# ============================================================
# MODEL / EXPLAINABILITY ARTIFACTS
# ============================================================

REPORTS_DIR = "reports"

def load_json_report(filename):
    path = os.path.join(REPORTS_DIR, filename)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def load_shap_importance(model_key):
    return load_json_report(f"shap_importance_{model_key}.json")

def shap_dataframe(model_key, top_n=12):
    payload = load_shap_importance(model_key)
    if not isinstance(payload, dict) or not payload:
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])
    rows = [{"feature": str(feature), "mean_abs_shap": float(value)} for feature, value in payload.items()]
    return (pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)
            .head(top_n).sort_values("mean_abs_shap", ascending=True).reset_index(drop=True))


# ============================================================
# NAV
# ============================================================

now_str = datetime.now(ZoneInfo("Asia/Karachi")).strftime("%d %b %Y, %H:%M PKT")

st.markdown(f"""
<div class="nav">
    <div class="nav-brand">
        <div class="nav-dot"></div>
        <span class="nav-name">Pearls AQI Predictor</span>
    </div>
    <div class="nav-right">
        <span class="nav-pill live">Prediction API Online</span>
        <span class="nav-pill">Karachi, PK</span>
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
        Air quality monitoring and forecasting powered by
        machine learning, engineered features, and a production
        Feature Store pipeline. The next-hour model drives a
        recursive 72-hour forecast.
    </div>
    <div class="hero-meta">
        <div class="hero-tag"><div class="hero-tag-dot"></div>Random Forest | R² 0.994 (next-hour model)</div>
        <div class="hero-tag"><div class="hero-tag-dot"></div>29 engineered features</div>
        <div class="hero-tag"><div class="hero-tag-dot"></div>FastAPI backend</div>
        <div class="hero-tag"><div class="hero-tag-dot"></div>Hopsworks Feature Store</div>
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
    with st.spinner("Connecting to prediction service..."):
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
model_obs_time = data.get("model_observation_timestamp")
model_obs_aqi = data.get("model_observation_aqi")

current_cat = get_category(current_aqi)
pred_cat = data.get("category", get_category(predicted_aqi))

if model_obs_aqi is not None:
    model_obs_aqi = float(model_obs_aqi)
    change = predicted_aqi - model_obs_aqi
else:
    change = predicted_aqi - current_aqi

if change > 0.05:
    trend, trend_sym, trend_cls = "Increasing", "↑", "stat-up"
elif change < -0.05:
    trend, trend_sym, trend_cls = "Decreasing", "↓", "stat-down"
else:
    trend, trend_sym, trend_cls = "Stable", "→", "stat-flat"

metrics = data.get("model_metrics", {})
mae  = metrics.get("mae")
rmse = metrics.get("rmse")
r2   = metrics.get("r2")
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

# Weather from API
weather = data.get("weather", {})
temperature = weather.get("temperature")
humidity    = weather.get("humidity")
pressure    = weather.get("pressure")
wind        = weather.get("wind")


# ============================================================
# ALERT BANNER
# ============================================================

icon, tip, alert_cls = HEALTH_TIPS.get(current_cat, ("◉", "", "warn"))
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

st.markdown('<div class="section-label">Current Conditions</div>', unsafe_allow_html=True)

cc, pc = aqi_style(current_cat), aqi_style(pred_cat)

left, right = st.columns(2, gap="medium")

with left:
    st.markdown(f"""
<div class="aqi-main-card {cc[0]}">
    <div class="card-eyebrow">Current AQI Estimate</div>
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
        <span class="card-city">{change:+.1f} AQI vs current</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# AQI SCALE
# ============================================================

_scale_colors = ["#22c55e", "#eab308", "#f97316", "#ef4444", "#8b5cf6", "#7f1d1d"]
_scale_labels = ["Good\n0–50", "Moderate\n51–100", "Sensitive\n101–150", "Unhealthy\n151–200", "V.Unhealthy\n201–300", "Hazardous\n301+"]
_scale_segs = "".join(f'<div class="aqi-seg" style="background:{c};"></div>' for c in _scale_colors)
_scale_lbls = "".join(f'<div class="aqi-scale-label">{l.replace(chr(10), "<br>")}</div>' for l in _scale_labels)

_cat_color = aqi_style(current_cat)[1]

st.markdown(f"""
<div class="aqi-scale-wrap">
    <div class="aqi-scale-title">AQI Reference Scale</div>
    <div class="aqi-scale-bar">{_scale_segs}</div>
    <div class="aqi-scale-labels">{_scale_lbls}</div>
    <div class="aqi-scale-pointer">
        <span class="aqi-pointer-dot" style="background:{_cat_color};"></span>
        Current AQI {current_aqi:.0f} falls in the <strong>&nbsp;{current_cat}&nbsp;</strong> range.
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# STAT STRIP
# ============================================================

st.markdown('<div class="section-label-sm">Forecast Analysis</div>', unsafe_allow_html=True)

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
    <div class="stat-label">Next-Hour Forecast</div>
    <div class="stat-value">{predicted_aqi:.1f}</div>
    <div class="stat-sub">{pred_cat}</div>
</div>
""", unsafe_allow_html=True)

with s3:
    st.markdown(f"""
<div class="stat-card">
    <div class="stat-label">AQI Change</div>
    <div class="stat-value {trend_cls}">{change:+.1f}</div>
    <div class="stat-sub">vs current AQI</div>
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
# WEATHER CONDITIONS
# ============================================================

def _weather_val(v, fmt=".1f", fallback="—"):
    if v is None:
        return fallback
    try:
        return f"{float(v):{fmt}}"
    except Exception:
        return fallback

st.markdown('<div class="section-label">Current Weather</div>', unsafe_allow_html=True)

w1, w2, w3, w4 = st.columns(4, gap="medium")

with w1:
    st.markdown(f"""
<div class="weather-card">
    <span class="weather-icon">🌡️</span>
    <div class="weather-label">Temperature</div>
    <div class="weather-value">{_weather_val(temperature)}</div>
    <div class="weather-unit">°C</div>
</div>
""", unsafe_allow_html=True)

with w2:
    st.markdown(f"""
<div class="weather-card">
    <span class="weather-icon">💧</span>
    <div class="weather-label">Humidity</div>
    <div class="weather-value">{_weather_val(humidity, ".0f")}</div>
    <div class="weather-unit">%</div>
</div>
""", unsafe_allow_html=True)

with w3:
    st.markdown(f"""
<div class="weather-card">
    <span class="weather-icon">🔵</span>
    <div class="weather-label">Pressure</div>
    <div class="weather-value">{_weather_val(pressure, ".0f")}</div>
    <div class="weather-unit">hPa</div>
</div>
""", unsafe_allow_html=True)

with w4:
    st.markdown(f"""
<div class="weather-card">
    <span class="weather-icon">💨</span>
    <div class="weather-label">Wind Speed</div>
    <div class="weather-value">{_weather_val(wind)}</div>
    <div class="weather-unit">km/h</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 72-HOUR FORECAST
# ============================================================

forecast_df = load_72h_forecast(pred_time, predicted_aqi)

if not forecast_df.empty:

    st.markdown(
        '<div class="section-label">72-Hour Forecast</div>',
        unsafe_allow_html=True
    )

    # ── +24h / +48h / +72h milestone cards ──────────────

    horizon_indices = [23, 47, 71]
    horizon_labels  = ["+24 Hours", "+48 Hours", "+72 Hours"]
    horizon_cols    = st.columns(3, gap="medium")

    for col, idx, label in zip(horizon_cols, horizon_indices, horizon_labels):
        if idx < len(forecast_df):
            value        = float(forecast_df.iloc[idx]["predicted_aqi"])
            category     = get_category(value)
            style        = aqi_style(category)
            forecast_time = forecast_df.iloc[idx]["timestamp"]

            with col:
                st.markdown(
                    f"""
                    <div class="aqi-main-card {style[0]}">
                        <div class="card-eyebrow">{label}</div>
                        <div class="card-aqi-number">{value:.1f}</div>
                        <div class="aqi-badge"
                             style="background:{style[2]};color:{style[3]};">
                            <div class="aqi-badge-dot"
                                 style="background:{style[1]};"></div>
                            {category}
                        </div>
                        <div style="margin-top:12px;font-size:12px;color:#64748b;font-weight:600;">
                            {forecast_time.strftime("%d %b %Y, %H:%M")}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # ── Forecast summary averages ────────────────────────

    st.markdown('<div class="section-label-sm" style="margin-top:24px;">Forecast Averages by Horizon</div>', unsafe_allow_html=True)

    horizon_ranges = [
        ("Next 24 Hours", 0, 24),
        ("24–48 Hours",   24, 48),
        ("48–72 Hours",   48, 72),
    ]

    avg_cols = st.columns(3, gap="medium")

    for col, (label, start_idx, end_idx) in zip(avg_cols, horizon_ranges):
        segment = forecast_df.iloc[start_idx:end_idx]
        if not segment.empty:
            avg = float(segment["predicted_aqi"].mean())
            avg_cat = get_category(avg)
            avg_style = aqi_style(avg_cat)
            with col:
                st.markdown(
                    f"""
                    <div class="stat-card">
                        <div class="stat-label">{label}</div>
                        <div class="stat-value">{avg:.1f}</div>
                        <div class="stat-sub" style="color:{avg_style[1]};font-weight:700;">{avg_cat}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # ── AQI History + Forecast chart ────────────────────

    st.markdown(
        '<div class="section-label" style="margin-top:28px;">AQI History + Forecast Timeline</div>',
        unsafe_allow_html=True
    )

    actual_df = load_actual_history(hours=168)

    if not actual_df.empty:
        actual_chart = actual_df.rename(columns={"aqi": "Historical AQI"})
        forecast_chart = forecast_df[["timestamp", "predicted_aqi"]].copy()
        forecast_chart = forecast_chart.rename(columns={"predicted_aqi": "Forecast AQI"})

        timeline = pd.merge(
            actual_chart[["timestamp", "Historical AQI"]],
            forecast_chart[["timestamp", "Forecast AQI"]],
            on="timestamp",
            how="outer",
        ).sort_values("timestamp")

        timeline_long = timeline.melt(
            id_vars=["timestamp"],
            value_vars=["Historical AQI", "Forecast AQI"],
            var_name="Series",
            value_name="AQI",
        ).dropna(subset=["AQI"])

        chart = (
            alt.Chart(timeline_long)
            .mark_line(strokeWidth=2.5, point=False)
            .encode(
                x=alt.X(
                    "timestamp:T",
                    title="Date / Time",
                    axis=alt.Axis(
                        format="%d %b %H:%M",
                        tickCount=10,
                        labelAngle=-30,
                        labelFontSize=11,
                    ),
                ),
                y=alt.Y(
                    "AQI:Q",
                    title="AQI",
                    scale=alt.Scale(zero=False),
                ),
                color=alt.Color(
                    "Series:N",
                    title=None,
                    scale=alt.Scale(
                        domain=["Historical AQI", "Forecast AQI"],
                        range=["#3b82f6", "#22c55e"],
                    ),
                    legend=alt.Legend(
                        orient="top-right",
                        labelFontSize=12,
                        labelFontWeight=600,
                    ),
                ),
                strokeDash=alt.condition(
                    alt.datum.Series == "Forecast AQI",
                    alt.value([6, 3]),
                    alt.value([0]),
                ),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Time", format="%d %b %Y, %H:%M"),
                    alt.Tooltip("Series:N", title="Series"),
                    alt.Tooltip("AQI:Q", title="AQI", format=".1f"),
                ],
            )
            .properties(height=380)
            .interactive()
        )

        st.altair_chart(chart, use_container_width=True)

        st.caption(
            "Historical AQI shows the latest 7 days available in the local historical data cache. "
            "Forecast AQI (dashed) begins after the latest historical point and shows the model's "
            "72-hour recursive forecast. Future pollutant and weather inputs are carried forward "
            "from the latest known data point — this is a model forecast, not future measured data."
        )

    else:
        forecast_only = (
            alt.Chart(forecast_df)
            .mark_line(strokeWidth=2.5, color="#22c55e", strokeDash=[6, 3])
            .encode(
                x=alt.X(
                    "timestamp:T",
                    title="Date / Time",
                    axis=alt.Axis(format="%d %b %H:%M", tickCount=8, labelAngle=-30),
                ),
                y=alt.Y("predicted_aqi:Q", title="AQI", scale=alt.Scale(zero=False)),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Time", format="%d %b %Y, %H:%M"),
                    alt.Tooltip("predicted_aqi:Q", title="Forecast AQI", format=".1f"),
                ],
            )
            .properties(height=380)
            .interactive()
        )
        st.altair_chart(forecast_only, use_container_width=True)
        st.caption(
            "Historical AQI data was unavailable. Only the 72-hour model forecast is shown."
        )

    # ── Expandable forecast detail table ────────────────

    with st.expander("View full 72-hour forecast table"):
        table_rows = ""
        for _, row in forecast_df.iterrows():
            aqi_val = float(row["predicted_aqi"])
            cat     = get_category(aqi_val)
            style   = aqi_style(cat)
            ts      = row["timestamp"]
            try:
                ts_str = ts.strftime("%d %b %Y, %H:%M")
            except Exception:
                ts_str = str(ts)
            table_rows += f"""
            <tr>
                <td>{ts_str}</td>
                <td style="font-weight:800;">{aqi_val:.1f}</td>
                <td><span class="aqi-badge" style="background:{style[2]};color:{style[3]};font-size:11px;padding:3px 10px 3px 7px;">
                    <span class="aqi-badge-dot" style="background:{style[1]};width:6px;height:6px;"></span>
                    {cat}
                </span></td>
            </tr>"""

        st.markdown(f"""
<div style="background:white;border-radius:14px;border:1px solid rgba(26,31,54,0.06);overflow:hidden;margin-top:4px;">
    <table class="forecast-table">
        <thead>
            <tr>
                <th>Forecast Time</th>
                <th>Predicted AQI</th>
                <th>Category</th>
            </tr>
        </thead>
        <tbody>{table_rows}</tbody>
    </table>
</div>
""", unsafe_allow_html=True)

    first_forecast = float(forecast_df.iloc[0]["predicted_aqi"])
    last_forecast  = float(forecast_df.iloc[-1]["predicted_aqi"])
    start_time     = forecast_df.iloc[0]["timestamp"]
    end_time       = forecast_df.iloc[-1]["timestamp"]

    st.caption(
        f"Forecast period: {start_time.strftime('%d %b %Y, %H:%M')} "
        f"– {end_time.strftime('%d %b %Y, %H:%M')}. "
        f"First hour: {first_forecast:.1f} AQI · Final hour: {last_forecast:.1f} AQI."
    )

else:
    st.info(
        "A valid 72-hour forecast was not found. "
        "Run generate_72h_forecast.py to create forecast_72h.csv."
    )


# ============================================================
# HISTORICAL ANALYSIS
# ============================================================

actual_history = load_actual_history(hours=168)

if not actual_history.empty:
    st.markdown('<div class="section-label">Historical AQI Analysis</div>', unsafe_allow_html=True)

    hist_min  = float(actual_history["aqi"].min())
    hist_max  = float(actual_history["aqi"].max())
    hist_avg  = float(actual_history["aqi"].mean())
    hist_last = float(actual_history["aqi"].iloc[-1])

    h1, h2, h3, h4 = st.columns(4, gap="medium")
    for col, label, val, sub in [
        (h1, "7-Day Minimum", hist_min, get_category(hist_min)),
        (h2, "7-Day Maximum", hist_max, get_category(hist_max)),
        (h3, "7-Day Average", hist_avg, get_category(hist_avg)),
        (h4, "Latest Historical", hist_last, get_category(hist_last)),
    ]:
        st_col = aqi_style(sub)
        with col:
            st.markdown(f"""
<div class="stat-card">
    <div class="stat-label">{label}</div>
    <div class="stat-value">{val:.1f}</div>
    <div class="stat-sub" style="color:{st_col[1]};font-weight:700;">{sub}</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# OBSERVATION DETAILS + POLLUTANTS
# ============================================================

st.markdown('<div class="section-label">Observation Details</div>', unsafe_allow_html=True)

dl, dr = st.columns(2, gap="medium")

with dl:
    st.markdown(f"""
<div class="detail-card">
    <div class="detail-card-header">Current Data</div>
    <div class="detail-row">
        <span class="detail-key">Location</span>
        <span class="detail-val">Karachi, Pakistan</span>
    </div>
    <div class="detail-row">
        <span class="detail-key">Data source</span>
        <span class="detail-val">Open-Meteo AQI estimate</span>
    </div>
    <div class="detail-row">
        <span class="detail-key">Current data time</span>
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
        <span class="detail-key">Next-hour forecast AQI</span>
        <span class="detail-val">{predicted_aqi:.2f}</span>
    </div>
    <div class="detail-row">
        <span class="detail-key">Forecast time</span>
        <span class="detail-val">{fmt_time(pred_time)}</span>
    </div>
    <div class="detail-row">
        <span class="detail-key">Expected change</span>
        <span class="detail-val">{change:+.1f} AQI points</span>
    </div>
    <div class="detail-row">
        <span class="detail-key">Horizon</span>
        <span class="detail-val">1 hour (next-hour model)</span>
    </div>
</div>
""", unsafe_allow_html=True)

with dr:
    pm25_bar = pol_bar("PM 2.5",  pm25, 250, "#ef4444")
    pm10_bar = pol_bar("PM 10",   pm10, 350, "#f97316")
    o3_bar   = pol_bar("O₃",       o3,   200, "#8b5cf6")
    no2_bar  = pol_bar("NO₂",      no2,  200, "#3b82f6")
    so2_bar  = pol_bar("SO₂",      so2,  150, "#eab308")
    co_bar   = pol_bar("CO",       co,   500, "#6b7280", unit="µg/m³")

    st.markdown(f"""
<div class="pol-card">
    <div class="pol-card-title">Pollutant Concentrations</div>
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
    "hour", "day", "month", "day_of_week", "is_weekend",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_6h", "aqi_lag_12h", "aqi_lag_24h",
    "aqi_roll_mean_3h", "aqi_roll_mean_6h", "aqi_roll_mean_12h", "aqi_roll_mean_24h",
    "aqi_roll_std_6h", "aqi_roll_std_24h", "aqi_change_1h", "aqi_change_6h", "aqi_change_24h",
    "pm25", "pm10", "o3", "no2", "so2", "co", "temperature", "humidity", "pressure", "wind",
]
tags = "".join(f'<span class="feature-tag">{f}</span>' for f in features_list)

st.markdown(f"""
<div class="model-card">
    <div class="model-name">{model_name}</div>
    <div class="model-badge">Selected production model</div>
    <div class="model-desc">
        Hopsworks Feature Store &nbsp;|&nbsp; Random Forest Regression<br>
        Chronological train/test splitting prevents data leakage. The model predicts
        one hour ahead from the latest available data point; the FastAPI service
        uses a local SQLite feature-store cache for fast, reliable inference.
    </div>
    <div class="model-metrics">
        <div class="model-metric">
            <div class="model-metric-label">MAE</div>
            <div class="model-metric-value">{mae_str}</div>
            <div class="model-metric-note">next-hour model</div>
        </div>
        <div class="model-metric">
            <div class="model-metric-label">RMSE</div>
            <div class="model-metric-value">{rmse_str}</div>
            <div class="model-metric-note">next-hour model</div>
        </div>
        <div class="model-metric">
            <div class="model-metric-label">R²</div>
            <div class="model-metric-value">{r2_str}</div>
            <div class="model-metric-note">next-hour model</div>
        </div>
    </div>
    <div class="model-metrics-disclaimer">
        These metrics reflect next-hour prediction accuracy, not 72-hour forecast accuracy.
        The 72-hour forecast is generated recursively and accumulates error over time.
    </div>
    <div class="feature-tags">{tags}</div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# MODEL EVALUATION + EXPLAINABILITY
# ============================================================

st.markdown('<div class="section-label">Model Evaluation & Explainability</div>', unsafe_allow_html=True)

training_report = load_json_report("training_metrics.json")
horizon_report = load_json_report("horizon_metrics.json")

if isinstance(training_report, dict):
    tm1, tm2, tm3 = st.columns(3, gap="medium")
    for col, label, key, note in [
        (tm1, "Next-Hour MAE", "mae", "production Random Forest"),
        (tm2, "Next-Hour RMSE", "rmse", "production Random Forest"),
        (tm3, "Next-Hour R²", "r2", "production Random Forest"),
    ]:
        value = training_report.get(key)
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">{label}</div>
                <div class="stat-value">{float(value):.4f}</div>
                <div class="stat-sub">{note}</div>
            </div>
            """, unsafe_allow_html=True)
    st.caption("These metrics are loaded from reports/training_metrics.json and describe the saved next-hour production model. They are not 72-hour recursive forecast metrics.")

if isinstance(horizon_report, dict) and isinstance(horizon_report.get("horizons"), dict):
    rows = []
    for horizon_key, item in horizon_report["horizons"].items():
        if not isinstance(item, dict):
            continue
        rows.append({
            "Horizon": f"+{int(item.get('horizon_hours', horizon_key.rstrip('h')))}h",
            "Model": item.get("model", "—"),
            "MAE": float(item["mae"]),
            "RMSE": float(item["rmse"]),
            "R²": float(item["r2"]),
            "Test samples": int(item["test_samples"]),
        })
    if rows:
        eval_df = pd.DataFrame(rows).sort_values("Horizon")
        st.markdown('<div class="section-label-sm" style="margin-top:24px;">Horizon-Specific Supervised Validation</div>', unsafe_allow_html=True)
        st.dataframe(eval_df, use_container_width=True, hide_index=True, column_config={
            "MAE": st.column_config.NumberColumn(format="%.4f"),
            "RMSE": st.column_config.NumberColumn(format="%.4f"),
            "R²": st.column_config.NumberColumn(format="%.4f"),
        })
        st.caption("Separate Random Forest models were evaluated with targets shifted by 24, 48, and 72 hours using chronological train/test splits. These supervised horizon results should not be confused with the recursive production forecast.")

st.markdown('<div class="section-label-sm" style="margin-top:24px;">SHAP Feature Importance Comparison</div>', unsafe_allow_html=True)
st.caption("The following explainability artifacts are loaded from the saved SHAP JSON files. Mean |SHAP| measures the average magnitude of each feature's contribution; it does not indicate whether the feature raises or lowers AQI.")

shap_tabs = st.tabs(["Random Forest", "Ridge Regression", "XGBoost"])
for tab, model_key, title in zip(shap_tabs, ["random_forest", "ridge_regression", "xgboost"], ["Random Forest", "Ridge Regression", "XGBoost"]):
    with tab:
        shap_df = shap_dataframe(model_key, top_n=12)
        if shap_df.empty:
            st.info(f"No saved SHAP JSON found for {title}.")
        else:
            shap_chart = (alt.Chart(shap_df).mark_bar().encode(
                x=alt.X("mean_abs_shap:Q", title="Mean |SHAP value|", axis=alt.Axis(format=".2f")),
                y=alt.Y("feature:N", sort=None, title=None),
                tooltip=[alt.Tooltip("feature:N", title="Feature"), alt.Tooltip("mean_abs_shap:Q", title="Mean |SHAP|", format=".4f")],
            ).properties(height=360))
            st.altair_chart(shap_chart, use_container_width=True)

st.markdown('<div class="model-metrics-disclaimer" style="margin-top:16px;"><strong>Interpretation note:</strong> The saved training report provides production Random Forest performance metrics, while the Ridge Regression and XGBoost artifacts available here are SHAP explainability outputs. Their SHAP rankings are shown for interpretation only; no unsupported performance ranking is claimed.</div>', unsafe_allow_html=True)


# ============================================================
# FORECAST METHODOLOGY
# ============================================================

st.markdown('<div class="section-label">Forecast Methodology</div>', unsafe_allow_html=True)

st.markdown("""
<div class="method-card">
    <div class="method-step">
        <div class="method-num">1</div>
        <div class="method-text"><strong>Data ingestion.</strong> Open-Meteo provides current AQI, pollutant concentrations (PM2.5, PM10, O₃, NO₂, SO₂, CO), and meteorological variables (temperature, humidity, pressure, wind). Data is cached in a local SQLite feature-store for fast inference.</div>
    </div>
    <div class="method-step">
        <div class="method-num">2</div>
        <div class="method-text"><strong>Feature engineering.</strong> 29 features are constructed from the raw inputs: lag features (1h, 3h, 6h, 12h, 24h), rolling statistics (mean and std over 3h, 6h, 12h, 24h windows), rate-of-change features, and cyclical time encodings.</div>
    </div>
    <div class="method-step">
        <div class="method-num">3</div>
        <div class="method-text"><strong>Feature Store.</strong> Features are managed through Hopsworks Feature Store, enabling reproducible training, versioning, and a clean boundary between feature preparation and model inference.</div>
    </div>
    <div class="method-step">
        <div class="method-num">4</div>
        <div class="method-text"><strong>Next-hour prediction.</strong> The Random Forest Regressor (trained with chronological splitting) takes the current feature row and produces a next-hour AQI estimate via the FastAPI prediction endpoint.</div>
    </div>
    <div class="method-step">
        <div class="method-num">5</div>
        <div class="method-text"><strong>72-hour recursive forecast.</strong> Starting from the latest available data point, the model predicts one hour ahead, carries that prediction forward as the next input, and repeats for 72 steps. Future pollutant and weather inputs are held at the latest known values. This is a model forecast — not future measured data.</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SYSTEM ARCHITECTURE
# ============================================================

st.markdown('<div class="section-label">System Architecture</div>', unsafe_allow_html=True)

st.markdown("""
<div class="arch-card">
    <div style="font-size:13px;font-weight:700;color:#1a1f36;margin-bottom:4px;">End-to-End Pipeline</div>
    <div style="font-size:12px;color:#64748b;margin-bottom:16px;">How data flows from source to dashboard</div>
    <div class="arch-flow">
        <div class="arch-node">Open-Meteo API</div>
        <div class="arch-arrow">→</div>
        <div class="arch-node">Feature Engineering<br><span style="font-size:10px;font-weight:500;color:#94a3b8;">29 features</span></div>
        <div class="arch-arrow">→</div>
        <div class="arch-node">Hopsworks Feature Store<br><span style="font-size:10px;font-weight:500;color:#94a3b8;">+ SQLite cache</span></div>
        <div class="arch-arrow">→</div>
        <div class="arch-node highlight">Random Forest Model<br><span style="font-size:10px;font-weight:500;color:#15803d;">next-hour</span></div>
        <div class="arch-arrow">→</div>
        <div class="arch-node">FastAPI<br><span style="font-size:10px;font-weight:500;color:#94a3b8;">:8000/predict</span></div>
        <div class="arch-arrow">→</div>
        <div class="arch-node">72h Recursive Forecast<br><span style="font-size:10px;font-weight:500;color:#94a3b8;">forecast_72h.csv</span></div>
        <div class="arch-arrow">→</div>
        <div class="arch-node">Streamlit Dashboard<br><span style="font-size:10px;font-weight:500;color:#94a3b8;">this page</span></div>
    </div>
    <div style="margin-top:16px;font-size:12px;color:#94a3b8;line-height:1.6;">
        The Streamlit dashboard does not perform inference directly. All predictions come from the FastAPI service,
        which reads the latest feature row from the SQLite cache and calls the trained Random Forest model.
        GitHub Actions automates periodic data ingestion, feature-store updates, and daily model training/registry updates. Forecast regeneration is performed separately by generate_72h_forecast.py.
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# DATA TRANSPARENCY
# ============================================================

st.markdown('<div class="section-label">Data Sources</div>', unsafe_allow_html=True)

st.markdown("""
<div class="source-card">
    <div class="source-item">
        <div class="source-dot" style="background:#3b82f6;"></div>
        <div>
            <div class="source-title">Open-Meteo Air Quality API</div>
            <div class="source-desc">Provides current and historical AQI estimates, pollutant concentrations, and meteorological variables for Karachi. Note: this is a modelled AQI estimate, not a ground monitoring station measurement.</div>
        </div>
    </div>
    <div class="source-item">
        <div class="source-dot" style="background:#8b5cf6;"></div>
        <div>
            <div class="source-title">Hopsworks Feature Store</div>
            <div class="source-desc">Manages training features and enables versioned, reproducible ML pipelines. Used during model training and for feature group management.</div>
        </div>
    </div>
    <div class="source-item">
        <div class="source-dot" style="background:#22c55e;"></div>
        <div>
            <div class="source-title">SQLite Feature Cache (data/feature_store.db)</div>
            <div class="source-desc">Local cache of the most recent feature rows. Used by the FastAPI service for low-latency inference and by the dashboard for historical AQI visualisation.</div>
        </div>
    </div>
    <div class="source-item">
        <div class="source-dot" style="background:#f97316;"></div>
        <div>
            <div class="source-title">72-Hour Forecast File (data/forecast_72h.csv)</div>
            <div class="source-desc">Generated by generate_72h_forecast.py using recursive model inference. Generated separately by generate_72h_forecast.py. The dashboard validates freshness by checking alignment with the live API next-hour prediction.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# REFRESH + CONTROLS
# ============================================================

st.markdown('<div class="section-label">Controls</div>', unsafe_allow_html=True)

col_btn, col_pad = st.columns([1, 3])
with col_btn:
    if st.button("Refresh prediction"):
        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown(f"""
<div class="site-footer">
    <p>
        <strong>Pearls AQI Predictor</strong> — Karachi Air Quality Forecasting<br>
        Hopsworks Feature Store &nbsp;|&nbsp; Random Forest Regression &nbsp;·&nbsp; FastAPI &nbsp;·&nbsp; Streamlit<br>
        AQI data: Open-Meteo (modelled estimate) &nbsp;·&nbsp; Last refreshed: {now_str}
    </p>
</div>
""", unsafe_allow_html=True)
