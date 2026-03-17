"""
Price Predictor Dashboard — LSTM
Instantly predicts future stock prices based on historical trends.
"""

import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Price Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #0a0a0c;
    --accent-primary: #8b5cf6;
    --text-main: #e2e8f0;
    --text-muted: #94a3b8;
}

html, body, [class*="css"] { 
    font-family: 'Inter', sans-serif; 
    background-color: var(--bg-primary);
    color: var(--text-main);
}

.stApp {
    background-color: var(--bg-primary);
}

/* Metric Styling */
[data-testid="metric-container"] {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 20px;
}

/* Header Sections */
.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    color: white;
    margin-bottom: 4px;
}

.hero-sub { 
    color: var(--text-muted); 
    font-size: 1rem; 
    margin-bottom: 32px;
}

.section-hdr {
    font-size: 1rem;
    font-weight: 600;
    color: var(--accent-primary);
    margin: 24px 0 16px;
}

/* Button */
div.stButton > button {
    background: var(--accent-primary);
    color: white;
    border-radius: 8px;
    padding: 12px;
    width: 100%;
    font-weight: 600;
}

hr { border-color: rgba(255, 255, 255, 0.05); }

/* Hide Streamlit components */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ── Load model & data ────────────────────────────────────────────────────────
MODEL_PATH  = "aapl_lstm_model.keras"
SCALER_PATH = "aapl_scaler.pkl"
DATA_PATH   = "AAPL_full.csv"
TIME_STEP   = 100

@st.cache_resource
def load_model_and_scaler():
    import joblib
    from tensorflow.keras.models import load_model
    model  = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

@st.cache_data
def load_historical_data():
    df = pd.read_csv(DATA_PATH, skiprows=3, header=None,
                     names=["Date", "Close"], index_col=0, parse_dates=True)
    df = df.dropna()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna()
    return df

files_ready = os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(DATA_PATH)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Settings")
    st.markdown("---")

    forecast_from = st.date_input(
        "Projection Start Date",
        value=date.today() - timedelta(days=1),
    )

    forecast_days = st.slider(
        "Days to Forecast",
        min_value=1, max_value=10, value=5,
    )

    context_days = st.slider(
        "History to View",
        min_value=30, max_value=365, value=120,
    )

    st.markdown("---")
    predict_btn = st.button("Generate Forecast")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Predictive Terminal</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Simple and accurate stock price forecasting.</div>', unsafe_allow_html=True)

if not files_ready:
    st.error("Model or data files not found.")
    st.stop()

# ── Load Data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading..."):
    model, scaler = load_model_and_scaler()
    df = load_historical_data()

# ── Key Metrics ───────────────────────────────────────────────────────────────
current_price = float(df["Close"].iloc[-1])
prev_price    = float(df["Close"].iloc[-2])
change_pct    = ((current_price - prev_price) / prev_price) * 100

m1, m2, m3 = st.columns(3)
m1.metric("Recent Close", f"${current_price:.2f}", f"{change_pct:+.2f}%")
m2.metric("Latest Update", str(df.index[-1].date()))
m3.metric("System Status", "Live")

st.markdown("---")

# ── Forecast Logic ────────────────────────────────────────────────────────────
if not predict_btn:
    st.info("👈 Adjust settings in the sidebar and click 'Generate Forecast'.")
    st.stop()

# Prepare seed data
from_ts = pd.Timestamp(forecast_from)
if from_ts > df.index[-1]:
    from_ts = df.index[-1]

seed_data = df[df.index <= from_ts].tail(TIME_STEP)

if len(seed_data) < TIME_STEP:
    st.error(f"Insufficient historical data (need {TIME_STEP} days).")
    st.stop()

with st.spinner("Calculating..."):
    seed_scaled = scaler.transform(seed_data[["Close"]].values)
    inputs = list(seed_scaled.flatten())

    preds = []
    for _ in range(forecast_days):
        arr = np.array(inputs[-TIME_STEP:]).reshape(1, TIME_STEP, 1)
        p = model.predict(arr, verbose=0)[0][0]
        preds.append(p)
        inputs.append(p)

    preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

future_dates = pd.bdate_range(start=from_ts + timedelta(days=1), periods=forecast_days)

# ── Result Metrics ────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">Forecast Results</div>', unsafe_allow_html=True)

start_price = float(seed_data["Close"].iloc[-1])
target_price = float(preds_inv[-1])
trend_pct = ((target_price - start_price) / start_price) * 100

r1, r2, r3 = st.columns(3)
r1.metric("Start Price", f"${start_price:.2f}")
r2.metric("Target Price", f"${target_price:.2f}", f"{trend_pct:+.2f}%")
r3.metric("Peak Forecast", f"${preds_inv.max():.2f}")

# ── Visualization ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-hdr">Visual Forecast</div>', unsafe_allow_html=True)

context_data = df.tail(context_days)

fig = go.Figure()

# Historical Price
fig.add_trace(go.Scatter(
    x=context_data.index, y=context_data["Close"],
    name="History",
    line=dict(color="#f8fafc", width=2),
))

# Forecast Range (Subtle)
fig.add_trace(go.Scatter(
    x=list(future_dates) + list(future_dates[::-1]),
    y=list(preds_inv * 1.02) + list(preds_inv[::-1] * 0.98),
    fill="toself",
    fillcolor="rgba(139, 92, 246, 0.1)",
    line=dict(color="rgba(0,0,0,0)"),
    name="Range",
))

# Forecast Line
fig.add_trace(go.Scatter(
    x=future_dates, y=preds_inv,
    name="Forecast",
    line=dict(color="#8b5cf6", width=3),
))

fig.update_layout(
    height=450,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    font=dict(color="#94a3b8"),
    hovermode="x unified",
    margin=dict(l=0, r=0, t=10, b=0),
    legend=dict(orientation="h", xanchor="right", x=1, yanchor="bottom", y=1.02),
    xaxis=dict(showgrid=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Price ($)"),
)

st.plotly_chart(fig, use_container_width=True)

# ── Details ───────────────────────────────────────────────────────────────────
with st.expander("Show Daily Projections"):
    details_df = pd.DataFrame({
        "Date": future_dates.strftime('%Y-%m-%d'),
        "Price": [f"${p:.2f}" for p in preds_inv],
    })
    st.dataframe(details_df.set_index("Date"), use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#475569;font-size:0.8rem;'>"
    "Neural Prediction Model • Version 2.5"
    "</div>",
    unsafe_allow_html=True,
)
