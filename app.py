"""
AAPL Stock Price Dashboard — Pre-trained LSTM
Loads aapl_lstm_model.keras + aapl_scaler.pkl instantly (no live training).
User picks a 'forecast from' date and a horizon, then sees the prediction.
"""

import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AAPL Stock Predictor",
    page_icon="🍎",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e0e0e0;
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255,255,255,0.1);
}
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 16px 20px;
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub { color: #94a3b8; font-size: 1.05rem; margin-bottom: 24px; }
.section-hdr {
    font-size: 1.15rem; font-weight: 600; color: #a78bfa;
    border-bottom: 1px solid rgba(167,139,250,0.3);
    padding-bottom: 6px; margin: 20px 0 14px;
}
div.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #2563eb);
    color: white; border: none; border-radius: 10px;
    padding: 10px 24px; font-size: 1rem; font-weight: 600;
    width: 100%; transition: all 0.3s;
}
div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(124,58,237,0.4);
}
hr { border-color: rgba(255,255,255,0.08); }
</style>
""", unsafe_allow_html=True)

# ── Load pre-trained model & scaler ──────────────────────────────────────────
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
    # yfinance CSVs have 3 metadata header rows: Price, Ticker, Date
    # skiprows=3 jumps past them; we supply our own column names
    df = pd.read_csv(DATA_PATH, skiprows=3, header=None,
                     names=["Date", "Close"], index_col=0, parse_dates=True)
    df = df.dropna()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna()
    return df

# Check files exist before loading
model_ready = os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(DATA_PATH)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🍎 AAPL Predictor")
    st.markdown("Pre-trained on **AAPL** data from 2010 to today.\nNo retraining needed — adjust and predict instantly.")
    st.markdown("---")

    # Date from which the forecast starts
    forecast_from = st.date_input(
        "📅 Forecast From",
        value=date.today() - timedelta(days=1),
        help="The model will use the 100 trading days leading up to this date as context, then forecast forward.",
    )

    # How many days to predict
    forecast_days = st.slider(
        "🔭 Forecast Horizon (Days)",
        min_value=7, max_value=90, value=30, step=7,
        help="Number of business days to predict into the future.",
    )

    # How many historical days to show on context chart
    context_days = st.slider(
        "📊 Historical Context (Days)",
        min_value=30, max_value=365, value=120, step=30,
        help="How many past days to display alongside the forecast.",
    )

    st.markdown("---")
    predict_btn = st.button("🔮 Generate Forecast")

    if not model_ready:
        st.error("⚠️ Model not found! Run `train_and_save.py` first.")

    st.markdown("---")
    st.markdown("""
    <div style='color:#64748b;font-size:0.82rem;line-height:1.7'>
    <b>How it works</b><br>
    1. Loads the pre-trained LSTM instantly<br>
    2. Seeds the model with the last 100 days before your chosen date<br>
    3. Rolls forward day-by-day for the chosen horizon<br>
    4. Shows the forecast on an interactive chart
    </div>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🍎 AAPL Stock Price Forecast</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Instant predictions using a pre-trained Stacked LSTM — no waiting, no retraining.</div>', unsafe_allow_html=True)

if not model_ready:
    st.warning("**Model files not found.** Please run this command once in your terminal to train and save the model:")
    st.code("E:\\anaconda3\\python.exe train_and_save.py", language="bash")
    st.info("Training takes ~5–10 minutes. After that, this dashboard will load instantly every time.")
    st.stop()

# ── Load everything ───────────────────────────────────────────────────────────
with st.spinner("Loading pre-trained model…"):
    model, scaler = load_model_and_scaler()
    df = load_historical_data()

# Summary metrics (always visible)
current_price = float(df["Close"].iloc[-1])
last_date     = df.index[-1].date()
total_days    = len(df)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Current AAPL Price", f"${current_price:.2f}")
m2.metric("Last Data Point",    str(last_date))
m3.metric("Training History",   f"{total_days:,} days")
m4.metric("Model",              "Stacked LSTM (3×50)")

st.markdown("---")

# ── Forecast on button click ──────────────────────────────────────────────────
if not predict_btn:
    st.markdown("""
    <div style='background:rgba(96,165,250,0.1);border:1px solid rgba(96,165,250,0.3);
    border-radius:10px;padding:14px 18px;color:#93c5fd;font-size:0.95rem;'>
    👈 &nbsp;Adjust the <b>forecast date</b> and <b>horizon</b> in the sidebar, then click
    <b>Generate Forecast</b> to see the prediction.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# --- Validate date ---
forecast_from_ts = pd.Timestamp(forecast_from)
if forecast_from_ts > df.index[-1]:
    # Use the last available date
    forecast_from_ts = df.index[-1]
    st.warning(f"Forecast date is in the future — using last available data point ({last_date}) as seed.")

# Grab the 100 days BEFORE the forecast date as seed
seed_window = df[df.index <= forecast_from_ts].tail(TIME_STEP)

if len(seed_window) < TIME_STEP:
    st.error(f"Not enough data before {forecast_from} — need at least {TIME_STEP} trading days. Please pick a later date.")
    st.stop()

# --- Roll-forward forecast ---
with st.spinner(f"Generating {forecast_days}-day forecast from {forecast_from}…"):
    seed_scaled = scaler.transform(seed_window.values)
    input_seq   = list(seed_scaled.flatten())

    future_preds = []
    for _ in range(forecast_days):
        arr  = np.array(input_seq[-TIME_STEP:]).reshape(1, TIME_STEP, 1)
        pred = model.predict(arr, verbose=0)[0][0]
        future_preds.append(pred)
        input_seq.append(pred)

    future_preds_inv = scaler.inverse_transform(
        np.array(future_preds).reshape(-1, 1)
    ).flatten()

# Future date index (business days only)
future_dates = pd.bdate_range(start=forecast_from_ts + timedelta(days=1), periods=forecast_days)

# --- Key numbers ---
seed_price     = float(seed_window["Close"].iloc[-1])
end_price      = float(future_preds_inv[-1])
delta_usd      = end_price - seed_price
delta_pct      = (delta_usd / seed_price) * 100
peak_price     = float(future_preds_inv.max())
trough_price   = float(future_preds_inv.min())

st.markdown('<div class="section-hdr">📊 Forecast Summary</div>', unsafe_allow_html=True)
fc1, fc2, fc3, fc4 = st.columns(4)
fc1.metric("Price at Forecast Start", f"${seed_price:.2f}")
fc2.metric(f"Price in {forecast_days}d", f"${end_price:.2f}", f"{delta_pct:+.2f}%")
fc3.metric("Forecast Peak",   f"${peak_price:.2f}")
fc4.metric("Forecast Trough", f"${trough_price:.2f}")

# ── Chart 1: Historical context + forecast ────────────────────────────────────
st.markdown('<div class="section-hdr">📈 Historical Price + Forecast</div>', unsafe_allow_html=True)

context = df.tail(context_days)

fig = go.Figure()

# Historical
fig.add_trace(go.Scatter(
    x=context.index, y=context["Close"],
    name="Historical Price",
    line=dict(color="#60a5fa", width=1.8),
))

# Connector dot
fig.add_trace(go.Scatter(
    x=[seed_window.index[-1], future_dates[0]],
    y=[seed_price, future_preds_inv[0]],
    mode="lines", showlegend=False,
    line=dict(color="#a78bfa", dash="dot", width=1.5),
))

# Forecast band (fill)
fig.add_trace(go.Scatter(
    x=list(future_dates) + list(future_dates[::-1]),
    y=list(future_preds_inv * 1.02) + list(future_preds_inv[::-1] * 0.98),
    fill="toself",
    fillcolor="rgba(167,139,250,0.08)",
    line=dict(color="rgba(0,0,0,0)"),
    name="Confidence Band",
    showlegend=True,
))

# Forecast line
fig.add_trace(go.Scatter(
    x=future_dates, y=future_preds_inv,
    name=f"{forecast_days}-Day Forecast",
    line=dict(color="#a78bfa", width=2.5),
    mode="lines+markers",
    marker=dict(size=4),
))

# Vertical line at forecast start
fig.add_vline(
    x=forecast_from_ts.timestamp() * 1000,
    line_dash="dash", line_color="rgba(255,255,255,0.3)",
    annotation_text="Forecast Start",
    annotation_font_color="#94a3b8",
)

fig.update_layout(
    height=440,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,12,41,0.6)",
    font=dict(color="#94a3b8"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Date"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Price (USD)"),
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

# ── Chart 2: Forecast-only zoomed in ─────────────────────────────────────────
st.markdown('<div class="section-hdr">🔭 Forecast Detail</div>', unsafe_allow_html=True)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=future_dates, y=future_preds_inv,
    name="Predicted Price",
    fill="tozeroy",
    fillcolor="rgba(167,139,250,0.1)",
    line=dict(color="#a78bfa", width=2.5),
    mode="lines+markers",
    marker=dict(size=5, color="#a78bfa"),
))

# Zero-line at current price
fig2.add_hline(
    y=seed_price,
    line_dash="dash", line_color="rgba(96,165,250,0.5)",
    annotation_text=f"Seed Price ${seed_price:.2f}",
    annotation_font_color="#60a5fa",
)

fig2.update_layout(
    height=320,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(15,12,41,0.6)",
    font=dict(color="#94a3b8"),
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Date"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Predicted Price (USD)"),
    hovermode="x unified",
)
st.plotly_chart(fig2, use_container_width=True)

# ── Raw forecast table ────────────────────────────────────────────────────────
with st.expander("📋 Full Forecast Table", expanded=False):
    forecast_df = pd.DataFrame({
        "Date":            future_dates,
        "Predicted Price": [f"${p:.2f}" for p in future_preds_inv],
        "Change vs Start": [f"{((p - seed_price)/seed_price)*100:+.2f}%" for p in future_preds_inv],
    })
    st.dataframe(forecast_df.set_index("Date"), use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#475569;font-size:0.82rem'>"
    "Built with Streamlit · TensorFlow · Plotly &nbsp;|&nbsp; "
    "For educational purposes only — not financial advice."
    "</div>",
    unsafe_allow_html=True,
)
