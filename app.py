"""
app.py  —  Stock Price Prediction Dashboard
- Live price feed via Twelve Data (refreshes every 30s during market hours)
- Auto-retrains if model is stale (>24h old) AND market is closed
- Manual "Retrain Now" button in sidebar
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import time
import os
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from sklearn.metrics import mean_squared_error, r2_score
from train_model import StockPredictionModel, build_features, fetch_historical, API_KEY

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Predictor", layout="wide", page_icon="📈")

MODEL_PATH   = "aapl_stock_model.pkl"
REFRESH_SECS = 30
ET           = ZoneInfo("America/New_York")

# ── Resolve API key: Streamlit Secrets > env var > hardcoded fallback ─────────
def get_api_key() -> str:
    try:
        return st.secrets["TWELVE_DATA_API_KEY"]
    except Exception:
        return API_KEY   # falls back to the key in train_model.py


# ── Helpers ───────────────────────────────────────────────────────────────────
def market_is_closed() -> bool:
    now = datetime.now(ET)
    return now.weekday() >= 5 or now.hour >= 16

def model_is_stale(max_age_hours=24) -> bool:
    if not os.path.exists(MODEL_PATH):
        return True
    mtime = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH), tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime) > timedelta(hours=max_age_hours)

def fetch_live_quote(ticker: str) -> dict:
    try:
        r = requests.get(
            "https://api.twelvedata.com/quote",
            params={"symbol": ticker, "apikey": get_api_key()},
            timeout=10
        )
        d = r.json()
        return {
            "price":      float(d.get("close", 0)),
            "open":       float(d.get("open", 0)),
            "high":       float(d.get("high", 0)),
            "low":        float(d.get("low", 0)),
            "change":     float(d.get("change", 0)),
            "pct_change": float(d.get("percent_change", 0)),
            "volume":     int(float(d.get("volume", 0))),
            "timestamp":  d.get("datetime", ""),
        }
    except Exception as e:
        return {"error": str(e)}

def do_retrain(ticker: str) -> dict:
    m = StockPredictionModel(ticker, get_api_key())
    m.fetch_and_preprocess()
    rmse, r2, mae = m.train_and_evaluate(MODEL_PATH)
    return {
        "rmse": rmse, "r2": r2, "mae": mae,
        "trained_at": datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")
    }


# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data(ttl=3600)
def get_df(ticker: str) -> pd.DataFrame:
    raw = fetch_historical(ticker, get_api_key())
    return build_features(raw)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("👨‍💻 Developer")
    st.markdown("**Abhishek Shelke**")
    st.markdown("M.Sc Computer Science  \nASM's CSIT, Pimpri")
    st.markdown("**Interests:** Data Science · Machine Learning · AI")
    st.markdown("[GitHub](https://github.com/Redskull2525) | "
                "[LinkedIn](https://www.linkedin.com/in/abhishek-s-b98895249)")
    st.divider()

    ticker = st.text_input("Stock Ticker", value="AAPL").upper()
    period = st.selectbox("Data Period", ["1y", "2y", "3y", "5y"], index=3)
    st.divider()

    # ── Model status ──────────────────────────────────────────────────────────
    st.markdown("### 🤖 Model Status")
    stale = model_is_stale()

    if os.path.exists(MODEL_PATH):
        try:
            meta = joblib.load(MODEL_PATH)
            st.success(
                f"✅ Model ready\n\n"
                f"🕒 `{meta.get('trained_at','unknown')}`\n\n"
                f"R²: `{meta.get('r2', 0):.4f}` | RMSE: `{meta.get('rmse', 0):.4f}`"
            )
        except Exception:
            st.warning("⚠️ Could not read model metadata.")
    else:
        st.error("❌ No model found — retrain below.")

    if stale:
        st.warning("⚠️ Model is stale (> 24 h)")

    if st.button("🔄 Retrain Now", type="primary", use_container_width=True):
        with st.spinner("Fetching data & training... (~30s)"):
            result = do_retrain(ticker)
        st.success(
            f"✅ Done! Trained at {result['trained_at']}\n\n"
            f"R²: `{result['r2']:.4f}` | RMSE: `{result['rmse']:.4f}`"
        )
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    st.divider()
    auto_refresh = st.toggle("⚡ Live Price (30s refresh)", value=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTO-RETRAIN  (stale + market closed)
# ═══════════════════════════════════════════════════════════════════════════════
if stale and market_is_closed():
    with st.spinner("🔄 Auto-retraining with latest data..."):
        result = do_retrain(ticker)
    st.toast(f"✅ Auto-retrained! R² = {result['r2']:.4f}")
    st.cache_resource.clear()
    st.cache_data.clear()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
st.title("📈 Stock Price Prediction Dashboard")

# ── Live quote ────────────────────────────────────────────────────────────────
st.subheader(f"⚡ Live Quote — {ticker}")
quote = fetch_live_quote(ticker)
q_container = st.empty()

if "error" not in quote:
    with q_container.container():
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Price",  f"${quote['price']:.2f}",
                  f"{quote['change']:+.2f} ({quote['pct_change']:+.2f}%)")
        c2.metric("Open",   f"${quote['open']:.2f}")
        c3.metric("High",   f"${quote['high']:.2f}")
        c4.metric("Low",    f"${quote['low']:.2f}")
        c5.metric("Volume", f"{quote['volume']:,}")
        st.caption(
            f"Last update: {quote['timestamp']}  |  "
            f"{'🟢 Market Open' if not market_is_closed() else '🔴 Market Closed'}"
        )
else:
    st.warning(f"Live feed error: {quote['error']}")

# ── Historical data & predictions ─────────────────────────────────────────────
try:
    df   = get_df(ticker)
    meta = load_model()
    model, scaler, features = meta["model"], meta["scaler"], meta["features"]

    pred_returns    = model.predict(scaler.transform(df[features]))
    prev_closes     = df['Close'].shift(1).fillna(df['Close'].iloc[0]).values
    df              = df.copy()
    df['Predicted'] = prev_closes * (1 + pred_returns)

    # ── Latest data table ─────────────────────────────────────────────────────
    st.subheader("📋 Latest Stock Data")
    st.dataframe(
        df[['Close','High','Low','Open','Volume','SMA_7','SMA_21']].tail(5),
        use_container_width=True
    )

    # ── Actual vs Predicted chart ─────────────────────────────────────────────
    st.subheader("📊 Actual vs Predicted Price")
    split   = int(len(df) * 0.8)
    test_df = df.iloc[split:].copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(test_df))), y=test_df['Close'],
        mode='lines', name='Actual Price',
        line=dict(color='white', width=1.5)
    ))
    fig.add_trace(go.Scatter(
        x=list(range(len(test_df))), y=test_df['Predicted'],
        mode='lines', name='Predicted Price',
        line=dict(color='cyan', width=1.5, dash='dot')
    ))
    if "price" in quote:
        fig.add_hline(
            y=quote["price"], line_dash="dash", line_color="orange",
            annotation_text=f"Live: ${quote['price']:.2f}",
            annotation_position="top right"
        )
    fig.update_layout(
        template='plotly_dark',
        xaxis_title='Trading Days (Test Set)',
        yaxis_title='Price (USD)',
        legend=dict(orientation='h', y=1.1),
        height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Next-day prediction ───────────────────────────────────────────────────
    st.subheader("🔮 Next-Day Price Prediction")
    pred_return   = model.predict(scaler.transform(df[features].tail(1)))[0]
    last_close    = float(df['Close'].iloc[-1])
    next_day_pred = last_close * (1 + pred_return)
    delta         = next_day_pred - last_close

    p1, p2, p3 = st.columns(3)
    p1.metric("Last Close",      f"${last_close:.2f}")
    p2.metric("Predicted Close", f"${next_day_pred:.2f}", f"{delta:+.2f}")
    p3.metric("Expected Move",   f"{pred_return * 100:+.3f}%")

    # ── Model metrics ─────────────────────────────────────────────────────────
    st.subheader("📐 Model Performance (Test Set)")
    mse  = mean_squared_error(test_df['Close'], test_df['Predicted'])
    rmse = np.sqrt(mse)
    r2   = r2_score(test_df['Close'], test_df['Predicted'])

    m1, m2, m3 = st.columns(3)
    m1.metric("RMSE",     f"{rmse:.4f}")
    m2.metric("MSE",      f"{mse:.4f}")
    m3.metric("R² Score", f"{r2:.4f}")

except FileNotFoundError:
    st.error("❌ No trained model found. Click **Retrain Now** in the sidebar.")
except Exception as e:
    st.error(f"Error: {e}")
    st.exception(e)


# ── Auto-refresh live price during market hours ───────────────────────────────
if auto_refresh:
    if not market_is_closed():
        time.sleep(REFRESH_SECS)
        st.rerun()
    else:
        st.caption("🔴 Market closed — live refresh paused.")
