"""
app.py  —  the Streamlit dashboard: data fetching, UI and the saved-model file.

All the date-sensitive logic (features, targets, split, return -> price,
next-day row, metrics) lives in stock_core.py, which has unit tests in tests/.
"""

import streamlit as st
import requests
import pandas as pd
import joblib
import plotly.graph_objects as go
import time, os
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import stock_core as core

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Predictor", layout="wide", page_icon="📈")

# STOCK_MODEL_PATH lets the tests use a temporary file instead of the real model.
MODEL_PATH   = os.environ.get("STOCK_MODEL_PATH", "aapl_stock_model.pkl")
REFRESH_SECS = 30
ET           = ZoneInfo("America/New_York")

# ── API key: Streamlit Secrets → env var (never hardcode keys) ───────────────
def get_api_key() -> str:
    try:
        return st.secrets["TWELVE_DATA_API_KEY"]
    except Exception:
        key = os.environ.get("TWELVE_DATA_API_KEY")
        if not key:
            st.error("TWELVE_DATA_API_KEY is not set. Add it to .streamlit/secrets.toml "
                     "or as an environment variable (see README).")
            st.stop()
        return key


# ══════════════════════════════════════════════════════════════════════════════
#  DATA FETCHING
# ══════════════════════════════════════════════════════════════════════════════
def fetch_historical(ticker: str, outputsize: int = 5000) -> pd.DataFrame:
    resp = requests.get(
        "https://api.twelvedata.com/time_series",
        params={"symbol": ticker, "interval": "1day",
                "outputsize": outputsize, "apikey": get_api_key(), "format": "JSON"},
        timeout=30
    )
    resp.raise_for_status()
    data = resp.json()
    if "values" not in data:
        raise ValueError(f"Twelve Data error: {data.get('message', data)}")
    df = pd.DataFrame(data["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.rename(columns={"datetime":"Date","open":"Open","high":"High",
                               "low":"Low","close":"Close","volume":"Volume"}
               )[["Date","Open","High","Low","Close","Volume"]]

def fetch_live_quote(ticker: str) -> dict:
    try:
        d = requests.get("https://api.twelvedata.com/quote",
                         params={"symbol": ticker, "apikey": get_api_key()},
                         timeout=10).json()
        return {"price": float(d.get("close", 0)),
                "open":  float(d.get("open", 0)),
                "high":  float(d.get("high", 0)),
                "low":   float(d.get("low", 0)),
                "change": float(d.get("change", 0)),
                "pct_change": float(d.get("percent_change", 0)),
                "volume": int(float(d.get("volume", 0))),
                "timestamp": d.get("datetime", "")}
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN  (the logic is in core.train_bundle; this just fetches and saves)
# ══════════════════════════════════════════════════════════════════════════════
def train(ticker: str) -> dict:
    bundle = core.train_bundle(fetch_historical(ticker), ticker,
                               trained_at=datetime.now(ET).strftime("%Y-%m-%d %H:%M ET"))
    joblib.dump(bundle, MODEL_PATH)
    return bundle


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def market_is_closed() -> bool:
    return core.market_is_closed(datetime.now(ET))

def model_is_stale(hours=24) -> bool:
    if not os.path.exists(MODEL_PATH):
        return True
    mtime = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH), tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime) > timedelta(hours=hours)

@st.cache_resource
def _load_bundle(path: str, mtime: float) -> dict:
    return joblib.load(path)            # mtime is only part of the cache key

def load_model():
    """Return (bundle, error_message). bundle is None if missing or unreadable."""
    if not os.path.exists(MODEL_PATH):
        return None, None
    try:
        return _load_bundle(MODEL_PATH, os.path.getmtime(MODEL_PATH)), None
    except Exception as e:              # e.g. saved with an incompatible scikit-learn
        return None, f"{type(e).__name__}: {e}"

@st.cache_data(ttl=3600)
def get_df(ticker: str) -> pd.DataFrame:
    """Feature table for every downloaded day, including the newest one."""
    return core.build_feature_table(fetch_historical(ticker))

def pct(x) -> str:
    return "n/a" if x is None or pd.isna(x) else f"{x*100:.1f}%"


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("👨‍💻 Developer")
    st.markdown("**Abhishek Shelke**")
    st.markdown("M.Sc Computer Science  \nASM's CSIT, Pimpri")
    st.markdown("**Interests:** Data Science · Machine Learning · AI")
    st.markdown("[GitHub](https://github.com/Redskull2525) | "
                "[LinkedIn](https://www.linkedin.com/in/abhishek-s-b98895249)")
    st.divider()

    ticker = st.text_input("Stock Ticker", value="AAPL").strip().upper()
    period = st.selectbox("Data Period", ["1y","2y","3y","5y"], index=3)
    st.divider()

    st.markdown("### 🤖 Model Status")
    stale = model_is_stale()
    bundle, load_error = load_model()

    if load_error:
        st.warning(f"⚠️ Saved model could not be loaded — click Retrain.\n\n`{load_error}`")
    elif bundle is None:
        st.error("❌ No model — click Retrain.")
    elif core.is_legacy_bundle(bundle):
        st.info(f"✅ Model ready (old format, {bundle.get('ticker', '?')})\n\n"
                f"🕒 `{bundle.get('trained_at','unknown')}`\n\n"
                "Its saved metrics were computed by the old, buggy code, so they "
                "are not shown. Retrain to get corrected metrics and the naive baseline.")
    else:
        st.success(f"✅ Model ready — **{bundle['ticker']}**\n\n"
                   f"🕒 `{bundle['trained_at']}`\n\n"
                   f"Train: `{bundle['train_start_date']}` → `{bundle['train_end_date']}`  \n"
                   f"Test: `{bundle['test_start_date']}` → `{bundle['test_end_date']}`")
        st.markdown(
            "| Test set | Model | Naive (tomorrow = today) |\n|---|---|---|\n"
            f"| RMSE | `{bundle['rmse']:.4f}` | `{bundle['baseline_rmse']:.4f}` |\n"
            f"| Direction right | `{pct(bundle['directional_accuracy'])}` "
            f"| `{pct(bundle['up_day_share'])}` (always 'up') |\n\n"
            f"R²: `{bundle['r2']:.4f}` · Skill vs naive: `{bundle['skill']:+.4f}` "
            "(> 0 means the model beats naive)")

    if bundle is not None and core.ticker_mismatch(bundle, ticker):
        st.warning(f"⚠️ This model was trained on **{bundle.get('ticker', 'an unknown ticker')}**, "
                   f"not {ticker}. Click Retrain to train one for {ticker}.")

    if stale:
        st.warning("⚠️ Model is stale (> 24 h)")

    if st.button("🔄 Retrain Now", type="primary", width="stretch"):
        with st.spinner("Fetching data & training... (~30s)"):
            result = train(ticker)
        st.success(f"✅ Trained at {result['trained_at']}\n\n"
                   f"R²: `{result['r2']:.4f}` | RMSE: `{result['rmse']:.4f}`")
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

    st.divider()
    auto_refresh = st.toggle("⚡ Live Price (30s)", value=True)


# ══════════════════════════════════════════════════════════════════════════════
#  AUTO-RETRAIN on startup if stale (or unreadable) + market closed
# ══════════════════════════════════════════════════════════════════════════════
if (stale or bundle is None) and market_is_closed():
    with st.spinner("🔄 Auto-retraining with latest data..."):
        bundle = train(ticker)
    st.toast(f"✅ Auto-retrained! R² = {bundle['r2']:.4f}")
    st.cache_resource.clear()
    st.cache_data.clear()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
st.title("📈 Stock Price Prediction Dashboard")

# ── Live quote ────────────────────────────────────────────────────────────────
st.subheader(f"⚡ Live Quote — {ticker}")
quote = fetch_live_quote(ticker)

if "error" not in quote:
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Price",  f"${quote['price']:.2f}",
              f"{quote['change']:+.2f} ({quote['pct_change']:+.2f}%)")
    c2.metric("Open",   f"${quote['open']:.2f}")
    c3.metric("High",   f"${quote['high']:.2f}")
    c4.metric("Low",    f"${quote['low']:.2f}")
    c5.metric("Volume", f"{quote['volume']:,}")
    st.caption(f"Last update: {quote['timestamp']}  |  "
               f"{'🟢 Market Open' if not market_is_closed() else '🔴 Market Closed'}")
else:
    st.warning(f"Live feed error: {quote['error']}")

# ── Historical predictions ────────────────────────────────────────────────────
def show_predictions(bundle: dict, ticker: str, quote: dict) -> None:
    if core.ticker_mismatch(bundle, ticker):
        # Refuse rather than score one stock with another stock's model (bug B4).
        st.warning(f"The saved model was trained on **{bundle.get('ticker', 'an unknown ticker')}**. "
                   f"Click **Retrain Now** in the sidebar to train a model for **{ticker}**.")
        return

    model, scaler, features = bundle["model"], bundle["scaler"], bundle["features"]
    df = get_df(ticker)        # every day, including the newest (its target is NaN)

    # Latest data table
    st.subheader("📋 Latest Stock Data")
    st.dataframe(df.set_index("Date")[['Close','High','Low','Open','Volume','SMA_7','SMA_21']].tail(5),
                 width="stretch")

    # Chart: test-period predictions, each plotted on the date it is FOR
    st.subheader("📊 Actual vs Predicted Price")
    rows    = core.training_rows(df, features)
    test_df = core.predict_frame(model, scaler, core.evaluation_window(rows, bundle), features)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Actual'],
        mode='lines', name='Actual Price', line=dict(color='white', width=1.5)))
    fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Predicted'],
        mode='lines', name='Predicted Price', line=dict(color='cyan', width=1.5, dash='dot')))
    if "price" in quote:
        fig.add_hline(y=quote["price"], line_dash="dash", line_color="orange",
                      annotation_text=f"Live: ${quote['price']:.2f}",
                      annotation_position="top right")
    fig.update_layout(template='plotly_dark', xaxis_title='Date (test period — not used for training)',
                      yaxis_title='Price (USD)', legend=dict(orientation='h', y=1.1),
                      height=450)
    st.plotly_chart(fig, width="stretch")
    st.caption("Each predicted point is yesterday's close × (1 + the model's predicted return), "
               "plotted on the day it predicts.")
    if core.is_legacy_bundle(bundle):
        st.caption("⚠️ Old-format model: its test start date wasn't saved, so the test window is "
                   "re-estimated as the last 20% of today's data and may overlap its training days.")

    # Next-day prediction (newest row with complete features)
    st.subheader("🔮 Next-Day Price Prediction")
    fc    = core.next_day_forecast(model, scaler, df, features)
    delta = fc["predicted_close"] - fc["base_close"]

    p1,p2,p3 = st.columns(3)
    p1.metric(f"Last Close ({fc['base_date']:%Y-%m-%d})", f"${fc['base_close']:.2f}")
    p2.metric(f"Predicted Close ({fc['target_date']:%a %Y-%m-%d})",
              f"${fc['predicted_close']:.2f}", f"{delta:+.2f}")
    p3.metric("Expected Move", f"{fc['predicted_return']*100:+.3f}%")
    st.caption("Forecast date = next weekday after the latest daily bar (exchange holidays are not "
               "accounted for). A learning project, not trading advice.")

    # Metrics, with the naive baseline next to the model
    st.subheader("📐 Model Performance (Test Set)")
    m = core.evaluate_predictions(test_df)
    m1,m2,m3 = st.columns(3)
    m1.metric("RMSE",     f"{m['rmse']:.4f}")
    m2.metric("MSE",      f"{m['rmse']**2:.4f}")
    m3.metric("R² Score", f"{m['r2']:.4f}")
    b1,b2,b3,b4 = st.columns(4)
    b1.metric("Naive RMSE (tomorrow = today)", f"{m['baseline_rmse']:.4f}")
    b2.metric("Skill vs naive", f"{m['skill']:+.4f}", help="1 − RMSE / naive RMSE. Above 0 = beats naive.")
    b3.metric("Directional accuracy", pct(m['directional_accuracy']))
    b4.metric("Up-day share ('always up')", pct(m['up_day_share']))
    st.caption(f"{m['n_test_days']} test days. A high R² on price is easy (naive scores high too); "
               "compare RMSE with the naive RMSE and directional accuracy with the up-day share.")


try:
    if bundle is None:
        raise FileNotFoundError(MODEL_PATH)
    show_predictions(bundle, ticker, quote)
except FileNotFoundError:
    st.error("❌ No trained model found. Click **Retrain Now** in the sidebar.")
except Exception as e:
    st.error(f"Error: {e}")
    st.exception(e)

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    if not market_is_closed():
        time.sleep(REFRESH_SECS)
        st.rerun()
    else:
        st.caption("🔴 Market closed — live refresh paused.")
