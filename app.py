"""
app.py  —  Fully self-contained. No imports from train_model.
All feature engineering, training, and inference live here.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import time, os
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Predictor", layout="wide", page_icon="📈")

MODEL_PATH   = "aapl_stock_model.pkl"
REFRESH_SECS = 30
ET           = ZoneInfo("America/New_York")

# ── API key: Streamlit Secrets → env var → hardcoded fallback ────────────────
def get_api_key() -> str:
    try:
        return st.secrets["TWELVE_DATA_API_KEY"]
    except Exception:
        return os.environ.get("TWELVE_DATA_API_KEY", "b6c3629a148846c1b256838291330785")


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
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def _rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - (100 / (1 + g / (l + 1e-9)))

def _macd(s, fast=12, slow=26, sig=9):
    m = s.ewm(span=fast,adjust=False).mean() - s.ewm(span=slow,adjust=False).mean()
    sg = m.ewm(span=sig,adjust=False).mean()
    return m, sg, m - sg

def _bb(s, p=20):
    mid = s.rolling(p).mean(); std = s.rolling(p).std()
    u, l = mid + 2*std, mid - 2*std
    return (u-l)/(mid+1e-9), (s-l)/(u-l+1e-9)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for w in [5,7,10,21,50]:
        df[f'SMA_{w}'] = df['Close'].rolling(w).mean()
        df[f'EMA_{w}'] = df['Close'].ewm(span=w,adjust=False).mean()
    for w in [5,10,21,50]:
        df[f'Close_SMA{w}_ratio'] = df['Close'] / (df[f'SMA_{w}'] + 1e-9)
    for lag in [1,2,3,5,10]:
        df[f'Return_lag_{lag}'] = df['Close'].pct_change(lag)
    for lag in [1,2,3,5]:
        df[f'Close_lag_{lag}_norm'] = df['Close'].shift(lag) / df['Close']
    df['Volume_SMA_10'] = df['Volume'].rolling(10).mean()
    df['Volume_ratio']  = df['Volume'] / (df['Volume_SMA_10'] + 1e-9)
    df['Momentum_5']    = df['Close'].pct_change(5)
    df['Momentum_10']   = df['Close'].pct_change(10)
    df['Volatility_10'] = df['Close'].pct_change().rolling(10).std()
    df['Volatility_21'] = df['Close'].pct_change().rolling(21).std()
    df['Daily_Range']   = (df['High'] - df['Low']) / df['Close']
    df['Body']          = abs(df['Close'] - df['Open']) / df['Close']
    df['Upper_Wick']    = (df['High'] - df[['Close','Open']].max(axis=1)) / df['Close']
    df['Lower_Wick']    = (df[['Close','Open']].min(axis=1) - df['Low'])  / df['Close']
    df['RSI_14'] = _rsi(df['Close'], 14)
    df['RSI_7']  = _rsi(df['Close'], 7)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = _macd(df['Close'])
    for c in ['MACD','MACD_signal','MACD_hist']:
        df[c] /= df['Close']
    df['BB_width'], df['BB_pct'] = _bb(df['Close'])
    for p in [5,10,21]:
        df[f'ROC_{p}'] = df['Close'].pct_change(p) * 100
    df['Target_return'] = df['Close'].pct_change().shift(-1)
    df['Target_price']  = df['Close'].shift(-1)
    return df.dropna().reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN
# ══════════════════════════════════════════════════════════════════════════════
def train(ticker: str) -> dict:
    raw = fetch_historical(ticker)
    df  = build_features(raw)
    exclude  = {'Date','Target_return','Target_price'}
    features = [c for c in df.columns if c not in exclude]

    X, y = df[features], df['Target_return']
    split = int(len(X) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    rf = RandomForestRegressor(n_estimators=500, max_depth=10,
             min_samples_split=5, min_samples_leaf=2,
             max_features=0.7, random_state=42, n_jobs=-1)
    gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
             max_depth=5, subsample=0.8, random_state=42)
    model = VotingRegressor([('rf', rf), ('gb', gb)])
    model.fit(X_tr_s, y_tr)

    pred_returns = model.predict(X_te_s)
    prev_closes  = df['Close'].iloc[split-1:-1].values
    y_pred_price = prev_closes * (1 + pred_returns)
    y_test_price = df['Target_price'].iloc[split:].values

    rmse = float(np.sqrt(mean_squared_error(y_test_price, y_pred_price)))
    r2   = float(r2_score(y_test_price, y_pred_price))
    mae  = float(np.mean(np.abs(y_test_price - y_pred_price)))

    joblib.dump({"model": model, "scaler": scaler, "features": features,
                 "ticker": ticker, "rmse": rmse, "r2": r2, "mae": mae,
                 "trained_at": datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")},
                MODEL_PATH)
    return {"rmse": rmse, "r2": r2, "mae": mae,
            "trained_at": datetime.now(ET).strftime("%Y-%m-%d %H:%M ET")}


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def market_is_closed() -> bool:
    now = datetime.now(ET)
    return now.weekday() >= 5 or now.hour >= 16

def model_is_stale(hours=24) -> bool:
    if not os.path.exists(MODEL_PATH):
        return True
    mtime = datetime.fromtimestamp(os.path.getmtime(MODEL_PATH), tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime) > timedelta(hours=hours)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data(ttl=3600)
def get_df(ticker: str) -> pd.DataFrame:
    return build_features(fetch_historical(ticker))


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

    ticker = st.text_input("Stock Ticker", value="AAPL").upper()
    period = st.selectbox("Data Period", ["1y","2y","3y","5y"], index=3)
    st.divider()

    st.markdown("### 🤖 Model Status")
    stale = model_is_stale()

    if os.path.exists(MODEL_PATH):
        try:
            meta = joblib.load(MODEL_PATH)
            st.success(f"✅ Model ready\n\n"
                       f"🕒 `{meta.get('trained_at','unknown')}`\n\n"
                       f"R²: `{meta.get('r2',0):.4f}` | RMSE: `{meta.get('rmse',0):.4f}`")
        except Exception:
            st.warning("⚠️ Could not read model metadata.")
    else:
        st.error("❌ No model — click Retrain.")

    if stale:
        st.warning("⚠️ Model is stale (> 24 h)")

    if st.button("🔄 Retrain Now", type="primary", use_container_width=True):
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
#  AUTO-RETRAIN on startup if stale + market closed
# ══════════════════════════════════════════════════════════════════════════════
if stale and market_is_closed():
    with st.spinner("🔄 Auto-retraining with latest data..."):
        result = train(ticker)
    st.toast(f"✅ Auto-retrained! R² = {result['r2']:.4f}")
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
try:
    df   = get_df(ticker)
    meta = load_model()
    model, scaler, features = meta["model"], meta["scaler"], meta["features"]

    pred_returns    = model.predict(scaler.transform(df[features]))
    prev_closes     = df['Close'].shift(1).fillna(df['Close'].iloc[0]).values
    df              = df.copy()
    df['Predicted'] = prev_closes * (1 + pred_returns)

    # Latest data table
    st.subheader("📋 Latest Stock Data")
    st.dataframe(df[['Close','High','Low','Open','Volume','SMA_7','SMA_21']].tail(5),
                 use_container_width=True)

    # Chart
    st.subheader("📊 Actual vs Predicted Price")
    split   = int(len(df) * 0.8)
    test_df = df.iloc[split:].copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(test_df))), y=test_df['Close'],
        mode='lines', name='Actual Price', line=dict(color='white', width=1.5)))
    fig.add_trace(go.Scatter(x=list(range(len(test_df))), y=test_df['Predicted'],
        mode='lines', name='Predicted Price', line=dict(color='cyan', width=1.5, dash='dot')))
    if "price" in quote:
        fig.add_hline(y=quote["price"], line_dash="dash", line_color="orange",
                      annotation_text=f"Live: ${quote['price']:.2f}",
                      annotation_position="top right")
    fig.update_layout(template='plotly_dark', xaxis_title='Trading Days (Test Set)',
                      yaxis_title='Price (USD)', legend=dict(orientation='h', y=1.1),
                      height=450)
    st.plotly_chart(fig, use_container_width=True)

    # Next-day prediction
    st.subheader("🔮 Next-Day Price Prediction")
    pred_return   = model.predict(scaler.transform(df[features].tail(1)))[0]
    last_close    = float(df['Close'].iloc[-1])
    next_day_pred = last_close * (1 + pred_return)
    delta         = next_day_pred - last_close

    p1,p2,p3 = st.columns(3)
    p1.metric("Last Close",      f"${last_close:.2f}")
    p2.metric("Predicted Close", f"${next_day_pred:.2f}", f"{delta:+.2f}")
    p3.metric("Expected Move",   f"{pred_return*100:+.3f}%")

    # Metrics
    st.subheader("📐 Model Performance (Test Set)")
    mse  = mean_squared_error(test_df['Close'], test_df['Predicted'])
    rmse = np.sqrt(mse)
    r2   = r2_score(test_df['Close'], test_df['Predicted'])
    m1,m2,m3 = st.columns(3)
    m1.metric("RMSE",     f"{rmse:.4f}")
    m2.metric("MSE",      f"{mse:.4f}")
    m3.metric("R² Score", f"{r2:.4f}")

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
