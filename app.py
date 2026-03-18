import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stock Predictor", layout="wide", page_icon="📈")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("👨‍💻 Developer")
    st.markdown("**Abhishek Shelke**")
    st.markdown("M.Sc Computer Science  \nASM's CSIT, Pimpri")
    st.markdown("**Interests:** Data Science · Machine Learning · AI")
    st.markdown("[GitHub](https://github.com/Redskull2525) | [LinkedIn](https://www.linkedin.com/in/abhishek-s-b98895249)")
    st.divider()
    ticker = st.text_input("Stock Ticker", value="AAPL").upper()
    period = st.selectbox("Data Period", ["1y", "2y", "3y", "5y"], index=3)


# ── Feature engineering (mirrors train_model.py exactly) ─────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast    = series.ewm(span=fast, adjust=False).mean()
    ema_slow    = series.ewm(span=slow, adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_bollinger(series, period=20):
    mid   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    width = (upper - lower) / (mid + 1e-9)
    pct_b = (series - lower) / (upper - lower + 1e-9)
    return upper, mid, lower, width, pct_b

def build_features(df):
    for w in [5, 7, 10, 21, 50]:
        df[f'SMA_{w}'] = df['Close'].rolling(w).mean()
        df[f'EMA_{w}'] = df['Close'].ewm(span=w, adjust=False).mean()
    for w in [5, 10, 21, 50]:
        df[f'Close_SMA{w}_ratio'] = df['Close'] / (df[f'SMA_{w}'] + 1e-9)
    for lag in [1, 2, 3, 5, 10]:
        df[f'Close_lag_{lag}']  = df['Close'].shift(lag)
        df[f'Return_lag_{lag}'] = df['Close'].pct_change(lag)
    df['Volume_SMA_10'] = df['Volume'].rolling(10).mean()
    df['Volume_ratio']  = df['Volume'] / (df['Volume_SMA_10'] + 1e-9)
    df['Momentum_5']    = df['Close'] - df['Close'].shift(5)
    df['Momentum_10']   = df['Close'] - df['Close'].shift(10)
    df['Volatility_10'] = df['Close'].pct_change().rolling(10).std()
    df['Volatility_21'] = df['Close'].pct_change().rolling(21).std()
    df['Daily_Range']   = df['High'] - df['Low']
    df['Body']          = abs(df['Close'] - df['Open'])
    df['Upper_Wick']    = df['High'] - df[['Close', 'Open']].max(axis=1)
    df['Lower_Wick']    = df[['Close', 'Open']].min(axis=1) - df['Low']
    df['RSI_14'] = compute_rsi(df['Close'], 14)
    df['RSI_7']  = compute_rsi(df['Close'], 7)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = compute_macd(df['Close'])
    df['BB_upper'], df['BB_mid'], df['BB_lower'], df['BB_width'], df['BB_pct'] = \
        compute_bollinger(df['Close'])
    for p in [5, 10, 21]:
        df[f'ROC_{p}'] = df['Close'].pct_change(p) * 100
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna().reset_index(drop=True)
    return df


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data(ticker, period):
    raw = yf.download(ticker, period=period, interval='1d', auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    return build_features(raw.copy())

@st.cache_resource
def load_model():
    return joblib.load("aapl_stock_model.pkl")


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("📈 Stock Price Prediction Dashboard")

with st.spinner("Fetching & processing data..."):
    df = load_data(ticker, period)

saved    = load_model()
model    = saved["model"]
scaler   = saved["scaler"]
features = saved["features"]

X_scaled        = scaler.transform(df[features])
df              = df.copy()
df['Predicted'] = model.predict(X_scaled)

# ── Latest data table ─────────────────────────────────────────────────────────
st.subheader("📋 Latest Stock Data")
st.dataframe(df[['Close','High','Low','Open','Volume','SMA_7','SMA_21']].tail(5),
             use_container_width=True)

# ── Actual vs Predicted chart ─────────────────────────────────────────────────
st.subheader("📊 Actual vs Predicted Price")
split   = int(len(df) * 0.8)
test_df = df.iloc[split:].copy()

fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(len(test_df))), y=test_df['Close'],
    mode='lines', name='Actual Price', line=dict(color='white', width=1.5)))
fig.add_trace(go.Scatter(x=list(range(len(test_df))), y=test_df['Predicted'],
    mode='lines', name='Predicted Price', line=dict(color='cyan', width=1.5, dash='dot')))
fig.update_layout(template='plotly_dark', xaxis_title='Trading Days (Test Set)',
    yaxis_title='Price (USD)', legend=dict(orientation='h', y=1.1), height=450)
st.plotly_chart(fig, use_container_width=True)

# ── Next-day prediction ───────────────────────────────────────────────────────
st.subheader("🔮 Next-Day Price Prediction")
next_day_pred = model.predict(scaler.transform(df[features].tail(1)))[0]
last_close    = float(df['Close'].iloc[-1])
delta         = next_day_pred - last_close
pct           = (delta / last_close) * 100

c1, c2, c3 = st.columns(3)
c1.metric("Last Close",      f"${last_close:.2f}")
c2.metric("Predicted Close", f"${next_day_pred:.2f}", f"{delta:+.2f}")
c3.metric("Expected Move",   f"{pct:+.2f}%")

# ── Model metrics ─────────────────────────────────────────────────────────────
st.subheader("📐 Model Performance (Test Set)")
mse  = mean_squared_error(test_df['Close'], test_df['Predicted'])
rmse = np.sqrt(mse)
r2   = r2_score(test_df['Close'], test_df['Predicted'])

m1, m2, m3 = st.columns(3)
m1.metric("RMSE",     f"{rmse:.4f}")
m2.metric("MSE",      f"{mse:.4f}")
m3.metric("R² Score", f"{r2:.4f}")
