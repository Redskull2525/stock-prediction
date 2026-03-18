import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from train_model import StockPredictionModel   # reuse feature engineering


# ── Page config ──────────────────────────────────────────────────────────────
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

# ── Load & preprocess ─────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data(ticker, period):
    m = StockPredictionModel(ticker, period)
    m.fetch_and_preprocess()
    return m

@st.cache_resource
def load_model():
    return joblib.load("aapl_stock_model.pkl")

with st.spinner("Fetching data..."):
    stock_model = load_data(ticker, period)
    df = stock_model.df

saved = load_model()
model    = saved["model"]
scaler   = saved["scaler"]
features = saved["features"]

# ── Align features (model trained on AAPL features; app may use different ticker) ──
X = df[features]
X_scaled = scaler.transform(X)
df['Predicted'] = model.predict(X_scaled)

# ── Latest data table ─────────────────────────────────────────────────────────
st.title("📈 Stock Price Prediction Dashboard")
st.subheader("📋 Latest Stock Data")
display_cols = ['Close', 'High', 'Low', 'Open', 'Volume', 'SMA_7', 'SMA_21']
st.dataframe(
    df[display_cols].tail(5).rename(index=lambda i: df.index[i] if hasattr(df.index, '__getitem__') else i),
    use_container_width=True
)

# ── Actual vs Predicted chart ─────────────────────────────────────────────────
st.subheader("📊 Actual vs Predicted Price")

split = int(len(df) * 0.8)
test_df = df.iloc[split:].copy()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=list(range(len(test_df))),
    y=test_df['Close'],
    mode='lines', name='Actual Price',
    line=dict(color='white', width=1.5)
))
fig.add_trace(go.Scatter(
    x=list(range(len(test_df))),
    y=test_df['Predicted'],
    mode='lines', name='Predicted Price',
    line=dict(color='cyan', width=1.5, dash='dot')
))
fig.update_layout(
    template='plotly_dark',
    xaxis_title='Trading Days (Test Set)',
    yaxis_title='Price (USD)',
    legend=dict(orientation='h', y=1.1),
    height=450
)
st.plotly_chart(fig, use_container_width=True)

# ── Next-day prediction ───────────────────────────────────────────────────────
st.subheader("🔮 Next-Day Price Prediction")
latest_features = df[features].tail(1)
latest_scaled   = scaler.transform(latest_features)
next_day_pred   = model.predict(latest_scaled)[0]
last_close      = df['Close'].iloc[-1]
delta           = next_day_pred - last_close
pct             = (delta / last_close) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Last Close",      f"${last_close:.2f}")
col2.metric("Predicted Close", f"${next_day_pred:.2f}", f"{delta:+.2f}")
col3.metric("Expected Move",   f"{pct:+.2f}%")

# ── Model metrics ─────────────────────────────────────────────────────────────
from sklearn.metrics import mean_squared_error, r2_score
mse  = mean_squared_error(test_df['Close'], test_df['Predicted'])
rmse = np.sqrt(mse)
r2   = r2_score(test_df['Close'], test_df['Predicted'])

st.subheader("📐 Model Performance (Test Set)")
m1, m2, m3 = st.columns(3)
m1.metric("RMSE",    f"{rmse:.4f}")
m2.metric("MSE",     f"{mse:.4f}")
m3.metric("R² Score",f"{r2:.4f}")
