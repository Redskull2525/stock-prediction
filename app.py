```python
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# -------- PAGE CONFIG -------- #
st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")

# -------- SIDEBAR -------- #
st.sidebar.title("👨‍💻 Developer")

st.sidebar.markdown("""
**Abhishek Shelke**

M.Sc Computer Science  
ASM's CSIT, Pimpri  

**Interests**
- Data Science
- Machine Learning
- AI

GitHub  
https://github.com/Redskull2525

LinkedIn  
https://www.linkedin.com/in/abhishek-s-b98895249
""")

# -------- INPUT -------- #
ticker = st.sidebar.text_input("Stock Ticker", "AAPL").upper().strip()
period = st.sidebar.selectbox("Data Period", ["6mo", "1y", "2y", "5y"])

# -------- LOAD MODEL -------- #
try:
    model = joblib.load("aapl_stock_model.pkl")
except:
    st.error("❌ Model file not found. Make sure 'aapl_stock_model.pkl' is uploaded.")
    st.stop()

features = ['Close','High','Low','Open','Volume','SMA_7','SMA_21']

# -------- LOAD DATA FUNCTION -------- #
@st.cache_data
def load_data(ticker, period):
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True)

        # Check if data exists
        if df is None or df.empty:
            return None

        # Fix multi-index issue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Required columns check
        required_cols = ["Close", "High", "Low", "Open", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                return None

        # -------- FEATURE ENGINEERING -------- #
        df["SMA_7"] = df["Close"].rolling(7).mean()
        df["SMA_21"] = df["Close"].rolling(21).mean()

        # Drop NaN values
        df = df.dropna()

        # Ensure enough data
        if len(df) < 30:
            return None

        return df

    except Exception as e:
        print(e)
        return None

# -------- LOAD DATA -------- #
df = load_data(ticker, period)

# -------- ERROR HANDLING -------- #
if df is None or df.empty:
    st.error("⚠️ Failed to fetch enough stock data.\n\nTry:\n- Different ticker (e.g., AAPL, MSFT)\n- Longer period (1y or 2y)")
    st.stop()

# -------- TITLE -------- #
st.title("📈 Stock Price Prediction Dashboard")

# -------- SHOW DATA -------- #
st.subheader("Latest Stock Data")
st.dataframe(df.tail())

# -------- PREPARE FEATURES -------- #
X = df[features].copy()

if len(X) == 0:
    st.error("❌ Not enough processed data for prediction")
    st.stop()

# -------- PREDICTION -------- #
predictions = model.predict(X)

df = df.loc[X.index]
df["Predicted"] = predictions

# -------- GRAPH -------- #
st.subheader("📊 Actual vs Predicted Price")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["Close"],
    name="Actual Price",
    mode='lines'
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["Predicted"],
    name="Predicted Price",
    mode='lines'
))

fig.update_layout(
    template="plotly_dark",
    xaxis_title="Date",
    yaxis_title="Price",
    legend_title="Legend"
)

st.plotly_chart(fig, use_container_width=True)

# -------- NEXT DAY PREDICTION -------- #
latest = X.tail(1)

try:
    next_price = model.predict(latest)[0]

    st.subheader("🔮 Next Day Prediction")
    st.metric("Predicted Next Day Price", f"${next_price:.2f}")

except:
    st.warning("Prediction failed. Try another stock.")

# -------- FOOTER -------- #
st.markdown("---")
st.markdown("⭐ Built with Streamlit | Machine Learning Project by Abhishek Shelke")
```
