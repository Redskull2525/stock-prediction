import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")

# ---------------- SIDEBAR ---------------- #

st.sidebar.title("👨‍💻 Developer Info")

st.sidebar.markdown("""
**Name:** Abhishek Shelke  

**Program:** M.Sc Computer Science  
ASM's CSIT, Pimpri  

**Interest:**  
- Data Science  
- Machine Learning  
- Artificial Intelligence  

**GitHub:**  
https://github.com/Redskull2525  

**LinkedIn:**  
https://www.linkedin.com/in/abhishek-s-b98895249
""")

st.sidebar.markdown("---")

ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
period = st.sidebar.selectbox("Data Period", ["1y","2y","5y"])

# ---------------- TITLE ---------------- #

st.title("📈 Machine Learning Stock Prediction Dashboard")

# ---------------- DATA ---------------- #

@st.cache_data
def load_data(ticker, period):

    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True)

    df['SMA_7'] = df['Close'].rolling(7).mean()
    df['SMA_21'] = df['Close'].rolling(21).mean()

    df = df.dropna()

    return df

df = load_data(ticker, period)

st.subheader("Recent Stock Data")

st.dataframe(df.tail())

# ---------------- FEATURES ---------------- #

features = ['Close','High','Low','Open','Volume','SMA_7','SMA_21']

X = df[features]

# ---------------- LOAD MODEL ---------------- #

model = joblib.load("aapl_stock_model.pkl")

# ---------------- PREDICTION ---------------- #

predictions = model.predict(X)

df["Predicted"] = predictions

# ---------------- GRAPH ---------------- #

st.subheader("📊 Actual vs Predicted Price")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.index,
    y=df['Close'],
    name="Actual Price"
))

fig.add_trace(go.Scatter(
    x=df.index,
    y=df['Predicted'],
    name="Predicted Price"
))

st.plotly_chart(fig, use_container_width=True)

# ---------------- NEXT DAY PREDICTION ---------------- #

latest = df[features].iloc[-1:]

next_price = model.predict(latest)[0]

st.subheader("🔮 Tomorrow Prediction")

st.metric("Predicted Next Day Price", f"${next_price:.2f}")
