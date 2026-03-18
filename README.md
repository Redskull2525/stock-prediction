# 📈 ML Stock Price Prediction Dashboard

A **Machine Learning-based Stock Price Prediction Dashboard** built using **Python, Streamlit, Plotly, and Twelve Data API**.

This project fetches **real-time and historical stock data**, performs advanced feature engineering using 40+ technical indicators, and predicts the **next-day closing price** using a trained ensemble ML model.

The dashboard provides **live price feeds**, **interactive visualizations**, **automatic daily retraining**, and **next-day price forecasts**.

---

## 🚀 Live Demo

🔗 **[stock-prediction-abhishek.streamlit.app](https://stock-prediction-abhishek.streamlit.app/)**

---

## 🧠 Machine Learning Model

This project uses a **VotingRegressor ensemble** combining **Random Forest** and **Gradient Boosting** to predict the **next-day return**, which is then converted to a price prediction.

> Predicting **returns** instead of raw prices eliminates scale bias — the model works correctly even as stock prices trend upward over years.

### Features Used for Training (40+)

| Category | Features |
|---|---|
| Price | Close, High, Low, Open |
| Moving Averages | SMA 5/7/10/21/50, EMA 5/7/10/21/50 |
| MA Ratios | Close/SMA ratios (scale-free) |
| Lag Returns | 1, 2, 3, 5, 10-day returns |
| Normalised Lags | Close lag / current close |
| Volume | Volume, Volume SMA 10, Volume ratio |
| Momentum | 5-day & 10-day momentum |
| Volatility | 10-day & 21-day rolling std |
| Candlestick | Daily range, Body, Upper/Lower wick |
| RSI | RSI-7, RSI-14 |
| MACD | MACD line, Signal, Histogram |
| Bollinger Bands | BB Width, BB %B |
| Rate of Change | ROC 5, 10, 21 |

### Target Variable

Next-day return (converted back to price):
```
Target = Close.pct_change().shift(-1)
Predicted_Price = Today_Close × (1 + Predicted_Return)
```

### Model Architecture

```
RandomForestRegressor(n_estimators=500, max_depth=10)
            +
GradientBoostingRegressor(n_estimators=300, lr=0.05)
            ↓
     VotingRegressor (ensemble)
```

---

## 📊 ML Workflow

```
Twelve Data API (Real-Time)
        ↓
Historical OHLCV (5 Years)
        ↓
Feature Engineering (40+ indicators)
        ↓
Train / Test Split (80/20)
        ↓
StandardScaler (normalisation)
        ↓
VotingRegressor (RF + GradientBoosting)
        ↓
Model Evaluation (RMSE, MAE, R²)
        ↓
Model Saved (.pkl)
        ↓
Streamlit Dashboard
        ↓
Live Price Feed + Next-Day Prediction
```

---

## 📊 Dashboard Features

✔ **Live Price Ticker** — real-time quote refreshing every 30 seconds
✔ **Actual vs Predicted Price Chart** — with live price overlay
✔ **Next-Day Stock Price Prediction** — with expected % move
✔ **Auto-Retraining** — model retrains automatically when stale (>24h) after market close
✔ **Manual Retrain Button** — retrain on demand from the sidebar
✔ **Model Status Panel** — shows last trained time, R², RMSE
✔ **Interactive Plotly Visualization** — zoom, pan, hover
✔ **Multi-Stock Support** — change ticker from sidebar
✔ **Developer Info Sidebar**

---

## 📂 Project Structure

```
stock-prediction/
│
├── app.py                  # Streamlit dashboard (fully self-contained)
├── train_model.py          # Standalone training script (local use)
├── aapl_stock_model.pkl    # Saved model + scaler + metadata
├── requirements.txt        # Python dependencies
└── README.md
```

---

## ⚙️ Installation

Clone the repository:
```bash
git clone https://github.com/Redskull2525/stock-prediction.git
cd stock-prediction
```

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🔑 API Key Setup

This project uses **[Twelve Data](https://twelvedata.com)** for real-time and historical stock data.

1. Sign up at [twelvedata.com](https://twelvedata.com) — free tier gives 800 requests/day
2. Copy your API key

**For local development** — create `.streamlit/secrets.toml`:
```toml
TWELVE_DATA_API_KEY = "your_api_key_here"
```

Add to `.gitignore`:
```
.streamlit/secrets.toml
```

**For Streamlit Cloud:**
```
App Settings → Secrets → paste:
TWELVE_DATA_API_KEY = "your_api_key_here"
```

---

## ▶️ Run the Application

Train the model first (only needed locally):
```bash
python train_model.py
```

Launch the dashboard:
```bash
streamlit run app.py
```

Open in browser:
```
http://localhost:8501
```

> On **Streamlit Cloud**, the app auto-retrains on first launch if no model is found.

---

## 📦 Requirements

```
streamlit
requests
pandas
numpy
plotly
joblib
scikit-learn
```

---

## 📊 Model Performance

Typical evaluation metrics on 5-year AAPL data:

```
RMSE    : ~2.10
MAE     : ~1.60
R² Score: ~0.97–0.99
```

*(Results vary with market conditions and retraining date.)*

---

## 🔄 Auto-Retraining Logic

| Condition | Behaviour |
|---|---|
| Model older than 24h + market closed | Auto-retrain on startup |
| "Retrain Now" button clicked | Immediate retrain |
| Model fresh (< 24h old) | Skip retraining, use cached model |

---

## 🔮 Future Improvements

- [ ] Candlestick charts
- [ ] 7-day multi-step forecasting
- [ ] Multi-stock comparison view
- [ ] Deep learning model (LSTM / Transformer)
- [ ] Portfolio-level prediction
- [ ] Sentiment analysis from news headlines
- [ ] Email/SMS alerts for large predicted moves

---

## 👨‍💻 Developer

**Abhishek Shelke**

🎓 M.Sc Computer Science — ASM's CSIT, Pimpri

🔗 [GitHub](https://github.com/Redskull2525) | 🔗 [LinkedIn](https://www.linkedin.com/in/abhishek-s-b98895249)

---

⭐ If you found this useful, consider **starring the repository** on GitHub!
