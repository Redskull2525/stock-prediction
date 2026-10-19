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
Target(t)          = Close(t+1) / Close(t) − 1        # Close.pct_change().shift(-1)
Predicted(t+1)     = Close(t) × (1 + Predicted_Return(t))
```
Every prediction is stored under the date it is **for** (t+1), so the actual and predicted prices line up by date.

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
✔ **Model Status Panel** — trained ticker and date ranges, RMSE vs the naive "tomorrow = today" baseline, directional accuracy vs the up-day share
✔ **Interactive Plotly Visualization** — zoom, pan, hover
✔ **Multi-Stock Support** — change ticker from sidebar
✔ **Developer Info Sidebar**

---

## 📂 Project Structure

```
stock-prediction/
│
├── app.py                  # Streamlit dashboard (UI, API calls, saving the model)
├── stock_core.py           # Pure logic: features, targets, split, return→price, metrics
├── aapl_stock_model.pkl    # Saved model + scaler + metadata
├── requirements.txt        # Pinned app dependencies
├── requirements-dev.txt    # + pytest
├── tests/                  # pytest unit tests + Streamlit AppTest smoke tests
├── .github/workflows/tests.yml  # runs pytest on every push
├── docs/INTERVIEW_GUIDE.md
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

Launch the dashboard:
```bash
streamlit run app.py
```

Open in browser:
```
http://localhost:8501
```

To train (or retrain for another ticker), click **🔄 Retrain Now** in the sidebar. The app also auto-retrains when the saved model is missing, unreadable or more than 24 h old and the market is closed.

**Deploying on Streamlit Cloud:** main file `app.py`, Python 3.13, and one secret, `TWELVE_DATA_API_KEY`. Without the secret the app shows a clear error message instead of crashing.

---

## 📦 Requirements

`requirements.txt` pins the versions the app and tests were run with (Python 3.13): streamlit, requests, pandas, numpy, joblib, scikit-learn, plotly. scikit-learn stays at **1.6.1** because `aapl_stock_model.pkl` was saved with it and newer versions can't load that file. If you upgrade scikit-learn, click Retrain once to rebuild the model file.

---

## 🧪 Running the tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests need no API key: the unit tests use small hand-made price series, and the smoke tests run `app.py` with Streamlit's `AppTest` and the Twelve Data calls mocked. GitHub Actions runs them on every push (`.github/workflows/tests.yml`).

---

## 📊 Model Performance

The model predicts the **next-day return**. The app turns that return back into a price and reports RMSE and R² on that price. Both use a chronological 80/20 split, so the model never sees future data during training.

**How to read these numbers:** each day's price is very close to the previous day's. A naive "tomorrow = today" baseline already scores an R² of around 0.99 on price, so a high R² here does **not** mean the model predicts the market well. The fairer tests are:
- whether the model beats that naive baseline on RMSE
- how often it predicts the **direction** of the move (up or down) correctly

Both are now shown next to the model's own numbers: the Model Status panel and the Model Performance section list the model's RMSE beside the naive RMSE, a skill score (`1 − RMSE / naive RMSE`, above 0 means the model beats naive), and the model's directional accuracy beside the up-day share (the accuracy of always guessing "up"). The values change with each retrain, so none are quoted here. This project is a learning exercise, not trading advice.

---

## 🐞 Bug fixes & validation (Oct 2026)

A review of the code (documented in [`docs/INTERVIEW_GUIDE.md`](docs/INTERVIEW_GUIDE.md), section 7) found date-alignment bugs. Each one was checked against the old code, fixed, and covered by a test. The pure logic now lives in `stock_core.py`, which has no Streamlit or network code, so it can be tested with small hand-made price series.

| Bug | Root cause (old code) | Fix | Test |
|---|---|---|---|
| **B1** Off-by-one base price in the training metrics | `prev_closes = df['Close'].iloc[split-1:-1]` used Close(t−1), but `pred(t)` forecasts Close(t) → Close(t+1) | `predictions_frame()` uses each row's own Close(t): `Predicted = Close(t) × (1 + pred(t))` (`stock_core.py:159-176`) | `test_reconstruction_uses_todays_close` |
| **B2** Chart's predicted line misaligned by a day | `Predicted(t) = Close(t−1) × (1 + pred(t))` was plotted against Close(t), on an x-axis of row numbers | Each prediction is indexed by `Target_date` = Date(t+1), the day it is for; the chart plots both lines against real dates (`stock_core.py:102`, `stock_core.py:175`, `app.py:236-252`) | `test_prediction_frame_is_indexed_by_the_date_it_is_for`, `test_rows_with_missing_features_do_not_shift_the_alignment` |
| **B3** "Next-day" forecast used the second-newest day | `dropna()` dropped the newest row because its target is NaN, so `tail(1)` was yesterday | The feature table keeps every row; `latest_feature_row()` picks the newest row with complete features (target may be NaN) and the forecast is labelled with the next trading day (`stock_core.py:122-135`, `stock_core.py:185-198`) | `test_forecast_uses_the_newest_row`, `test_forecast_skips_a_newest_row_with_incomplete_features`, `test_next_trading_day` |
| **B4** Saved model could score a different ticker | The bundle stored `ticker` but nothing checked it | `ticker_mismatch()` (`stock_core.py:289-292`); the dashboard refuses and asks you to retrain (`app.py:166-168`, `app.py:221-225`) | `test_ticker_mismatch`, `test_app_refuses_model_for_other_ticker` |
| **B5** Dashboard test window drifted after new data | The dashboard re-split `int(len(df)*0.8)` on fresh data | The bundle stores the train/test date ranges; `evaluation_window()` uses `test_start_date` (`stock_core.py:144-153`, `stock_core.py:253-281`) | `test_evaluation_window_stays_fixed_when_new_days_arrive`, `test_chronological_split_has_no_overlap` |
| Honest evaluation | Only RMSE/R² on price | `evaluate_predictions()` adds naive RMSE/MAE, skill, directional accuracy, up-day share (`stock_core.py:207-234`) | `test_metrics_against_naive_baseline`, `test_perfect_model_beats_naive` |

**Saved model file.** New bundles (`bundle_version` 2) also store the ticker, train/test date ranges, feature list and the baseline metrics. The existing `aapl_stock_model.pkl` (old format) still loads and predicts; the app labels it "old format" and hides its saved metrics, which were computed with bug B1. Clicking Retrain replaces it with a new-format bundle. If the file can't be loaded at all (for example after a scikit-learn upgrade), the app treats it as missing and retrains (`test_existing_model_file_is_compatible`, `test_app_loads_legacy_model_file`).

**Smoke tests.** `tests/test_app_smoke.py` runs the whole app with the network mocked, checks the newest bar and next trading day appear in the forecast labels, and checks that a missing API key shows the friendly error (`test_app_without_api_key_shows_friendly_error`).

No performance numbers are quoted here: they depend on the data at retrain time, and these fixes were validated with synthetic data, not real market data.

---

## 🎓 Interview guide

[`docs/INTERVIEW_GUIDE.md`](docs/INTERVIEW_GUIDE.md) is a beginner-friendly walkthrough of this project: feature engineering, leakage-safe evaluation, why price R² is near-trivial, known bugs, likely interview questions and hands-on exercises.

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
