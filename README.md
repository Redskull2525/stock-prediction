# 📈 Machine Learning Stock Price Prediction Dashboard

A **Machine Learning-based Stock Price Prediction Dashboard** built using **Python, Random Forest, Streamlit, and Plotly**.
This project fetches historical stock data, performs feature engineering using moving averages, and predicts the **next-day closing price** using a trained ML model.

The dashboard provides an **interactive visualization of actual vs predicted stock prices** and allows users to explore predictions for different stocks.

---

# 🚀 Live Demo

🔗 **Streamlit App**

https://stock-prediction-abhishek.streamlit.app/

---

# 🧠 Machine Learning Model

This project uses a **Random Forest Regressor** to predict the **next day's closing stock price**.

### Features Used for Training

| Feature | Description                  |
| ------- | ---------------------------- |
| Close   | Daily closing price          |
| High    | Highest price of the day     |
| Low     | Lowest price of the day      |
| Open    | Opening price                |
| Volume  | Total trading volume         |
| SMA_7   | 7-day Simple Moving Average  |
| SMA_21  | 21-day Simple Moving Average |

### Target Variable

Next-day closing price:

```
Target = Close.shift(-1)
```

---

# 📊 Machine Learning Workflow

```
Yahoo Finance Data
        ↓
Feature Engineering
(SMA Indicators)
        ↓
Train/Test Split
        ↓
Random Forest Model
        ↓
Model Evaluation
(MSE, R² Score)
        ↓
Model Saved (.pkl)
        ↓
Streamlit Dashboard
        ↓
Real-Time Prediction
```

---

# 📊 Dashboard Features

✔ Interactive **Stock Data Viewer**
✔ **Actual vs Predicted Price Graph**
✔ **Next-Day Stock Price Prediction**
✔ Sidebar with **Developer Information**
✔ Automatic **Yahoo Finance Data Fetching**
✔ **Interactive Plotly Visualization**

---

# 📈 Visualization

The dashboard compares **Actual vs Predicted prices** using an interactive Plotly chart.

```
Actual Price ─────────────
              ╲
               ╲
Predicted Price ╲─────────
```

Users can visually evaluate how well the model predicts stock price movements.

---

# 🖥️ Dashboard Preview

*(Add screenshots here for better GitHub presentation)*

```
images/dashboard.png
images/prediction_graph.png
```

Example:

```
![Dashboard](images/dashboard.png)

![Prediction Graph](images/prediction_graph.png)
```

---

# 📂 Project Structure

```
stock-prediction
│
├── app.py
├── aapl_stock_model.pkl
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

Clone the repository:

```
git clone https://github.com/Redskull2525/stock-prediction.git
cd stock-prediction
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

# 📦 Requirements

```
streamlit
yfinance
pandas
numpy
plotly
joblib
scikit-learn
```

---

# 📊 Example Model Performance

Example evaluation metrics:

```
Mean Squared Error (MSE): 4.21
R² Score: 0.91
```

*(Results may vary depending on market conditions.)*

---

# 🔮 Future Improvements

* Candlestick charts
* RSI & MACD indicators
* Multi-stock comparison
* 7-day price forecasting
* Deep learning models (LSTM)

---

# 👨‍💻 Developer

**Abhishek Shelke**

🎓 M.Sc Computer Science
ASM's CSIT, Pimpri

🔗 GitHub
https://github.com/Redskull2525

🔗 LinkedIn
https://www.linkedin.com/in/abhishek-s-b98895249

---

⭐ If you like this project, consider **starring the repository** on GitHub.
