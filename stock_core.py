"""
stock_core.py  —  the pure logic behind app.py.

Nothing in here talks to Streamlit or the network, so every function can be
tested with a small hand-made price table (see tests/).

The one rule that keeps the dates straight
------------------------------------------
Row t of the feature table holds:
    * features built only from days <= t
    * Target_return = Close(t+1) / Close(t) - 1   (tomorrow's % move)
    * Target_price  = Close(t+1)                  (tomorrow's close)
    * Target_date   = Date(t+1)                   (the day the target is FOR)

So the model's output for row t, pred(t), is a forecast for Target_date(t),
and the matching price forecast is

    Predicted(t+1) = Close(t) * (1 + pred(t))

Every prediction is stored under the date it is FOR (Target_date), which is
the same date as the actual price it should be compared with.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

BUNDLE_VERSION = 2          # version 1 = the bundles saved before the Oct 2026 fixes
TEST_FRACTION = 0.2         # last 20% of days are the test set
TARGET_COLUMNS = ["Target_return", "Target_price", "Target_date"]
NON_FEATURE_COLUMNS = {"Date", *TARGET_COLUMNS}


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURES AND TARGETS
# ══════════════════════════════════════════════════════════════════════════════
def _rsi(s, p=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - (100 / (1 + g / (l + 1e-9)))

def _macd(s, fast=12, slow=26, sig=9):
    m = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    sg = m.ewm(span=sig, adjust=False).mean()
    return m, sg, m - sg

def _bb(s, p=20):
    mid = s.rolling(p).mean(); std = s.rolling(p).std()
    u, l = mid + 2*std, mid - 2*std
    return (u-l)/(mid+1e-9), (s-l)/(u-l+1e-9)


def add_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Add the 48 indicator columns. Only past rows are used (rolling / ewm / shift(+n)).

    No rows are dropped here: the first ~50 rows have NaN features (not enough
    history yet) and are filtered out later, where it is clear why.
    """
    df = raw.copy()
    for w in [5, 7, 10, 21, 50]:
        df[f'SMA_{w}'] = df['Close'].rolling(w).mean()
        df[f'EMA_{w}'] = df['Close'].ewm(span=w, adjust=False).mean()
    for w in [5, 10, 21, 50]:
        df[f'Close_SMA{w}_ratio'] = df['Close'] / (df[f'SMA_{w}'] + 1e-9)
    for lag in [1, 2, 3, 5, 10]:
        df[f'Return_lag_{lag}'] = df['Close'].pct_change(lag)
    for lag in [1, 2, 3, 5]:
        df[f'Close_lag_{lag}_norm'] = df['Close'].shift(lag) / df['Close']
    df['Volume_SMA_10'] = df['Volume'].rolling(10).mean()
    df['Volume_ratio']  = df['Volume'] / (df['Volume_SMA_10'] + 1e-9)
    df['Momentum_5']    = df['Close'].pct_change(5)
    df['Momentum_10']   = df['Close'].pct_change(10)
    df['Volatility_10'] = df['Close'].pct_change().rolling(10).std()
    df['Volatility_21'] = df['Close'].pct_change().rolling(21).std()
    df['Daily_Range']   = (df['High'] - df['Low']) / df['Close']
    df['Body']          = abs(df['Close'] - df['Open']) / df['Close']
    df['Upper_Wick']    = (df['High'] - df[['Close', 'Open']].max(axis=1)) / df['Close']
    df['Lower_Wick']    = (df[['Close', 'Open']].min(axis=1) - df['Low'])  / df['Close']
    df['RSI_14'] = _rsi(df['Close'], 14)
    df['RSI_7']  = _rsi(df['Close'], 7)
    df['MACD'], df['MACD_signal'], df['MACD_hist'] = _macd(df['Close'])
    for c in ['MACD', 'MACD_signal', 'MACD_hist']:
        df[c] /= df['Close']
    df['BB_width'], df['BB_pct'] = _bb(df['Close'])
    for p in [5, 10, 21]:
        df[f'ROC_{p}'] = df['Close'].pct_change(p) * 100
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """shift(-1) pulls TOMORROW's values up onto today's row.

    The newest row gets NaN targets (tomorrow hasn't happened yet). That row is
    still kept: it is exactly the row we need for the next-day forecast.
    """
    df = df.copy()
    df['Target_return'] = df['Close'].pct_change().shift(-1)
    df['Target_price']  = df['Close'].shift(-1)
    df['Target_date']   = df['Date'].shift(-1)
    return df


def build_feature_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Features + targets for every raw row. Nothing is dropped."""
    return add_targets(add_features(raw)).reset_index(drop=True)


def feature_columns(table: pd.DataFrame) -> list:
    """Every column except the date and the target columns (no future data)."""
    return [c for c in table.columns if c not in NON_FEATURE_COLUMNS]


def training_rows(table: pd.DataFrame, features: list) -> pd.DataFrame:
    """Rows usable for training/evaluation: complete features AND a known target."""
    needed = list(features) + ['Target_return', 'Target_price']
    return table.dropna(subset=needed).reset_index(drop=True)


def latest_feature_row(table: pd.DataFrame, features: list) -> pd.DataFrame:
    """The newest row whose FEATURES are complete. Its target may be NaN (that's the point)."""
    complete = table.dropna(subset=list(features))
    if complete.empty:
        raise ValueError("Not enough history to build features (need ~50 trading days).")
    return complete.iloc[[-1]]


def next_trading_day(date) -> pd.Timestamp:
    """Next weekday after `date`. Exchange holidays are not known here."""
    return (pd.Timestamp(date) + pd.offsets.BDay(1)).normalize()


# ══════════════════════════════════════════════════════════════════════════════
#  SPLITTING
# ══════════════════════════════════════════════════════════════════════════════
def chronological_split(rows: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    """First (1 - test_fraction) of the days train, the rest test. Never shuffled."""
    split = int(len(rows) * (1 - test_fraction))
    return rows.iloc[:split], rows.iloc[split:]


def evaluation_window(rows: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Rows the saved model did NOT train on.

    New bundles store the first test date, so the window stays correct even
    after new days arrive. Old bundles don't, so we fall back to re-splitting.
    """
    test_start = bundle.get("test_start_date")
    if test_start:
        return rows[rows['Date'] >= pd.Timestamp(test_start)]
    return chronological_split(rows)[1]


# ══════════════════════════════════════════════════════════════════════════════
#  RETURN  ->  PRICE
# ══════════════════════════════════════════════════════════════════════════════
def predictions_frame(rows: pd.DataFrame, pred_returns) -> pd.DataFrame:
    """Turn predicted returns into prices, indexed by the date each prediction is FOR.

    For row t:  Predicted = Close(t) * (1 + pred(t)),  Actual = Close(t+1),
    and both sit under Target_date = Date(t+1), so they line up by date.
    """
    pred_returns = np.asarray(pred_returns, dtype=float)
    base_close = rows['Close'].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "Base_date":        rows['Date'].to_numpy(),
            "Base_close":       base_close,
            "Predicted_return": pred_returns,
            "Predicted":        base_close * (1 + pred_returns),
            "Actual":           rows['Target_price'].to_numpy(dtype=float),
        },
        index=pd.DatetimeIndex(rows['Target_date'], name="Date"),
    )


def predict_frame(model, scaler, rows: pd.DataFrame, features: list) -> pd.DataFrame:
    """Run the model on `rows` and return predictions_frame(...)."""
    pred_returns = model.predict(scaler.transform(rows[features]))
    return predictions_frame(rows, pred_returns)


def next_day_forecast(model, scaler, table: pd.DataFrame, features: list) -> dict:
    """Forecast the close for the trading day AFTER the newest complete row."""
    row = latest_feature_row(table, features)
    pred_return = float(model.predict(scaler.transform(row[features]))[0])
    base_close = float(row['Close'].iloc[0])
    base_date = pd.Timestamp(row['Date'].iloc[0])
    return {
        "base_date":        base_date,
        "base_close":       base_close,
        "target_date":      next_trading_day(base_date),
        "predicted_return": pred_return,
        "predicted_close":  base_close * (1 + pred_return),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS (model vs the naive "tomorrow = today" baseline)
# ══════════════════════════════════════════════════════════════════════════════
def _rmse(a, b) -> float:
    return float(np.sqrt(mean_squared_error(a, b)))


def evaluate_predictions(frame: pd.DataFrame) -> dict:
    """Metrics on rows of predictions_frame() that have a known Actual.

    * rmse / mae / r2         : model's price forecast vs actual price
    * baseline_rmse / _mae    : naive forecast "tomorrow's close = today's close"
    * skill                   : 1 - rmse / baseline_rmse  (> 0 means we beat naive)
    * directional_accuracy    : share of days where (predicted move > 0) == (actual move > 0)
    * up_day_share            : share of up days = accuracy of always guessing "up".
                                The naive forecast predicts "no change", so it has no
                                direction; "always up" is the honest direction baseline.
    """
    f = frame.dropna(subset=["Actual"])
    if len(f) < 2:
        raise ValueError("Need at least 2 evaluated days to compute metrics.")
    actual, predicted, base = f["Actual"], f["Predicted"], f["Base_close"]
    actual_up = (actual / base - 1) > 0
    rmse, baseline_rmse = _rmse(actual, predicted), _rmse(actual, base)
    return {
        "rmse":                 rmse,
        "mae":                  float(np.mean(np.abs(actual - predicted))),
        "r2":                   float(r2_score(actual, predicted)),
        "baseline_rmse":        baseline_rmse,
        "baseline_mae":         float(np.mean(np.abs(actual - base))),
        "skill":                float(1 - rmse / baseline_rmse) if baseline_rmse > 0 else float("nan"),
        "directional_accuracy": float(np.mean((f["Predicted_return"] > 0) == actual_up)),
        "up_day_share":         float(np.mean(actual_up)),
        "n_test_days":          int(len(f)),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING AND THE SAVED BUNDLE
# ══════════════════════════════════════════════════════════════════════════════
def make_model():
    rf = RandomForestRegressor(n_estimators=500, max_depth=10,
             min_samples_split=5, min_samples_leaf=2,
             max_features=0.7, random_state=42, n_jobs=-1)
    gb = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
             max_depth=5, subsample=0.8, random_state=42)
    return VotingRegressor([('rf', rf), ('gb', gb)])


def _iso(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


def train_bundle(raw: pd.DataFrame, ticker: str, trained_at: str,
                 model_factory=make_model, test_fraction: float = TEST_FRACTION) -> dict:
    """Build features, split by time, fit, evaluate, and return everything to save."""
    table = build_feature_table(raw)
    features = feature_columns(table)
    rows = training_rows(table, features)
    train_rows, test_rows = chronological_split(rows, test_fraction)
    if len(train_rows) == 0 or len(test_rows) < 2:
        raise ValueError(f"Not enough data to train ({len(rows)} usable rows).")

    scaler = StandardScaler()                      # fitted on TRAIN rows only
    X_train = scaler.fit_transform(train_rows[features])
    model = model_factory()
    model.fit(X_train, train_rows['Target_return'])

    metrics = evaluate_predictions(predict_frame(model, scaler, test_rows, features))
    return {
        "bundle_version":   BUNDLE_VERSION,
        "model":            model,
        "scaler":           scaler,
        "features":         features,
        "ticker":           ticker.upper(),
        "trained_at":       trained_at,
        "train_start_date": _iso(train_rows['Date'].iloc[0]),
        "train_end_date":   _iso(train_rows['Date'].iloc[-1]),
        "test_start_date":  _iso(test_rows['Date'].iloc[0]),
        "test_end_date":    _iso(test_rows['Date'].iloc[-1]),
        **metrics,
    }


def is_legacy_bundle(bundle: dict) -> bool:
    """True for bundles saved before the Oct 2026 fixes (their metrics used the B1 bug)."""
    return bundle.get("bundle_version", 1) < BUNDLE_VERSION


def ticker_mismatch(bundle: dict, ticker: str) -> bool:
    """True if the bundle was trained on a different ticker (or doesn't say which)."""
    saved = bundle.get("ticker")
    return not saved or str(saved).upper() != ticker.strip().upper()


def market_is_closed(now) -> bool:
    """Weekend or after 4 pm (pass `now` in US/Eastern). Holidays/pre-market not handled."""
    return now.weekday() >= 5 or now.hour >= 16
