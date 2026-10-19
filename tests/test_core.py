"""Unit tests for stock_core.py using small series where the right answer is known."""

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

import stock_core as core
from synthetic import ConstantModel, IdentityScaler, make_raw, random_walk

REPO = Path(__file__).resolve().parents[1]


# ── Targets ──────────────────────────────────────────────────────────────────
def test_targets_are_tomorrows_values():
    raw = make_raw([100, 101, 103, 102])
    t = core.add_targets(raw)

    np.testing.assert_allclose(t["Target_return"].iloc[:3], [0.01, 2 / 101, -1 / 103])
    np.testing.assert_allclose(t["Target_price"].iloc[:3], [101, 103, 102])
    assert list(t["Target_date"].iloc[:3]) == list(raw["Date"].iloc[1:])
    # The newest row has no tomorrow yet, but it is kept.
    assert len(t) == 4
    assert t[["Target_return", "Target_price", "Target_date"]].iloc[-1].isna().all()


def test_targets_are_not_features():
    table = core.build_feature_table(make_raw(random_walk(80)))
    features = core.feature_columns(table)
    assert len(features) == 48
    assert not set(features) & {"Date", "Target_return", "Target_price", "Target_date"}


# ── Return -> price (bug B1) ─────────────────────────────────────────────────
def test_reconstruction_uses_todays_close():
    raw = make_raw([100, 101, 103, 102, 104])
    rows = core.add_targets(raw).iloc[:-1]           # rows with a known target
    perfect = rows["Target_return"].to_numpy()       # a model that is always right

    frame = core.predictions_frame(rows, perfect)

    # Predicted(t+1) = Close(t) * (1 + pred(t)) equals the real Close(t+1).
    np.testing.assert_allclose(frame["Predicted"], frame["Actual"])
    np.testing.assert_allclose(frame["Base_close"], [100, 101, 103, 102])

    # The old code used Close(t-1) as the base. Even a perfect model looks wrong then.
    old_base = raw["Close"].shift(1).iloc[:-1].to_numpy()
    old_pred = old_base[1:] * (1 + perfect[1:])
    assert not np.allclose(old_pred, frame["Actual"].iloc[1:])


# ── Chart dataframe (bug B2) ─────────────────────────────────────────────────
def test_prediction_frame_is_indexed_by_the_date_it_is_for():
    raw = make_raw([100, 101, 103, 102, 104])
    rows = core.add_targets(raw).iloc[:-1]
    frame = core.predictions_frame(rows, np.full(len(rows), 0.01))

    close_by_date = raw.set_index("Date")["Close"]
    assert list(frame.index) == list(raw["Date"].iloc[1:])
    # On every date, "Actual" is that date's real close ...
    np.testing.assert_allclose(frame["Actual"], close_by_date.loc[frame.index])
    # ... and "Predicted" for that date was built from the PREVIOUS day's close.
    assert (frame["Base_date"].to_numpy() < frame.index.to_numpy()).all()
    assert frame.loc[raw["Date"].iloc[2], "Predicted"] == pytest.approx(101 * 1.01)


def test_rows_with_missing_features_do_not_shift_the_alignment():
    raw = make_raw(random_walk(120))
    raw.loc[90, "Volume"] = np.nan                    # one bad row in the middle
    table = core.build_feature_table(raw)
    features = core.feature_columns(table)
    rows = core.training_rows(table, features)
    frame = core.predictions_frame(rows, np.zeros(len(rows)))

    # A zero-return prediction for date D must equal the close of the day before D.
    prev_close = raw.set_index("Date")["Close"].shift(1)
    np.testing.assert_allclose(frame["Predicted"], prev_close.loc[frame.index])


# ── Next-day forecast (bug B3) ───────────────────────────────────────────────
def test_forecast_uses_the_newest_row():
    raw = make_raw(100 * 1.01 ** np.arange(80))       # ends Friday 2026-10-16
    table = core.build_feature_table(raw)
    features = core.feature_columns(table)

    # Training rows stop one day early (the newest row has no target) ...
    assert core.training_rows(table, features)["Date"].iloc[-1] == raw["Date"].iloc[-2]
    # ... but the forecast row is the newest day.
    assert core.latest_feature_row(table, features)["Date"].iloc[0] == raw["Date"].iloc[-1]

    fc = core.next_day_forecast(ConstantModel(0.02), IdentityScaler(), table, features)
    assert fc["base_date"] == raw["Date"].iloc[-1]
    assert fc["base_close"] == pytest.approx(raw["Close"].iloc[-1])
    assert fc["predicted_close"] == pytest.approx(raw["Close"].iloc[-1] * 1.02)
    assert fc["target_date"] == pd.Timestamp("2026-10-19")   # Friday -> Monday


def test_forecast_skips_a_newest_row_with_incomplete_features():
    raw = make_raw(random_walk(80))
    raw.loc[79, "Volume"] = np.nan
    table = core.build_feature_table(raw)
    row = core.latest_feature_row(table, core.feature_columns(table))
    assert row["Date"].iloc[0] == raw["Date"].iloc[78]


def test_next_trading_day():
    assert core.next_trading_day("2026-10-14") == pd.Timestamp("2026-10-15")  # Wed -> Thu
    assert core.next_trading_day("2026-10-16") == pd.Timestamp("2026-10-19")  # Fri -> Mon


# ── Ticker check (bug B4) ────────────────────────────────────────────────────
def test_ticker_mismatch():
    assert not core.ticker_mismatch({"ticker": "AAPL"}, "aapl")
    assert not core.ticker_mismatch({"ticker": "AAPL"}, " AAPL ")
    assert core.ticker_mismatch({"ticker": "AAPL"}, "MSFT")
    assert core.ticker_mismatch({}, "AAPL")             # unknown ticker -> refuse


# ── Chronological split and the saved bundle (B5) ────────────────────────────
def test_chronological_split_has_no_overlap():
    rows = core.add_targets(make_raw(random_walk(100))).iloc[:-1]
    train, test = core.chronological_split(rows)
    assert len(train) == 79 and len(test) == 20
    assert train["Date"].max() < test["Date"].min()
    assert not set(train["Date"]) & set(test["Date"])
    pd.testing.assert_frame_equal(pd.concat([train, test]), rows)


def test_train_bundle_records_ticker_dates_and_metrics():
    raw = make_raw(random_walk(300))
    bundle = core.train_bundle(raw, "aapl", trained_at="now", model_factory=LinearRegression)

    assert bundle["ticker"] == "AAPL"
    assert bundle["bundle_version"] == core.BUNDLE_VERSION
    assert not core.is_legacy_bundle(bundle)
    assert bundle["features"] == core.feature_columns(core.build_feature_table(raw))
    assert bundle["train_end_date"] < bundle["test_start_date"]
    assert bundle["test_end_date"] == raw["Date"].iloc[-2].strftime("%Y-%m-%d")
    for key in ["rmse", "baseline_rmse", "skill", "directional_accuracy", "up_day_share"]:
        assert np.isfinite(bundle[key])


def test_evaluation_window_stays_fixed_when_new_days_arrive():
    raw = make_raw(random_walk(300))
    bundle = core.train_bundle(raw, "AAPL", trained_at="now", model_factory=LinearRegression)

    longer = make_raw(random_walk(330), end="2026-11-27")   # 30 more days of data
    rows = core.training_rows(core.build_feature_table(longer), bundle["features"])
    window = core.evaluation_window(rows, bundle)
    assert window["Date"].min() == pd.Timestamp(bundle["test_start_date"])
    # Re-splitting 80/20 (what the old dashboard did) would start the window later,
    # so some of the model's real test days would be missing from the chart.
    assert core.chronological_split(rows)[1]["Date"].min() > window["Date"].min()


# ── Metrics vs the naive baseline ────────────────────────────────────────────
def test_metrics_against_naive_baseline():
    frame = pd.DataFrame({
        "Base_close":       [100.0, 100.0, 100.0, 100.0],
        "Actual":           [101.0,  99.0, 102.0, 100.0],
        "Predicted_return": [0.01,   0.01, -0.01,  0.0],
    })
    frame["Predicted"] = frame["Base_close"] * (1 + frame["Predicted_return"])
    m = core.evaluate_predictions(frame)

    # Model errors 0, 2, -3, 0 ; naive ("tomorrow = today") errors 1, -1, 2, 0
    assert m["rmse"] == pytest.approx(np.sqrt(13 / 4))
    assert m["mae"] == pytest.approx(1.25)
    assert m["baseline_rmse"] == pytest.approx(np.sqrt(6 / 4))
    assert m["baseline_mae"] == pytest.approx(1.0)
    assert m["skill"] == pytest.approx(1 - np.sqrt(13 / 4) / np.sqrt(6 / 4))
    # Up days: T F T F (flat is not up). Predicted up: T T F F. Right on days 1 and 4.
    assert m["directional_accuracy"] == pytest.approx(0.5)
    assert m["up_day_share"] == pytest.approx(0.5)
    assert m["n_test_days"] == 4


def test_perfect_model_beats_naive():
    rows = core.add_targets(make_raw(random_walk(50))).iloc[:-1]
    m = core.evaluate_predictions(core.predictions_frame(rows, rows["Target_return"]))
    assert m["rmse"] == pytest.approx(0, abs=1e-9)
    assert m["skill"] == pytest.approx(1.0)
    assert m["directional_accuracy"] == 1.0


# ── Old model file still works ───────────────────────────────────────────────
def test_existing_model_file_is_compatible():
    path = REPO / "aapl_stock_model.pkl"
    if not path.exists():
        pytest.skip("aapl_stock_model.pkl not present")
    bundle = joblib.load(path)
    assert core.is_legacy_bundle(bundle)
    assert bundle["ticker"] == "AAPL"

    table = core.build_feature_table(make_raw(random_walk(120)))
    assert bundle["features"] == core.feature_columns(table)
    fc = core.next_day_forecast(bundle["model"], bundle["scaler"], table, bundle["features"])
    assert np.isfinite(fc["predicted_close"])


def test_market_is_closed():
    assert core.market_is_closed(datetime(2026, 10, 17, 12))      # Saturday
    assert core.market_is_closed(datetime(2026, 10, 16, 16, 5))   # Friday after close
    assert not core.market_is_closed(datetime(2026, 10, 16, 11))  # Friday midday
