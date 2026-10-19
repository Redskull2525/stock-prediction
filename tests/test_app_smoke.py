"""Smoke tests: run app.py with Streamlit's AppTest and the network mocked out."""

import shutil
from pathlib import Path
from unittest import mock

import joblib
import pytest
import streamlit as st
from sklearn.linear_model import LinearRegression
from streamlit.testing.v1 import AppTest

import stock_core as core
from synthetic import make_raw, random_walk

REPO = Path(__file__).resolve().parents[1]
APP = str(REPO / "app.py")
RAW = make_raw(random_walk(300))                    # last bar: Friday 2026-10-16


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def fake_get(url, params=None, timeout=None):
    if url.endswith("/time_series"):                # Twelve Data: newest first, values as strings
        values = [{"datetime": r.Date.strftime("%Y-%m-%d"), "open": str(r.Open),
                   "high": str(r.High), "low": str(r.Low), "close": str(r.Close),
                   "volume": str(int(r.Volume))}
                  for r in RAW.iloc[::-1].itertuples()]
        return FakeResponse({"values": values})
    last = RAW.iloc[-1]
    return FakeResponse({"close": last.Close, "open": last.Open, "high": last.High,
                         "low": last.Low, "change": 0.5, "percent_change": 0.3,
                         "volume": 123456, "datetime": "2026-10-16"})


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    """Fresh model in a temp file, fake API key, mocked network, market 'closed'."""
    model_path = tmp_path / "model.pkl"
    joblib.dump(core.train_bundle(RAW, "AAPL", "test", model_factory=LinearRegression), model_path)
    monkeypatch.setenv("STOCK_MODEL_PATH", str(model_path))
    monkeypatch.setenv("TWELVE_DATA_API_KEY", "test-key")
    st.cache_data.clear()
    st.cache_resource.clear()
    # Market closed -> no 30 s sleep; fresh model -> no auto-retrain.
    with mock.patch("requests.get", side_effect=fake_get) as get, \
         mock.patch.object(core, "market_is_closed", return_value=True):
        yield {"model_path": model_path, "get": get}


def run_app():
    return AppTest.from_file(APP, default_timeout=120).run()


def labels(at):
    return [m.label for m in at.metric]


def test_app_runs_and_shows_baseline(app_env):
    at = run_app()
    assert not at.exception
    assert not at.error, [e.value for e in at.error]
    assert "Naive RMSE (tomorrow = today)" in labels(at)
    assert "Last Close (2026-10-16)" in labels(at)               # newest bar, not the one before
    assert "Predicted Close (Mon 2026-10-19)" in labels(at)      # next trading day


def test_app_refuses_model_for_other_ticker(app_env):
    at = run_app()
    at.sidebar.text_input[0].set_value("MSFT").run()
    assert not at.exception
    assert any("trained on **AAPL**" in w.value for w in at.warning)
    assert "Predicted Close (Mon 2026-10-19)" not in labels(at)


def test_app_loads_legacy_model_file(app_env):
    legacy = REPO / "aapl_stock_model.pkl"
    if not legacy.exists():
        pytest.skip("aapl_stock_model.pkl not present")
    shutil.copy(legacy, app_env["model_path"])               # copy -> fresh mtime, not stale
    at = run_app()
    assert not at.exception
    assert not at.error, [e.value for e in at.error]
    assert any("old format" in i.value for i in at.sidebar.info)
    assert "Predicted Close (Mon 2026-10-19)" in labels(at)


def test_app_without_api_key_shows_friendly_error(app_env, monkeypatch):
    monkeypatch.delenv("TWELVE_DATA_API_KEY")
    at = run_app()
    assert not at.exception
    assert any("TWELVE_DATA_API_KEY is not set" in e.value for e in at.error)
