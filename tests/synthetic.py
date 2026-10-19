"""Small hand-made price tables for the tests (no network needed)."""

import numpy as np
import pandas as pd


def make_raw(closes, end="2026-10-16") -> pd.DataFrame:
    """OHLCV table with one row per weekday, ending on `end` (a Friday by default)."""
    closes = np.asarray(closes, dtype=float)
    dates = pd.bdate_range(end=end, periods=len(closes))
    return pd.DataFrame({
        "Date":   dates,
        "Open":   closes * 0.995,
        "High":   closes * 1.01,
        "Low":    closes * 0.99,
        "Close":  closes,
        "Volume": np.full(len(closes), 1_000_000.0),
    })


def random_walk(n=300, seed=0, start=100.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return start * np.cumprod(1 + rng.normal(0.0005, 0.01, n))


class IdentityScaler:
    def transform(self, X):
        return np.asarray(X, dtype=float)


class ConstantModel:
    """Always predicts the same return."""
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)
