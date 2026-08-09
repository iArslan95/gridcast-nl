"""A small synthetic series so the tests run in a second and depend on no
download. Shape does not matter here; only the index arithmetic does."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


@pytest.fixture(scope="session")
def series() -> pd.DataFrame:
    n = 24 * 400
    idx = pd.date_range("2019-01-01", periods=n, freq="H", tz="UTC")
    local = idx.tz_convert("Europe/Amsterdam")
    hour = np.asarray(local.hour)
    dow = np.asarray(local.dayofweek)
    rng = np.random.default_rng(7)
    load = (12000
            + 1500 * np.sin(2 * np.pi * hour / 24)
            - 900 * (dow >= 5)
            + 800 * np.sin(2 * np.pi * np.arange(n) / 8766)
            + rng.normal(0, 120, n))
    return pd.DataFrame({
        "load": load,
        "hour": hour,
        "dow": dow,
        "month": np.asarray(local.month),
        "doy": np.asarray(local.dayofyear),
        "is_weekend": dow >= 5,
        "is_holiday": np.zeros(n, dtype=bool),
        "is_yearend": np.zeros(n, dtype=bool),
    }, index=idx)
