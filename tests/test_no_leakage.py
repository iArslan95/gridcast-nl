"""The test this repo exists for.

A rolling-origin backtest is easy to write and easy to get subtly wrong: one
rolling mean that is centred instead of trailing, one lag off by an hour, and
the score is excellent and meaningless. These check the property directly
rather than checking that the code looks right.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridcast import backtest, features


def test_features_ignore_everything_after_the_origin(series):
    """Corrupt the future, rebuild, and demand identical features.

    If any feature reads past its own origin, replacing every value after the
    origin with nonsense has to change something. Nothing may change.
    """
    origins = np.arange(features.MIN_ORIGIN_POS, len(series) - 200, 97)
    horizons = np.array([1, 24, 72, 168])
    X, _, index = features.build(series, origins, horizons)

    corrupt = series.copy()
    load = corrupt["load"].to_numpy().copy()
    cut = int(origins.min())
    load[cut + 1:] = -1e6                      # obvious nonsense
    corrupt["load"] = load

    X2, _, _ = features.build(corrupt, origins, horizons)
    same = index["origin_pos"].to_numpy() == cut
    pd.testing.assert_frame_equal(X[same], X2[same])


def test_target_lag_would_leak_beyond_one_week():
    """tlag_168 sits at origin + h - 168, which is only in the past while
    h <= 168. The guard is what keeps that from silently breaking if someone
    raises the horizon."""
    assert max(features.TARGET_LAGS) >= features.MAX_HORIZON
    assert features.MAX_HORIZON == 168


def test_horizons_outside_the_safe_range_are_refused(series):
    o = np.array([features.MIN_ORIGIN_POS + 10])
    with pytest.raises(ValueError):
        features.build(series, o, np.array([features.MAX_HORIZON + 1]))
    with pytest.raises(ValueError):
        features.build(series, o, np.array([0]))
    with pytest.raises(ValueError):
        features.build(series, np.array([features.MIN_ORIGIN_POS - 1]),
                       np.array([1]))


def test_every_fold_trains_only_on_the_past(series):
    folds = backtest.make_folds(series, "2019-09-01", refit_days=30)
    assert len(folds) >= 2
    for fold in folds:
        backtest.assert_no_leakage(series, fold)


def test_training_origins_stop_a_full_horizon_before_the_cut(series):
    folds = backtest.make_folds(series, "2019-09-01", refit_days=30)
    cut = folds[0].cut_pos
    origins = backtest.training_origins(cut)
    assert origins.max() + features.MAX_HORIZON <= cut
