"""Lags have to point where their names say they point."""
from __future__ import annotations

import numpy as np

from gridcast import baseline, features


def test_origin_lags_are_measured_from_the_origin(series):
    origins = np.array([1000, 2000, 3000])
    horizons = np.array([1, 48, 168])
    X, y, index = features.build(series, origins, horizons)
    load = series["load"].to_numpy()

    o = index["origin_pos"].to_numpy()
    t = index["target_pos"].to_numpy()
    np.testing.assert_allclose(X["olag_0"], load[o])
    np.testing.assert_allclose(X["olag_24"], load[o - 24])
    np.testing.assert_allclose(X["tlag_168"], load[t - 168])
    np.testing.assert_allclose(y, load[t])


def test_rolling_means_are_trailing_and_end_at_the_origin(series):
    origins = np.array([500, 900])
    X, _, index = features.build(series, origins, np.array([1]))
    load = series["load"].to_numpy()
    for row, o in enumerate(index["origin_pos"].to_numpy()):
        assert np.isclose(X["oroll_24"].iloc[row], load[o - 23:o + 1].mean())
        assert np.isclose(X["oroll_168"].iloc[row], load[o - 167:o + 1].mean())


def test_seasonal_naive_baseline_equals_the_week_ago_feature(series):
    """Same quantity computed twice, in two modules. If they ever disagree,
    one of them has an off-by-one."""
    origins = np.array([400, 800, 1200])
    horizons = np.array([1, 24, 168])
    X, _, index = features.build(series, origins, horizons)
    np.testing.assert_allclose(baseline.seasonal_naive(series, index),
                               X["tlag_168"].to_numpy())


def test_pairs_keep_their_row_order(series):
    o = np.array([600, 400, 600])
    h = np.array([168, 1, 3])
    X, y, index = features.build_pairs(series, o, h)
    np.testing.assert_array_equal(index["origin_pos"].to_numpy(), o)
    np.testing.assert_array_equal(index["h"].to_numpy(), h)
    load = series["load"].to_numpy()
    np.testing.assert_allclose(y, load[o + h])
