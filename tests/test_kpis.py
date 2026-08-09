"""Metrics checked against a hand-worked example, not against themselves."""
from __future__ import annotations

import numpy as np

from gridcast import kpis

Y = np.array([100.0, 200.0, 300.0, 400.0])
YHAT = np.array([110.0, 180.0, 300.0, 440.0])
# errors: +10, -20, 0, +40  ->  |e| = 10, 20, 0, 40


def test_score_matches_hand_computation():
    s = kpis.score(Y, YHAT)
    assert s["mae"] == (10 + 20 + 0 + 40) / 4                       # 17.5
    assert np.isclose(s["rmse"], np.sqrt((100 + 400 + 0 + 1600) / 4))
    # 10/100 + 20/200 + 0/300 + 40/400 = 0.10 + 0.10 + 0 + 0.10
    assert np.isclose(s["mape"], 0.30 / 4 * 100)                    # 7.5%
    assert s["bias"] == (10 - 20 + 0 + 40) / 4                      # +7.5
    assert s["n"] == 4


def test_bias_and_accuracy_separate_two_different_failures():
    """Same MAE, opposite meaning: one is noise, the other is a systematic
    shortfall. The KPI module must not blur them into one number."""
    noisy = Y + np.array([20.0, -20.0, 20.0, -20.0])
    short = Y - 20.0
    assert kpis.score(Y, noisy)["mae"] == kpis.score(Y, short)["mae"]
    assert kpis.score(Y, noisy)["bias"] == 0.0
    assert kpis.score(Y, short)["bias"] == -20.0


def test_asymmetric_cost_charges_the_two_directions_differently():
    # under-forecast MWh: 20 (the -20 error) ; over: 10 + 40 = 50
    c = kpis.asymmetric_cost(Y, YHAT, eur_under=100.0, eur_over=10.0)
    assert c["mwh_under"] == 20.0
    assert c["mwh_over"] == 50.0
    assert c["total"] == 20.0 * 100.0 + 50.0 * 10.0                 # 2500
    assert np.isclose(c["annual"], 2500 / 4 * kpis.HOURS_PER_YEAR)


def test_coverage_counts_realisations_inside_the_band():
    lo = np.array([90.0, 190.0, 290.0, 500.0])
    hi = np.array([120.0, 210.0, 310.0, 600.0])
    assert kpis.coverage(Y, lo, hi) == 75.0
    assert kpis.interval_width(lo, hi) == np.mean(hi - lo)
