"""Rolling origin backtest: walk forward, refit, score only unseen hours.

There is no train/test split in this file. A split assumes the future is
exchangeable with the past, which for a load series it is not: shuffle the
hours and a model can interpolate between the hour before and the hour after
and look excellent while being useless.

Instead the clock runs forward. At each refit cut the model sees targets up to
and including that cut and nothing after it. It then forecasts every day of
the following quarter, 168 hours out, and those forecasts are scored once.
`assert_no_leakage` re-derives that property from the actual arrays rather
than trusting the loop, and the test suite runs it on every fold.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import features

# Denser near the origin, where the error curve moves fastest.
TRAIN_HORIZONS = np.array(
    list(range(1, 25)) + [30, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168])
TRAIN_ORIGIN_STRIDE = 3          # hours between training origins
TEST_HORIZONS = np.arange(1, features.MAX_HORIZON + 1)
REFIT_DAYS = 91                  # a quarterly retrain, as a team would run it

# Four forecast origins a day rather than one. With a single daily origin every
# horizon lands on a fixed clock hour: h = 24, 48 ... 168 would all fall at
# midnight, the steadiest hour there is, and the error-versus-horizon curve
# would be measuring the time of day instead of the horizon. Spreading the
# origins over 00, 06, 12 and 18 UTC mixes four clock hours into every horizon.
TEST_ORIGIN_STRIDE = 6


@dataclass(frozen=True)
class Fold:
    number: int
    cut_pos: int                 # last position the model is allowed to learn
    origins: np.ndarray          # forecast origins scored in this fold
    cut: pd.Timestamp
    first_origin: pd.Timestamp
    last_target: pd.Timestamp

    @property
    def n_forecasts(self) -> int:
        return len(self.origins) * len(TEST_HORIZONS)


def make_folds(series: pd.DataFrame, test_start: str,
               refit_days: int = REFIT_DAYS,
               origin_stride: int = TEST_ORIGIN_STRIDE) -> list[Fold]:
    """Quarterly refits from `test_start`, forecast origins every six hours."""
    n = len(series)
    index = series.index
    start_pos = int(index.get_indexer([pd.Timestamp(test_start, tz="UTC")])[0])
    if start_pos < features.MIN_ORIGIN_POS:
        raise ValueError("test_start leaves too little history to build features")

    last_origin = n - 1 - features.MAX_HORIZON
    folds = []
    cut_pos = start_pos
    while cut_pos <= last_origin:
        nxt = min(cut_pos + refit_days * 24, last_origin + 1)
        origins = np.arange(cut_pos, nxt, origin_stride)
        origins = origins[origins <= last_origin]
        if len(origins) == 0:
            break
        folds.append(Fold(
            number=len(folds) + 1,
            cut_pos=cut_pos,
            origins=origins,
            cut=index[cut_pos],
            first_origin=index[origins[0]],
            last_target=index[origins[-1] + features.MAX_HORIZON],
        ))
        cut_pos = nxt
    return folds


def training_origins(cut_pos: int, stride: int = TRAIN_ORIGIN_STRIDE) -> np.ndarray:
    """Origins whose every training target lands at or before the cut.

    The bound is `cut_pos - MAX_HORIZON`, not `cut_pos`: an origin closer to
    the cut than the longest horizon would need a target the model is not
    allowed to see, and dropping only those rows would silently reweight the
    long horizons out of the training set.
    """
    last = cut_pos - features.MAX_HORIZON
    if last < features.MIN_ORIGIN_POS:
        raise ValueError("not enough history before the cut to train")
    return np.arange(features.MIN_ORIGIN_POS, last + 1, stride)


def training_set(series: pd.DataFrame, cut_pos: int):
    origins = training_origins(cut_pos)
    return features.build(series, origins, TRAIN_HORIZONS)


def test_set(series: pd.DataFrame, fold: Fold):
    return features.build(series, fold.origins, TEST_HORIZONS)


def assert_no_leakage(series: pd.DataFrame, fold: Fold) -> None:
    """Every training target is at or before the cut; every scored target is
    strictly after it. Derived from the arrays, not from the loop that made
    them."""
    _, _, tr = training_set(series, fold.cut_pos)
    _, _, te = test_set(series, fold)

    if tr["target_pos"].max() > fold.cut_pos:
        raise AssertionError(
            f"fold {fold.number}: training target at position "
            f"{tr['target_pos'].max()} is past the cut {fold.cut_pos}")
    if te["target_pos"].min() <= fold.cut_pos:
        raise AssertionError(
            f"fold {fold.number}: scored target at position "
            f"{te['target_pos'].min()} is at or before the cut {fold.cut_pos}")
    reach = features.latest_origin_uses_only_past(
        series, fold.origins, TEST_HORIZONS)
    if reach > 0:
        raise AssertionError(
            f"a feature reads {reach} hours past its own origin")


def summary(folds: list[Fold]) -> pd.DataFrame:
    return pd.DataFrame([{
        "fold": f.number,
        "trained through": f.cut.strftime("%Y-%m-%d"),
        "first origin": f.first_origin.strftime("%Y-%m-%d"),
        "last target": f.last_target.strftime("%Y-%m-%d"),
        "origins": len(f.origins),
        "forecasts": f.n_forecasts,
    } for f in folds])
