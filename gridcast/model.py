"""Three models, all of which I can defend line by line.

  seasonal naive     lives in baseline.py. It is listed here too because on a
                     series this regular it is a real competitor, not a straw
                     man, and the other two have to beat it to exist.
  seasonal Fourier    ridge regression on log load with Fourier terms for the
                     daily, weekly and yearly cycles, plus holiday effects.
                     It sees no recent load at all, so its error is flat over
                     the horizon: a week out it is exactly as good as an hour
                     out. That contrast is the point of putting it in.
  gradient boosting  LightGBM over origin lags, target lags, calendar and the
                     horizon itself. It has both kinds of information, so its
                     error should start below the others and converge towards
                     the Fourier model as the horizon grows.

No deep learning and no Prophet. Neither would earn its place here, and both
would cost the ability to say why a number came out the way it did.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from . import backtest, features


class SeasonalFourier:
    """Calendar only: what the clock and the almanac imply, nothing else."""

    name = "fourier"
    label = "Seasonal Fourier (ridge, log load)"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.sigma2 = 0.0

    @staticmethod
    def _design(series: pd.DataFrame, pos: np.ndarray) -> np.ndarray:
        cols = features.calendar_block(series, pos)
        four = np.column_stack([v for k, v in cols.items()
                                if k.startswith(("sin_", "cos_"))])
        flags = np.column_stack([
            cols["is_weekend"], cols["is_holiday"], cols["is_yearend"]]).astype(float)
        daily = np.column_stack([v for k, v in cols.items()
                                 if k.startswith(("sin_d", "cos_d"))])
        # The shape of a weekend day and of a holiday differs from a working
        # day, so the daily harmonics get their own copy for each.
        inter = np.column_stack([daily * flags[:, [0]], daily * flags[:, [1]]])
        trend = (pos / 8766.0).reshape(-1, 1)
        return np.column_stack([four, flags, inter, trend])

    def fit(self, series: pd.DataFrame, cut_pos: int) -> "SeasonalFourier":
        pos = np.arange(0, cut_pos + 1)
        X = self._design(series, pos)
        y = np.log(series["load"].to_numpy()[pos])
        self.model.fit(X, y)
        resid = y - self.model.predict(X)
        # Half the residual variance corrects the bias that exp() otherwise
        # introduces when transforming back from logs.
        self.sigma2 = float(np.var(resid))
        return self

    def predict(self, series: pd.DataFrame, index: pd.DataFrame) -> np.ndarray:
        pos = index["target_pos"].to_numpy()
        return np.exp(self.model.predict(self._design(series, pos)) + self.sigma2 / 2)


class GradientBoosting:
    """Origin lags, target lags, calendar and the horizon, in one model."""

    name = "lgbm"
    label = "Gradient boosting (LightGBM, direct multi-horizon)"

    PARAMS = dict(
        objective="l2",
        n_estimators=500,
        learning_rate=0.06,
        num_leaves=63,
        min_child_samples=60,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        n_jobs=-1,
        verbose=-1,
    )

    def __init__(self, **overrides):
        self.params = {**self.PARAMS, **overrides}
        self.model = None
        self.feature_names: list[str] = []

    def fit(self, series: pd.DataFrame, cut_pos: int) -> "GradientBoosting":
        import lightgbm as lgb

        X, y, _ = backtest.training_set(series, cut_pos)
        self.feature_names = list(X.columns)
        self.model = lgb.LGBMRegressor(**self.params)
        self.model.fit(X, y)
        return self

    def predict(self, series: pd.DataFrame, index: pd.DataFrame) -> np.ndarray:
        X, _, _ = features.build_pairs(series, index["origin_pos"].to_numpy(),
                                       index["h"].to_numpy())
        return self.model.predict(X[self.feature_names])

    def importances(self) -> pd.DataFrame:
        return (pd.DataFrame({"feature": self.feature_names,
                              "gain": self.model.booster_.feature_importance("gain")})
                .sort_values("gain", ascending=False, ignore_index=True))


def all_models() -> list:
    return [SeasonalFourier(), GradientBoosting()]


def residual_quantiles(residuals: pd.DataFrame, level: float = 0.80) -> pd.DataFrame:
    """Empirical prediction intervals, one pair of quantiles per horizon.

    Deliberately not a formula. A normal interval around a boosted tree is a
    claim about a distribution nobody checked; the realised errors of previous
    folds are the thing you actually have. Which is also why fold 1 gets no
    interval at all: at that point no out-of-sample error existed yet.
    """
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    g = residuals.groupby("h")["resid"]
    return pd.DataFrame({"lo": g.quantile(lo_q), "hi": g.quantile(hi_q)})
