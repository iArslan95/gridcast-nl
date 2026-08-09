"""The shared yardstick. Every model and every baseline is scored here, by
these functions, on exactly the same rows.

Accuracy and bias are reported separately on purpose. A model that is 400 MW
off in random directions and a model that is 400 MW short every single hour
have the same MAE and call for completely different actions: the first is
noise you buy a reserve against, the second is a systematic error you fix.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# One hour at a constant MW is that many MWh, so an error in MW is an error in
# MWh per hour. The euro panel leans on that.
HOURS_PER_YEAR = 8766.0


def _err(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    return yhat - y


def score(y: np.ndarray, yhat: np.ndarray) -> dict:
    e = _err(np.asarray(y, float), np.asarray(yhat, float))
    y = np.asarray(y, float)
    return {
        "mae": float(np.mean(np.abs(e))),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "mape": float(np.mean(np.abs(e / y)) * 100),
        "bias": float(np.mean(e)),
        "bias_pct": float(np.mean(e) / np.mean(y) * 100),
        "n": int(len(e)),
    }


def by_group(df: pd.DataFrame, pred_cols: list[str], group: str) -> pd.DataFrame:
    """Score every prediction column within each level of `group`."""
    rows = []
    for key, part in df.groupby(group, observed=True):
        for col in pred_cols:
            rows.append({group: key, "model": col.replace("pred_", ""),
                         **score(part["y"].to_numpy(), part[col].to_numpy())})
    return pd.DataFrame(rows)


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Share of realisations that actually fell inside the interval."""
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    ok = np.isfinite(lo) & np.isfinite(hi)
    if not ok.any():
        return float("nan")
    return float(np.mean((y[ok] >= lo[ok]) & (y[ok] <= hi[ok])) * 100)


def interval_width(lo: np.ndarray, hi: np.ndarray) -> float:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    ok = np.isfinite(lo) & np.isfinite(hi)
    return float(np.mean(hi[ok] - lo[ok])) if ok.any() else float("nan")


def segments(df: pd.DataFrame) -> pd.Series:
    """Where a model fails matters more than the average. These are the cuts
    an operations team would ask about."""
    local = df["target"].dt.tz_convert("Europe/Amsterdam")
    seg = pd.Series("working day", index=df.index, dtype=object)
    seg[local.dt.dayofweek >= 5] = "weekend"
    seg[df["is_holiday"].astype(bool)] = "public holiday"
    return seg


def asymmetric_cost(y: np.ndarray, yhat: np.ndarray,
                    eur_under: float, eur_over: float) -> dict:
    """Under- and over-forecasting do not cost the same, so one number for
    'accuracy' hides the decision. Errors are in MW over one hour, hence MWh.

    Illustrative by construction: the real cost function is settled with the
    business, not chosen by whoever built the model.
    """
    e = _err(np.asarray(y, float), np.asarray(yhat, float))
    under = np.maximum(-e, 0.0)      # forecast below realisation
    over = np.maximum(e, 0.0)
    total = float(under.sum() * eur_under + over.sum() * eur_over)
    hours = len(e)
    return {
        "total": total,
        "per_hour": total / hours if hours else float("nan"),
        "annual": total / hours * HOURS_PER_YEAR if hours else float("nan"),
        "mwh_under": float(under.sum()),
        "mwh_over": float(over.sum()),
    }


def value_against(df: pd.DataFrame, model_col: str, baseline_col: str,
                  eur_under: float, eur_over: float) -> dict:
    """What the model is worth relative to the baseline, in euro per year."""
    y = df["y"].to_numpy()
    m = asymmetric_cost(y, df[model_col].to_numpy(), eur_under, eur_over)
    b = asymmetric_cost(y, df[baseline_col].to_numpy(), eur_under, eur_over)
    return {
        "model_annual": m["annual"],
        "baseline_annual": b["annual"],
        "saving_annual": b["annual"] - m["annual"],
        "saving_pct": (b["annual"] - m["annual"]) / b["annual"] * 100
        if b["annual"] else float("nan"),
    }
