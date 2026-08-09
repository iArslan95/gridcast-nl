"""The feature matrix, and the rule that makes leakage structurally impossible.

A forecast is issued at an *origin* O and covers targets t = O + h for
h = 1..168. Everything the model may look at has to be known at O. Two kinds of
feature satisfy that:

  origin lags     load at O, O-1, ..., O-168 and rolling means ending at O.
                  Anchored to the origin, so identical for every horizon.
  target lags     load at t-168 and t-336. These sit at O + h - 168 and
                  O + h - 336, which is at or before O for every h <= 168.
                  That is the only reason they are allowed, and it is why
                  MAX_HORIZON is a hard constant rather than a default.
  calendar        hour, weekday, holiday, Fourier terms of the *target*. Known
                  years ahead; no forecast of its own is required.

Weather is deliberately absent. Temperature explains a lot of load, but using
it at t+168 means feeding the model a weather forecast whose own error is not
accounted for anywhere in this backtest. The app shows the relation and leaves
it out of the model.

One model covers all 168 horizons (direct multi-horizon) with h as a feature,
so the error growth over the horizon is something the model produces rather
than something the setup imposes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAX_HORIZON = 168
ORIGIN_LAGS = (0, 1, 2, 3, 24, 48, 168)
ORIGIN_ROLLS = (24, 168)
TARGET_LAGS = (168, 336)
MIN_ORIGIN_POS = max(max(ORIGIN_LAGS), max(ORIGIN_ROLLS), max(TARGET_LAGS))

# Fourier pairs per cycle: day, week, year.
FOURIER = ((24.0, 4), (168.0, 3), (8766.0, 3))


def _fourier(t: np.ndarray, period: float, order: int, tag: str) -> dict:
    out = {}
    for k in range(1, order + 1):
        ang = 2.0 * np.pi * k * t / period
        out[f"sin_{tag}_{k}"] = np.sin(ang)
        out[f"cos_{tag}_{k}"] = np.cos(ang)
    return out


def calendar_block(series: pd.DataFrame, pos: np.ndarray) -> dict:
    """Calendar features of the target positions. Pure clock and almanac."""
    hours_since_epoch = pos.astype(float)
    cols = {
        "hour": series["hour"].to_numpy()[pos],
        "dow": series["dow"].to_numpy()[pos],
        "month": series["month"].to_numpy()[pos],
        "doy": series["doy"].to_numpy()[pos],
        "is_weekend": series["is_weekend"].to_numpy()[pos].astype(np.int8),
        "is_holiday": series["is_holiday"].to_numpy()[pos].astype(np.int8),
        "is_yearend": series["is_yearend"].to_numpy()[pos].astype(np.int8),
    }
    for period, order in FOURIER:
        tag = {24.0: "d", 168.0: "w", 8766.0: "y"}[period]
        cols.update(_fourier(hours_since_epoch, period, order, tag))
    return cols


def build(series: pd.DataFrame, origins: np.ndarray,
          horizons: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Cross every origin with every horizon.

    Returns (X, y, index) where index carries origin_pos, h and target_pos so
    the backtest can group by horizon without recomputing anything.
    """
    horizons = np.asarray(horizons, dtype=np.int64)
    origins = np.asarray(origins, dtype=np.int64)
    return build_pairs(series,
                       np.repeat(origins, len(horizons)),
                       np.tile(horizons, len(origins)))


def build_pairs(series: pd.DataFrame, o: np.ndarray,
                h: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Same features for explicit (origin, horizon) pairs, in the order given.

    Prediction goes through here so a row of X always belongs to the row of the
    index frame beside it, rather than to whatever order a cross product
    happened to produce.
    """
    o = np.asarray(o, dtype=np.int64)
    h = np.asarray(h, dtype=np.int64)
    if len(o) != len(h):
        raise ValueError("origin and horizon arrays must be the same length")
    if h.min() < 1 or h.max() > MAX_HORIZON:
        raise ValueError(f"horizons must lie in 1..{MAX_HORIZON}; "
                         f"target lags are only origin-safe there")
    if o.min() < MIN_ORIGIN_POS:
        raise ValueError(f"origin needs {MIN_ORIGIN_POS} hours of history")
    if (o + h).max() >= len(series):
        raise ValueError("origin + horizon runs past the end of the series")

    t = o + h

    load = series["load"].to_numpy(dtype=np.float64)
    cols = {"h": h.astype(np.int16), "h_day": (h // 24).astype(np.int8)}

    for lag in ORIGIN_LAGS:
        cols[f"olag_{lag}"] = load[o - lag]
    for win in ORIGIN_ROLLS:
        csum = np.concatenate([[0.0], np.cumsum(load)])
        cols[f"oroll_{win}"] = (csum[o + 1] - csum[o + 1 - win]) / win
    for lag in TARGET_LAGS:
        cols[f"tlag_{lag}"] = load[t - lag]

    # Level relative to the recent week: keeps the trees from having to learn
    # the absolute level, which drifts.
    cols["odev_24_168"] = cols["oroll_24"] - cols["oroll_168"]
    cols.update(calendar_block(series, t))

    X = pd.DataFrame(cols)
    y = load[t]
    index = pd.DataFrame({
        "origin_pos": o, "h": h, "target_pos": t,
        "origin": series.index.to_numpy()[o],
        "target": series.index.to_numpy()[t],
    })
    return X, y, index


def latest_origin_uses_only_past(series: pd.DataFrame, origins: np.ndarray,
                                 horizons: np.ndarray) -> int:
    """Highest series position any feature reads. Must never exceed the origin.

    Recomputed from the definitions rather than asserted by hand, so a new
    feature that breaks the rule is caught instead of trusted.
    """
    reach = []
    for lag in ORIGIN_LAGS:
        reach.append(-lag)
    for win in ORIGIN_ROLLS:
        reach.append(0)
    for lag in TARGET_LAGS:
        reach.append(int(horizons.max()) - lag)
    return max(reach)
