"""The baselines the model has to beat before it is worth anything.

Three, in rising order of competence, all of them things a person could do
with a spreadsheet:

  persistence      whatever the load was at the origin, held flat.
  seasonal naive   the same hour one week ago. Cheap, and genuinely hard to
                   beat on a series with this much weekly structure.
  hour of week     the average of that hour of the week over the last four
                   weeks. A dispatcher's rule of thumb.

None of them needs fitting, and all three read only positions at or before
their origin, exactly like the models.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

WEEK = 168
HOW_WEEKS = 4

NAMES = {
    "persistence": "Persistence (load at origin)",
    "seasonal_naive": "Seasonal naive (same hour, last week)",
    "hour_of_week": "Hour-of-week mean (last 4 weeks)",
}


def persistence(series: pd.DataFrame, index: pd.DataFrame) -> np.ndarray:
    load = series["load"].to_numpy()
    return load[index["origin_pos"].to_numpy()]


def seasonal_naive(series: pd.DataFrame, index: pd.DataFrame) -> np.ndarray:
    load = series["load"].to_numpy()
    return load[index["target_pos"].to_numpy() - WEEK]


def hour_of_week(series: pd.DataFrame, index: pd.DataFrame) -> np.ndarray:
    load = series["load"].to_numpy()
    t = index["target_pos"].to_numpy()
    stack = np.stack([load[t - WEEK * k] for k in range(1, HOW_WEEKS + 1)])
    return stack.mean(axis=0)


ALL = {
    "persistence": persistence,
    "seasonal_naive": seasonal_naive,
    "hour_of_week": hour_of_week,
}


def predict_all(series: pd.DataFrame, index: pd.DataFrame) -> dict:
    return {name: fn(series, index) for name, fn in ALL.items()}
