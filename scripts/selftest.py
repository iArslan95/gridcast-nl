"""Invariants over the committed backtest, not a list of loose assertions.

    python scripts/selftest.py

Each block states a property the results have to keep having. Two of them
assert things that look like failures: the intervals are too narrow, and there
is a fold where the boosted model loses to seasonal naive. Both are true, both
are the most informative results in the repository, and pinning them means a
later change cannot quietly paper over either one.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from gridcast import kpis  # noqa: E402

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS = ["lgbm", "seasonal_naive", "hour_of_week", "fourier", "persistence"]
COVID = ("2020-03-15", "2020-07-01")


def load():
    bt = pd.read_parquet(DATA / "backtest.parquet")
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    return bt, meta


def check_same_yardstick(bt):
    n = len(bt)
    for m in MODELS:
        col = f"pred_{m}"
        assert bt[col].notna().all(), f"{m} has gaps"
        assert len(bt[col]) == n, f"{m} scored on a different number of rows"
    assert bt["y"].notna().all()
    print(f"yardstick : {n:,} forecasts, all {len(MODELS)} predictors on every one")


def check_model_earns_its_place(bt):
    s = {m: kpis.score(bt["y"], bt[f"pred_{m}"]) for m in MODELS}
    assert s["lgbm"]["mape"] < s["seasonal_naive"]["mape"], \
        "the model must beat seasonal naive overall or it has no reason to exist"
    assert s["lgbm"]["mae"] < s["seasonal_naive"]["mae"]
    assert s["seasonal_naive"]["mape"] < s["persistence"]["mape"]

    by_h = bt.groupby("h").apply(
        lambda d: pd.Series({m: kpis.score(d["y"], d[f"pred_{m}"])["mape"]
                             for m in MODELS}))
    losses = int((by_h["lgbm"] >= by_h["seasonal_naive"]).sum())
    assert losses == 0, f"the model loses to seasonal naive at {losses} horizons"
    print(f"accuracy  : MAPE {s['lgbm']['mape']:.2f}% vs seasonal naive "
          f"{s['seasonal_naive']['mape']:.2f}%, ahead at all 168 horizons")
    return by_h


def check_horizon_behaves(by_h):
    """The boosted model holds recent load, so it has something to lose as the
    horizon grows. The other two do not, so they must stay flat.

    Compared at h against h + 24, never h against h + 1. Origins run every six
    hours, so each horizon lands on a fixed set of four clock hours and that
    set shifts with h: comparing neighbouring horizons compares different times
    of day. A day apart, the clock hours match exactly and only the horizon
    differs.
    """
    lgbm = by_h["lgbm"]
    first = lgbm.loc[1:24].mean()
    last = lgbm.loc[145:168].mean()
    assert last > first, "the boosted model should get worse further out"

    def day_apart(col):
        h = by_h.index.to_numpy()
        pairs = [(k, by_h[col].loc[k + 24] - by_h[col].loc[k])
                 for k in h if k + 24 in by_h.index]
        return np.array([d for _, d in pairs])

    for flat in ("seasonal_naive", "fourier"):
        drift = np.abs(day_apart(flat)).max()
        assert drift < 0.10, (
            f"{flat} uses the same information at every horizon, so a day of "
            f"extra lead time must change nothing; largest matched-hour move is "
            f"{drift:.3f} pp")

    steps = day_apart("lgbm")
    assert (steps > 0).mean() > 0.9, (
        "a day of extra lead time should cost the boosted model accuracy at "
        "almost every matched-hour comparison")
    print(f"horizon   : {first:.2f}% at 1-24h rising to {last:.2f}% at 145-168h; "
          f"a day of lead time costs it {steps.mean():.3f} pp on average, and "
          f"costs the flat models at most "
          f"{np.abs(day_apart('seasonal_naive')).max():.3f} pp")


def check_intervals_are_honest(bt, meta):
    s = bt[bt["lo80"].notna()]
    cov = kpis.coverage(s["y"], s["lo80"], s["hi80"])
    assert bt[bt["fold"] == 1]["lo80"].isna().all(), \
        "fold 1 must carry no interval: no out-of-sample error existed yet"
    assert cov < 80.0, (
        "coverage is at or above nominal, which would mean this run no longer "
        "shows the calibration failure the app is built around")

    width = s.groupby("h").apply(lambda d: float((d["hi80"] - d["lo80"]).mean()))
    assert width.loc[168] > width.loc[1], "bands must widen with the horizon"

    covid = s[(s["target"] >= COVID[0]) & (s["target"] < COVID[1])]
    rest = s[(s["target"] < COVID[0]) | (s["target"] >= COVID[1])]
    c_covid = kpis.coverage(covid["y"], covid["lo80"], covid["hi80"])
    c_rest = kpis.coverage(rest["y"], rest["lo80"], rest["hi80"])
    assert c_covid < c_rest - 5, \
        "the calibration failure should concentrate in the 2020 regime change"
    print(f"intervals : 80% band covers {cov:.1f}% overall — "
          f"{c_rest:.1f}% outside spring 2020, {c_covid:.1f}% inside it")


def check_the_model_loses_somewhere(bt):
    """An honest backtest has a bad patch in it. This asserts that the repo
    still contains one, and prints where."""
    rows = []
    for fold, d in bt.groupby("fold"):
        rows.append({
            "fold": int(fold),
            "from": str(d["target"].min().date()),
            "lgbm": kpis.score(d["y"], d["pred_lgbm"])["mape"],
            "seasonal_naive": kpis.score(d["y"], d["pred_seasonal_naive"])["mape"],
        })
    folds = pd.DataFrame(rows)
    lost = folds[folds["lgbm"] >= folds["seasonal_naive"]]
    assert len(lost) >= 1, (
        "no fold where the model loses: either the data changed or the 2020 "
        "regime break stopped being visible, and the app's honesty claim with it")
    for _, r in lost.iterrows():
        print(f"regime    : fold {int(r['fold'])} from {r['from']} — model "
              f"{r['lgbm']:.2f}% vs seasonal naive {r['seasonal_naive']:.2f}%, "
              f"the model is the worse of the two")
    return folds


def check_segments(bt):
    seg = bt.groupby("segment").apply(
        lambda d: pd.Series({
            "lgbm": kpis.score(d["y"], d["pred_lgbm"])["mape"],
            "seasonal_naive": kpis.score(d["y"], d["pred_seasonal_naive"])["mape"]}))
    hol = seg.loc["public holiday"]
    assert hol["lgbm"] < hol["seasonal_naive"] * 0.75, (
        "holidays are where a calendar-aware model should be clearly ahead")
    print(f"segments  : public holidays {hol['lgbm']:.2f}% vs "
          f"{hol['seasonal_naive']:.2f}% for seasonal naive")


def check_tail_bias(bt):
    """The headline bias is near zero. The bias in the top decile is not.

    For a national series that is a footnote. For anything with a hard capacity
    limit it is the whole question, because the model is at its most optimistic
    exactly where optimism is most expensive. Pinned here so a later change
    cannot quietly drop the finding the decision tab is built on.
    """
    d = bt[bt["h"] == 24]
    overall_bias = float((d["pred_lgbm"] - d["y"]).mean())
    top = d[d["y"] >= d["y"].quantile(0.90)]
    tail_bias = float((top["pred_lgbm"] - top["y"]).mean())
    assert tail_bias < 0, (
        "the model is expected to under-forecast the top decile; if that ever "
        "flips, the decision tab needs rewriting rather than reprinting")
    assert tail_bias < overall_bias - 50, (
        "the tail bias should be materially worse than the headline bias, "
        f"got {tail_bias:+.0f} against {overall_bias:+.0f} MW")
    sn_tail = float((top["pred_seasonal_naive"] - top["y"]).mean())
    print(f"staart    : day-ahead bias {overall_bias:+.0f} MW overall but "
          f"{tail_bias:+.0f} MW in the top decile "
          f"(seasonal naive {sn_tail:+.0f} MW there)")


def main():
    bt, meta = load()
    check_same_yardstick(bt)
    by_h = check_model_earns_its_place(bt)
    check_horizon_behaves(by_h)
    check_intervals_are_honest(bt, meta)
    check_segments(bt)
    check_tail_bias(bt)
    check_the_model_loses_somewhere(bt)
    print("ALL OK")


if __name__ == "__main__":
    main()
