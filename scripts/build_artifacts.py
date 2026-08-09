"""Everything expensive happens here, once, on my machine.

Fetches the sources, builds the series, walks the rolling-origin backtest with
a quarterly refit, and writes parquet to data/processed. Those files are
committed; the app reads them and trains nothing. A backtest over five years
is not something a visitor should wait for.

    python scripts/build_artifacts.py

Roughly ten minutes on a laptop. Add --fast for a two-fold smoke run.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from gridcast import backtest, baseline, data, kpis, model  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "processed"
TEST_START = "2018-01-01"
LEVEL = 0.80

PRED_COLS = ["pred_persistence", "pred_seasonal_naive", "pred_hour_of_week",
             "pred_fourier", "pred_lgbm"]


def run_backtest(series: pd.DataFrame, folds) -> pd.DataFrame:
    parts = []
    for fold in folds:
        t0 = time.time()
        backtest.assert_no_leakage(series, fold)

        _, y, index = backtest.test_set(series, fold)
        out = index.copy()
        out["fold"] = fold.number
        out["cut"] = fold.cut
        out["y"] = y

        for name, pred in baseline.predict_all(series, index).items():
            out[f"pred_{name}"] = pred

        for m in model.all_models():
            m.fit(series, fold.cut_pos)
            out[f"pred_{m.name}"] = m.predict(series, index)

        parts.append(out)
        print(f"  fold {fold.number:>2}  trained through {fold.cut.date()}  "
              f"{len(index):>6,} forecasts  {time.time() - t0:5.1f}s", flush=True)
    return pd.concat(parts, ignore_index=True)


def add_intervals(bt: pd.DataFrame) -> pd.DataFrame:
    """Intervals for fold k come from the realised errors of folds < k. Fold 1
    has no history to learn a width from and is left empty rather than
    borrowed from the future."""
    bt = bt.copy()
    bt["lo80"] = np.nan
    bt["hi80"] = np.nan
    for fold in sorted(bt["fold"].unique()):
        past = bt[bt["fold"] < fold]
        if past.empty:
            continue
        resid = pd.DataFrame({"h": past["h"],
                              "resid": past["y"] - past["pred_lgbm"]})
        q = model.residual_quantiles(resid, LEVEL)
        cur = bt["fold"] == fold
        h = bt.loc[cur, "h"]
        bt.loc[cur, "lo80"] = bt.loc[cur, "pred_lgbm"] + h.map(q["lo"]).to_numpy()
        bt.loc[cur, "hi80"] = bt.loc[cur, "pred_lgbm"] + h.map(q["hi"]).to_numpy()
    return bt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="two folds only, for a quick end-to-end check")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("series...", flush=True)
    raw = data.load_raw_opsd()
    series = data.build_series()
    summary = data.summarise(series)
    print(f"  {summary['hours']:,} hours  {summary['start']} .. {summary['end']}",
          flush=True)

    folds = backtest.make_folds(series, TEST_START)
    if args.fast:
        folds = folds[:2]
    print(f"backtest: {len(folds)} folds, "
          f"{sum(f.n_forecasts for f in folds):,} forecasts", flush=True)

    bt = run_backtest(series, folds)
    bt = add_intervals(bt)
    bt["is_holiday"] = series["is_holiday"].to_numpy()[bt["target_pos"].to_numpy()]
    bt["segment"] = kpis.segments(bt)

    for c in PRED_COLS + ["y", "lo80", "hi80"]:
        bt[c] = bt[c].astype("float32")

    keep = ["fold", "origin", "target", "h", "y", *PRED_COLS,
            "lo80", "hi80", "is_holiday", "segment"]
    bt[keep].to_parquet(OUT / "backtest.parquet", index=False)

    series.reset_index().rename(columns={"index": "utc_timestamp"}) \
        .to_parquet(OUT / "series.parquet", index=False)
    data.level_break_evidence(raw).reset_index() \
        .to_parquet(OUT / "monthly_levels.parquet", index=False)

    overall = {c.replace("pred_", ""): kpis.score(bt["y"], bt[c]) for c in PRED_COLS}
    scored = bt[bt["lo80"].notna()]
    meta = {
        "built_utc": pd.Timestamp.utcnow().isoformat(),
        "series": summary,
        "test_start": TEST_START,
        "refit_days": backtest.REFIT_DAYS,
        "folds": backtest.summary(folds).to_dict("records"),
        "overall": overall,
        "coverage80": kpis.coverage(scored["y"], scored["lo80"], scored["hi80"]),
        "interval_width80": kpis.interval_width(scored["lo80"], scored["hi80"]),
        "build_seconds": round(time.time() - t_start, 1),
        "fast": args.fast,
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\noverall MAPE, all horizons pooled:")
    for name, s in sorted(overall.items(), key=lambda kv: kv[1]["mape"]):
        print(f"  {name:<16} {s['mape']:5.2f}%   bias {s['bias']:+7.0f} MW")
    print(f"\n80% interval covered {meta['coverage80']:.1f}% of realisations "
          f"(width {meta['interval_width80']:,.0f} MW)")
    print(f"\nwritten to {OUT}  ({meta['build_seconds']}s)")
    for f in sorted(OUT.glob("*")):
        print(f"  {f.name:<24} {f.stat().st_size / 1e6:6.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
