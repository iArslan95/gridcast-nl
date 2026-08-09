# ⚡ GridCast — Dutch Electricity Load Forecasting

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-gradient%20boosting-green)
![Streamlit](https://img.shields.io/badge/Streamlit-app-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

How much electricity will the Netherlands draw in each of the next 168 hours,
and how much of that forecast should anyone believe?

Hourly national load, forecast one to seven days ahead, scored by a
rolling-origin backtest: 11 quarterly refits over 2018 to 2020, 669,984
forecasts, every one made from data that existed at the time. Real public data,
three defensible models, and every number sitting next to the baseline it has
to beat.

**Live app:** [gridcast-nl.streamlit.app](https://gridcast-nl.streamlit.app/)
(interface in Dutch; code, comments and this README in English)

## Why

The other demos in this portfolio are optimisation problems with forecasting in
a supporting role. This one is the forecast itself, and the harder half of it:
knowing when the number is trustworthy.

Forecast national load too low and there is not enough capacity contracted for
the hour. Forecast it too high and capacity is paid for and left unused. Those
two mistakes do not cost the same, which is why an accuracy percentage is not
yet an answer. What follows is the full chain: baselines first, an honest
backtest, calibration checked rather than claimed, and the error translated
into a quantity someone with a budget can act on.

The data is real, which is the point. In an optimisation demo you can invent a
scenario that behaves. In forecasting the difficulty is precisely that reality
does not cooperate, and this series proves it twice: a reporting break in the
source, and a pandemic in the final folds.

## What the app does

- **Takes the demand apart.** The yearly, weekly and daily layer of national
  load, as an hour-by-weekday heatmap and as day profiles split by day type and
  by season; which public holiday costs how much against a normal working day in
  the same month; and the temperature relation the model deliberately does not
  use.
- **Scores the forecast on those same segments.** Day type, season, time of day,
  hour by hour and horizon by horizon, with the baseline beside every number,
  because a single headline MAPE mostly reports how good the model is at night.
- **Puts the congestion where it actually is.** The operators' capacity map,
  one dot per supply area, coloured green to red by transport capacity status
  and sized by the queue in MW, filterable by network operator and by
  consumption or feed-in.
- **Backtests properly.** Walk forward, refit quarterly, forecast every six
  hours for 168 hours, score once. No random train/test split anywhere in the
  repository.
- **Reports error by horizon and by segment.** Working day against weekend
  against holiday, summer against winter, peak against night, fold by fold.
  Where a model fails is more useful than where it succeeds.
- **Checks its own intervals.** The 80% band contains 76.0% of realisations.
  That shortfall is on the front page, not buried.
- **Answers the capacity question, not the accuracy question.** Set a limit and
  a lead time, and each point forecast becomes a probability that the limit is
  passed, read off the empirical residual distribution of that horizon. The
  decision curve sweeps the alarm threshold from cautious to trigger-happy and
  marks where the plain "forecast above the limit" rule sits: one fixed point on
  a curve you should be choosing from. A reliability plot checks whether the
  probability means anything.
- **Prices the error.** MAPE converted into euros per year against the
  seasonal-naive baseline, with under- and over-forecasting priced separately,
  plus the same fact with no price attached at all: 937 GWh of forecast error
  avoided per year at day-ahead.
- **Monitors.** Weekly accuracy and bias with an alert threshold, and the
  March 2020 break that trips it for 24 straight weeks.

## Results

| | Gradient boosting | Seasonal naive | Hour-of-week mean | Seasonal Fourier | Persistence |
|---|---|---|---|---|---|
| MAPE | **3.56%** | 4.13% | 4.38% | 5.78% | 16.06% |
| MAE | **446 MW** | 526 MW | 559 MW | 707 MW | 2,004 MW |
| Bias | +84 MW | +5 MW | +20 MW | +252 MW | +100 MW |

Ahead of seasonal naive at all 168 horizons, on 669,984 forecasts scored on
identical hours.

The number that matters more than any of those: at day-ahead the overall bias is
**+76 MW**, essentially nothing, while in the **top decile of load the same model
runs 123 MW low**. It is at its most optimistic exactly in the hours that sit
against a capacity limit. A headline MAPE cannot see that, and for anything with
a hard limit it is the finding, not a footnote. `selftest.py` asserts it so a
later change cannot quietly lose it.

## The models

| Model | What it sees | Why it is here |
|---|---|---|
| **Seasonal naive** | the same hour one week ago | On a series this regular it is a real competitor, not a straw man. Anything that cannot beat it does not deserve to run. |
| **Seasonal Fourier** | Fourier terms for the daily, weekly and yearly cycle, holiday flags, a slow trend. Ridge on log load, no recent load at all | The pure calendar view. Its flat error curve is the reference that gives the boosted model's slope a meaning. Fitted on logs because holidays and seasons act proportionally, with a half-variance correction on the back-transform so the log fit adds no bias of its own. |
| **Gradient boosting** | origin lags (0 to 168 h), trailing means, the same hour one and two weeks before the target, calendar, and the horizon itself | One model for all 168 horizons with `h` as a feature, so the shape of the error-versus-horizon curve is produced by the model rather than imposed by the setup. |

No deep learning and no Prophet. Neither would earn its place on 41,640 hourly
observations, and both would cost the ability to explain a number.

### Why leakage is structurally impossible here

A forecast issued at origin `O` covers targets `t = O + h`. Only two kinds of
feature are allowed: values anchored at or before `O`, and values at
`t - 168` and `t - 336`, which sit at `O + h - 168` and are in the past for
every `h ≤ 168`. That inequality is why `MAX_HORIZON` is a hard constant rather
than a configurable default.

Training rows stop a full 168 hours before each refit cut, not at the cut. An
origin closer than the longest horizon would need a target the model may not
see, and dropping only those rows would quietly delete the long horizons from
the training set.

`tests/test_no_leakage.py` checks the property rather than the intention: it
replaces every value after the origin with nonsense, rebuilds the features, and
requires them to come out identical.

## What this does not do

The demo is honest about its own edges, so here they are in one place.

- **The weather is not in the model, and that costs something real.** Using
  temperature at t+168 means feeding in a weather forecast whose own error is
  measured nowhere in this backtest, so a model scored on temperatures that
  actually happened reports an accuracy nobody reproduces in operation. Leaving
  it out is the honest choice available without building a second error budget.
  The bill arrives in August 2020: in the hottest week of the whole series the
  model runs 1,119 MW under realisation, and the monitoring tab says so.
- **National load is not grid load.** A distribution network operator forecasts
  a substation or a neighbourhood, where a single industrial connection or one
  street of heat pumps moves the series in a way that averages out completely at
  national level. Two consequences: the accuracy figures here are optimistic for
  an asset-level series, and the relevant target changes from an average to a
  tail, since what a limit cares about is the top percent of hours. The decision
  tab is built for the second point on the data that exists; the first stays a
  caveat. The methods transfer, the numbers do not.
- **Gross consumption is not net load.** With enough solar and wind, the
  quantity that matters becomes demand minus generation, which ramps harder and
  is driven by weather rather than by the calendar. In this dataset that
  argument is weaker than expected: over 2019 the 99th percentile hourly ramp is
  1,676 MW net against 1,694 MW gross, because Dutch renewables were still small
  relative to load. That is exactly why a series ending in 2020 does not
  describe the grid of today.
- **School holidays are missing.** They matter, and the historical Dutch tables
  are no longer publicly retrievable (the Rijksoverheid open-data endpoint
  returns 404, and the holiday sites now list 2026 onwards only). Inventing the
  dates was not an option, so the feature is absent and the effect sits in the
  July and August residuals.
- **The intervals are too narrow.** 76.0% inside an 80% band. They are the
  empirical quantiles of errors the model previously made, which describes the
  past well and holds only while the future resembles it. In spring 2020 it did
  not, and coverage fell to 66.5%.
- **The series stops on 30 September 2020.** That is where the source ends. It
  covers no energy crisis, no post-2021 electrification, and no recent solar
  build-out.

## The data

| Source | What | Licence |
|---|---|---|
| [Open Power System Data](https://data.open-power-system-data.org/time_series/2020-10-06/), time series 2020-10-06 | Hourly NL load, originally ENTSO-E Transparency | CC-BY 4.0 |
| [Open-Meteo](https://open-meteo.com/en/docs/historical-weather-api) historical archive (ERA5) | Hourly 2 m temperature, De Bilt | CC-BY 4.0 |
| [`holidays`](https://pypi.org/project/holidays/) | Dutch public holidays | MIT |
| [Capaciteitskaart elektriciteitsnet](https://capaciteitskaart.netbeheernederland.nl/), Netbeheer Nederland via Esri Nederland | Congestion status, queue in MW and parties waiting per supply area | Esri Nederland Terms of Use |

The capacity map is **not** an open licence, so only a derived summary is
committed: area name, operator, status, queue, and a centroid. The polygons
themselves are not redistributed. It is also a live snapshot rather than a
2016-2020 series, and it is in the app for exactly that contrast: the forecast
is national and historical, the congestion problem is local and current.

Two things about that source worth knowing before anyone builds on it. The 927
drawn shapes describe 571 supply areas, so summing the queue over rows counts
the same megawatts two or three times, and 116 areas carry a different status on
different shapes (the app takes the most severe, which is a choice, not a fact).
And the status codes carry no domain in the service metadata: their meaning is
taken from the publisher's own renderer expression, which inverts the intuition,
since code 0 means capacity is available rather than unknown. Guessing that from
the numbering would have put the wrong colours on the map.

Two findings from the source are in the app rather than in a footnote, because
checking them was the first real work of the project.

**The series starts in 2016, not 2015.** Every month of 2015 sits 1.7 to 2.0 GW
below the same month of every later year, while 2016 through 2019 lie on top of
each other. Demand does not move 16% in one January and then hold still for four
years. That is a change in what was reported.

**The published day-ahead forecast is not used as a baseline.** The same file
ships an ENTSO-E day-ahead load forecast, which was the obvious hard benchmark.
Scored against the actuals beside it, it runs about +5% for three years and then
flips to −14%, and on Christmas Day 2018 it predicts 19,423 MW against 12,139 MW
realised. Those are two columns on different bases, not a forecast that got
worse. It appears in the app as a worked example of checking a published number
before building on it.

No raw data is committed. `data/raw/` is gitignored and reproduced by the build
script; `data/processed/` holds the 12 MB of parquet the app reads.

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

That runs the app against the committed artifacts. To rebuild them from source,
which downloads roughly 130 MB and takes about three minutes:

```bash
pip install -r requirements-build.txt
python scripts/build_artifacts.py
```

Checks:

```bash
python -m pytest tests -q     # 13 tests: leakage, lag arithmetic, metrics
python scripts/selftest.py    # invariants over the committed backtest
python scripts/ui_test.py     # headless UI test (streamlit.testing)
```

`selftest.py` asserts three things that look like failures: that the intervals
are too narrow, that the model under-forecasts the top decile, and that a fold
exists where it loses outright to seasonal naive. All three are true, all three
are the most informative results here, and pinning them stops a later change
from quietly hiding any of them.

## Architecture

The app trains nothing and makes no network calls. `scripts/build_artifacts.py`
fetches the sources, runs the full backtest and writes parquet; `app.py` reads
those files under `@st.cache_data` and does light aggregation. A CP-SAT solve of
a few seconds can run live, as it does in the other demos in this portfolio. A
backtest over five years cannot, and a visitor should not wait for one.

The deployed app installs five packages. Everything needed to rebuild the
artifacts, including LightGBM and scikit-learn, lives in
`requirements-build.txt` and never reaches the server, which is what keeps the
cold start to a parquet read.

## Deploy

1. Push to GitHub (public).
2. [share.streamlit.io](https://share.streamlit.io) → **New app** → pick the
   repo, main file `app.py` → **Deploy**. No secrets are required.
3. The workflow in `.github/workflows/keepalive.yml` opens the app in headless
   Chromium every five hours so Community Cloud does not put it to sleep. A
   portfolio link a visitor has to wake up first is a portfolio link that costs
   you the opportunity.

## Project structure

```
app.py                      Streamlit UI, six sections behind a sidebar
gridcast/
  data.py                   sources, cleaning, the 2016 cut, calendar
  capacity.py               the operators' capacity map, deduplicated to areas
  features.py               origin-safe feature matrix and the horizon guard
  backtest.py               rolling-origin folds and the leakage assertion
  model.py                  seasonal Fourier ridge, LightGBM, residual quantiles
  baseline.py               persistence, seasonal naive, hour-of-week mean
  kpis.py                   the shared yardstick: error, bias, coverage, euros
scripts/build_artifacts.py  the expensive half, run locally, output committed
scripts/selftest.py         invariants over the committed results
scripts/ui_test.py          headless UI test
tests/                      leakage, lag arithmetic, hand-computed metrics
data/processed/             parquet the app reads (committed)
  backtest.parquet          every forecast, every baseline, the 80% band
  residual_quantiles.parquet  the error distribution per fold and horizon,
                            built from folds that had already closed. This is
                            what turns a point forecast into P(limit passed).
  series.parquet            the hourly series with calendar and temperature
  monthly_levels.parquet    the evidence behind the 2016 cut
  capaciteit.parquet        congestion status per supply area, derived summary
```

## From demo to production

- **Retrain on a trigger, not a calendar.** A quarterly refit was three months
  too slow in March 2020. The monitoring signal should drive the retrain.
- **Drift as a gate, not a chart.** Past the threshold the forecast waits for a
  human rather than being published with a band nobody re-checked.
- **Recalibrate the intervals on a rolling window** so a regime change widens
  them instead of silently invalidating them.
- **Weather as a forecast, with its own error budget**, scored against archived
  weather *forecasts* rather than against what the weather turned out to be.
- **Hierarchical forecasts**, national down to region and substation, reconciled
  so the levels agree.
- **Net load rather than gross**, once behind-the-meter solar is large enough to
  make that the operationally relevant series.

## Disclaimer

Educational portfolio project. Built on public data under CC-BY 4.0; not
affiliated with or endorsed by ENTSO-E, any transmission or distribution system
operator, or any market party. The euro figures are illustrative: the cost of a
forecast error is settled with the business, not chosen by whoever built the
model.
