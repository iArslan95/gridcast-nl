"""GridCast — hourly Dutch electricity load, one to seven days ahead.

The app trains nothing. Everything it shows was produced by
scripts/build_artifacts.py: a rolling-origin backtest with a quarterly refit
over 2018-2020, written to parquet and committed. Here we only read and
aggregate.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="GridCast — Dutch load forecasting",
                   page_icon="⚡", layout="wide")

DATA = pathlib.Path(__file__).parent / "data" / "processed"
TZ = "Europe/Amsterdam"
ACCENT = "#4338ca"

SERIES_STYLE = {
    "lgbm": dict(name="Gradient boosting", color=ACCENT, width=2.6, dash=None),
    "seasonal_naive": dict(name="Seasonal naive", color="#57534e", width=1.8,
                           dash=None),
    "hour_of_week": dict(name="Hour-of-week mean", color="#a8a29e", width=1.4,
                         dash="dot"),
    "fourier": dict(name="Seasonal Fourier", color="#0f766e", width=1.4,
                    dash="dash"),
    "persistence": dict(name="Persistence", color="#d6d3d1", width=1.2,
                        dash="dot"),
}
MODELS = ["lgbm", "seasonal_naive", "hour_of_week", "fourier"]

CSS = """
<style>
.block-container {padding-top: 1.4rem; max-width: 1500px;}
.hero {
  background: #ffffff;
  border: 1px solid #e7e5e4; border-left: 4px solid #4338ca;
  border-radius: 14px; padding: 26px 30px; margin-bottom: 18px;
}
.hero h1 {margin: 0; font-size: 1.8rem; color: #1c1917; letter-spacing: -0.01em;}
.hero p {margin: 8px 0 0; color: #78716c; font-size: 0.98rem; max-width: 95ch;}
[data-testid="stMetric"] {
  background: #ffffff; border: 1px solid #e7e5e4;
  border-radius: 12px; padding: 14px 16px;
  box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}
.note {
  background: #ffffff; border: 1px solid #e7e5e4; border-left: 3px solid #a8a29e;
  border-radius: 10px; padding: 13px 18px; margin: 6px 0 16px;
  color: #57534e; font-size: 0.93rem; max-width: 100ch;
}
.warn {
  background: #fffbeb; border: 1px solid #fde68a; border-left: 3px solid #b45309;
  border-radius: 10px; padding: 13px 18px; margin: 6px 0 16px;
  color: #78350f; font-size: 0.93rem; max-width: 100ch;
}
.footer {color: #a8a29e; font-size: 0.85rem; margin-top: 30px;}
.footer a {color: #a8a29e;}
.stTabs [data-baseweb="tab-list"] {gap: 8px; padding: 2px 0 10px;}
.stTabs [data-baseweb="tab"] {
  background: #ffffff; border: 1px solid #e7e5e4; border-radius: 10px;
  padding: 9px 18px; font-weight: 600; font-size: 0.97rem; color: #57534e;
}
.stTabs [data-baseweb="tab"]:hover {border-color: #4338ca; color: #4338ca;}
.stTabs [aria-selected="true"] {
  background: #4338ca; border-color: #4338ca; color: #ffffff;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display: none;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- loading ---
@st.cache_data(show_spinner="Reading the backtest…")
def load_backtest() -> pd.DataFrame:
    bt = pd.read_parquet(DATA / "backtest.parquet")
    local = bt["target"].dt.tz_convert(TZ)
    bt["target_local"] = local
    # Week buckets in local time; the period conversion has no use for the
    # offset and warns if it is left on.
    bt["week"] = local.dt.tz_localize(None).dt.to_period("W").dt.start_time
    bt["season"] = np.select(
        [local.dt.month.isin([12, 1, 2]), local.dt.month.isin([6, 7, 8])],
        ["winter", "summer"], default="spring / autumn")
    bt["period"] = np.where(local.dt.hour.between(7, 19), "peak 07-19", "off-peak")
    return bt


@st.cache_data
def load_series() -> pd.DataFrame:
    s = pd.read_parquet(DATA / "series.parquet")
    s = s.rename(columns={s.columns[0]: "utc_timestamp"})
    s["local"] = s["utc_timestamp"].dt.tz_convert(TZ)
    return s


@st.cache_data
def load_meta() -> dict:
    return json.loads((DATA / "meta.json").read_text(encoding="utf-8"))


@st.cache_data
def load_monthly_levels() -> pd.DataFrame:
    return pd.read_parquet(DATA / "monthly_levels.parquet")


def mape(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs((p - y) / y)) * 100)


@st.cache_data
def by_horizon(_key: str = "v1") -> pd.DataFrame:
    bt = load_backtest()
    rows = []
    for h, d in bt.groupby("h"):
        row = {"h": h}
        for m in MODELS:
            e = d[f"pred_{m}"] - d["y"]
            row[f"mape_{m}"] = float(np.mean(np.abs(e / d["y"])) * 100)
            row[f"mae_{m}"] = float(np.mean(np.abs(e)))
            row[f"bias_{m}"] = float(np.mean(e))
        s = d[d["lo80"].notna()]
        row["coverage"] = float(
            np.mean((s["y"] >= s["lo80"]) & (s["y"] <= s["hi80"])) * 100)
        row["width"] = float(np.mean(s["hi80"] - s["lo80"]))
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data
def by_week(_key: str = "v1") -> pd.DataFrame:
    bt = load_backtest()
    rows = []
    for wk, d in bt.groupby("week"):
        s = d[d["lo80"].notna()]
        rows.append({
            "week": wk,
            "mape_lgbm": mape(d["y"], d["pred_lgbm"]),
            "mape_seasonal_naive": mape(d["y"], d["pred_seasonal_naive"]),
            "bias": float(np.mean(d["pred_lgbm"] - d["y"])),
            "coverage": float(np.mean((s["y"] >= s["lo80"])
                                      & (s["y"] <= s["hi80"])) * 100)
            if len(s) else np.nan,
        })
    return pd.DataFrame(rows)


@st.cache_data
def by_segment(column: str) -> pd.DataFrame:
    bt = load_backtest()
    rows = []
    for key, d in bt.groupby(column):
        row = {"segment": key, "hours": len(d)}
        for m in MODELS:
            row[SERIES_STYLE[m]["name"]] = mape(d["y"], d[f"pred_{m}"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("hours", ascending=False)


def layout(fig: go.Figure, ytitle: str, xtitle: str, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height, template="simple_white",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        hovermode="x unified",
        font=dict(color="#44403c", size=12),
        plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(title=ytitle, gridcolor="#f0efee")
    fig.update_xaxes(title=xtitle)
    return fig


meta = load_meta()
overall = meta["overall"]
summary = meta["series"]

st.markdown(f"""
<div class="hero">
  <h1>⚡ GridCast — Dutch electricity load, 1 to 7 days ahead</h1>
  <p>Hourly national load for the Netherlands, forecast up to 168 hours out and
  scored by a rolling-origin backtest over {meta['test_start'][:4]} to 2020:
  {len(meta['folds'])} quarterly refits, {int(sum(f['forecasts'] for f in meta['folds'])):,}
  forecasts, every one of them made with data that existed at the time. Every
  number on this page sits next to the baseline it has to beat.</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Model MAPE", f"{overall['lgbm']['mape']:.2f}%",
          f"{overall['lgbm']['mape'] - overall['seasonal_naive']['mape']:+.2f} pp "
          f"vs seasonal naive", delta_color="inverse")
c2.metric("Seasonal naive MAPE", f"{overall['seasonal_naive']['mape']:.2f}%",
          "the baseline to beat", delta_color="off")
c3.metric("Mean absolute error", f"{overall['lgbm']['mae']:,.0f} MW",
          f"{overall['lgbm']['bias']:+,.0f} MW bias", delta_color="off")
c4.metric("80% interval, realised coverage", f"{meta['coverage80']:.1f}%",
          f"{meta['coverage80'] - 80:.1f} pp vs nominal", delta_color="inverse")

tab1, tab2, tab3, tab4 = st.tabs(
    ["The problem", "Model and backtest", "What it is worth", "Monitoring"])

# ------------------------------------------------------------- 1. problem ---
with tab1:
    st.markdown("""
    Forecast too low and there is not enough capacity contracted for the hour;
    forecast too high and capacity is paid for and not used. Those two mistakes
    do not cost the same amount, which is why this page ends in euros rather
    than in percentages.
    """)

    st.subheader("The series, and why it starts in 2016")
    lv = load_monthly_levels()
    fig = go.Figure()
    years = [c for c in lv.columns if c != "month"]
    for y in years:
        excluded = str(y) == "2015"
        fig.add_trace(go.Scatter(
            x=lv["month"], y=lv[y], name=str(y), mode="lines+markers",
            line=dict(color="#b45309" if excluded else ACCENT,
                      width=2.6 if excluded else 1.4,
                      dash="dash" if excluded else None),
            opacity=1.0 if excluded else 0.45))
    fig = layout(fig, "mean load (MW)", "calendar month")
    fig.update_xaxes(tickmode="linear", dtick=1)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="note">
    Every month of 2015 (amber) sits 1.7 to 2.0 GW below the same month of every
    later year, while 2016 through 2019 lie on top of each other. Demand does not
    move 16% in one January and then hold still for four years: this is a change
    in what was reported, not in what was consumed. The series therefore starts
    at 2016-01-01, which costs a year of history and buys a series that measures
    one thing throughout.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("The shape of a day, and what a holiday does to it")
    s = load_series()
    s["dow"] = s["local"].dt.dayofweek
    s["hr"] = s["local"].dt.hour
    groups = {
        "Working day": s[(s["dow"] < 5) & ~s["is_holiday"]],
        "Weekend": s[s["dow"] >= 5],
        "Public holiday": s[s["is_holiday"]],
    }
    fig = go.Figure()
    for (label, part), color in zip(groups.items(), [ACCENT, "#57534e", "#b45309"]):
        prof = part.groupby("hr")["load"].mean()
        fig.add_trace(go.Scatter(x=prof.index, y=prof.values, name=label,
                                 mode="lines", line=dict(width=2.4, color=color)))
    fig = layout(fig, "mean load (MW)", "hour of day (local time)")
    fig.update_xaxes(dtick=3)
    st.plotly_chart(fig, use_container_width=True)
    work = s[(s["dow"] < 5) & ~s["is_holiday"]]["load"].mean()
    hol = s[s["is_holiday"]]["load"].mean()
    st.markdown(f"""
    <div class="note">
    A public holiday runs {hol / work - 1:.1%} below a working day and keeps the
    weekend's flatter shape. That is a calendar fact, knowable years ahead, and
    it is where a model earns most of its advantage over "same hour last week":
    last week was a Tuesday, this Tuesday is Ascension Day.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Temperature, which this model does not use")
    daily = (s.set_index("local").resample("D")
             .agg({"load": "mean", "temp_c": "mean", "dow": "first",
                   "is_holiday": "max"}).dropna())
    daily = daily[(daily["dow"] < 5) & ~daily["is_holiday"].astype(bool)]
    fig = go.Figure(go.Scattergl(
        x=daily["temp_c"], y=daily["load"], mode="markers",
        marker=dict(size=5, color=ACCENT, opacity=0.35),
        name="working day", hovertemplate="%{x:.1f} °C, %{y:,.0f} MW<extra></extra>"))
    binned = daily.groupby(pd.cut(daily["temp_c"], np.arange(-6, 30, 2)),
                           observed=True)["load"].mean()
    centres = [iv.mid for iv in binned.index]
    fig.add_trace(go.Scatter(x=centres, y=binned.values, mode="lines+markers",
                             line=dict(color="#b45309", width=2.6),
                             name="2 °C bin mean"))
    fig = layout(fig, "daily mean load (MW)", "daily mean temperature (°C)")
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="note">
    Clear and U-shaped: heating below roughly 15 °C, cooling above roughly 20 °C,
    and a flat valley in between. It explains a large share of the day-to-day
    variation, and it is still left out of the model on purpose. Using
    temperature at t+168 means feeding in a weather forecast, whose own error is
    not measured anywhere in this backtest. A model scored on temperatures that
    actually happened reports an accuracy nobody will reproduce in operation.
    The honest options are to leave it out, or to build a second error budget for
    the weather. This demo takes the first.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("A published forecast that does not survive being checked"):
        st.markdown("""
        The same source ships a day-ahead load forecast alongside the
        realisations, and it was the obvious hard baseline for this project.
        It does not hold up. Scored against the actuals in the very same file:
        """)
        fc = s.dropna(subset=["tso_day_ahead"]).copy()
        fc["year"] = fc["local"].dt.year
        rows = []
        for y, d in fc.groupby("year"):
            e = d["tso_day_ahead"] - d["load"]
            rows.append({"year": int(y), "hours": len(d),
                         "MAPE %": mape(d["load"], d["tso_day_ahead"]),
                         "bias MW": float(e.mean()),
                         "bias %": float(e.mean() / d["load"].mean() * 100)})
        st.dataframe(pd.DataFrame(rows).round(2), hide_index=True,
                     use_container_width=True)
        st.markdown("""
        <div class="warn">
        A systematic error that runs about +5% for three years and then flips to
        −14% is not a forecast that got worse, it is two columns measured on
        different bases. On Christmas Day 2018 the file predicts 19,423 MW
        against 12,139 MW realised. So it is not used as a baseline here, and it
        is on this page instead: checking a published number against the
        realisations in the same file takes an afternoon, and skipping that step
        is how a whole project ends up anchored to something that was never
        comparable.
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------ 2. backtest ---
with tab2:
    st.subheader("How the accuracy was measured")
    st.markdown("""
    Walk forward, never sideways. The model is refit every quarter on
    everything up to that cut and on nothing after it, then forecasts every six
    hours for the following quarter, 168 hours out each time. Those forecasts
    are scored once and never revisited. There is no random train/test split
    anywhere in the repository: shuffling hours lets a model interpolate
    between the hour before and the hour after and score beautifully while
    being useless.
    """)
    folds = pd.DataFrame(meta["folds"])
    st.dataframe(folds, hide_index=True, use_container_width=True)
    st.markdown("""
    <div class="note">
    Training rows stop a full 168 hours before each cut, not at the cut. An
    origin closer than the longest horizon would need a target the model is not
    allowed to see, and dropping only those rows would quietly delete the long
    horizons from the training set. <code>tests/test_no_leakage.py</code> checks
    the property rather than the intention: it replaces every value after the
    origin with nonsense, rebuilds the features and requires them to come out
    bit for bit identical.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Error by horizon")
    h = by_horizon()
    fig = go.Figure()
    for m in MODELS:
        sty = SERIES_STYLE[m]
        fig.add_trace(go.Scatter(
            x=h["h"], y=h[f"mape_{m}"], name=sty["name"], mode="lines",
            line=dict(color=sty["color"], width=sty["width"], dash=sty["dash"])))
    fig = layout(fig, "MAPE (%)", "horizon h (hours ahead)")
    fig.update_xaxes(dtick=24)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"""
    <div class="note">
    The model runs from {h['mape_lgbm'].iloc[0]:.2f}% one hour out to
    {h['mape_lgbm'].iloc[-1]:.2f}% a week out. That is a small rise, and it is
    the most interesting result here: at national level a week ahead is barely
    harder than an hour ahead, because almost all the signal is calendar shape
    that is equally knowable at both. It also explains the two flat lines.
    Seasonal naive uses the same value at every horizon, and the Fourier model
    uses no recent load at all, so neither has anything left to lose as the
    horizon grows. Only the boosted model holds recent information, and only it
    has a curve that slopes.
    <br><br>
    The forecasts are issued at 00, 06, 12 and 18 UTC rather than once a day.
    With a single daily origin, h = 24, 48 … 168 would all land at midnight,
    the steadiest hour of the day, and this chart would be measuring the clock
    instead of the horizon. Four origins do not remove that entirely: each
    horizon still lands on a fixed set of four clock hours, and the set shifts
    with h, which is the small ripple in these lines. The clean comparison is h
    against h + 24, where the hours match exactly and only the lead time
    differs. Done that way a day of extra lead time costs the boosted model
    0.04 pp and costs the two flat models at most 0.01 pp, which is the
    statement the chart is making.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Bias, kept apart from accuracy")
    fig = go.Figure()
    for m in MODELS:
        sty = SERIES_STYLE[m]
        fig.add_trace(go.Scatter(
            x=h["h"], y=h[f"bias_{m}"], name=sty["name"], mode="lines",
            line=dict(color=sty["color"], width=sty["width"], dash=sty["dash"])))
    fig.add_hline(y=0, line=dict(color="#a8a29e", width=1))
    fig = layout(fig, "mean error (MW), positive = forecast too high",
                 "horizon h (hours ahead)", height=320)
    fig.update_xaxes(dtick=24)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="note">
    Two models can share a MAE and mean completely different things. Errors
    scattered around zero are noise, and you hold a reserve against them.
    Errors that lean one way every hour are a systematic shortfall, and you fix
    the model instead. The Fourier model leans high by roughly 250 MW because it
    has no way to notice that the level has moved; the boosted model does not,
    until 2020, which the monitoring tab picks up.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Do the intervals mean what they say?")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["h"], y=h["coverage"], name="realised coverage",
                             mode="lines", line=dict(color=ACCENT, width=2.6)))
    fig.add_hline(y=80, line=dict(color="#b45309", width=1.6, dash="dash"),
                  annotation_text="80% nominal", annotation_position="top left")
    fig = layout(fig, "share of realisations inside the band (%)",
                 "horizon h (hours ahead)", height=320)
    fig.update_yaxes(range=[50, 95])
    fig.update_xaxes(dtick=24)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"""
    <div class="warn">
    An 80% interval that contains {meta['coverage80']:.1f}% of realisations is
    too narrow, and this is the number I would put in front of a client first.
    The bands are the empirical quantiles of the errors the model actually made
    in earlier folds, so they describe the past well and only hold while the
    future keeps behaving like it. In 2020 it stopped, coverage fell to 55%, and
    a band nobody re-checked would still have been drawn at the same width. The
    first fold carries no interval at all: at that point no out-of-sample error
    existed to learn a width from, and borrowing one from later folds would be
    reading the future.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Where it fails, which is more useful than where it works")
    left, right = st.columns(2)
    with left:
        st.caption("MAPE by day type")
        st.dataframe(by_segment("segment").round(2), hide_index=True,
                     use_container_width=True)
        st.caption("MAPE by season")
        st.dataframe(by_segment("season").round(2), hide_index=True,
                     use_container_width=True)
    with right:
        st.caption("MAPE by time of day")
        st.dataframe(by_segment("period").round(2), hide_index=True,
                     use_container_width=True)
        st.caption("MAPE by fold")
        bt = load_backtest()
        rows = []
        for f, d in bt.groupby("fold"):
            rows.append({"fold": int(f),
                         "through": str(d["target"].min().date()),
                         "Gradient boosting": mape(d["y"], d["pred_lgbm"]),
                         "Seasonal naive": mape(d["y"], d["pred_seasonal_naive"])})
        st.dataframe(pd.DataFrame(rows).round(2), hide_index=True,
                     use_container_width=True)
    st.markdown("""
    <div class="note">
    Public holidays are where the model is worth having: it roughly halves the
    error of "same hour last week", which has no way of knowing that today is
    not a normal Thursday. Peak hours are twice as hard as the night, so a
    single headline MAPE flatters the hours nobody worries about. And in the
    final fold the boosted model is <em>beaten</em> by seasonal naive, which is
    the whole 2020 story and the reason the monitoring tab exists.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("The three models")
    st.markdown("""
| Model | What it sees | Why it is here |
|---|---|---|
| **Seasonal naive** | the same hour one week ago | On a series this regular it is a genuine competitor, not a straw man. Anything that cannot beat it does not deserve to run. |
| **Seasonal Fourier** | Fourier terms for the daily, weekly and yearly cycle, holiday flags, a slow trend. Ridge on log load, no recent load at all | The pure calendar view. Its flat error curve is the reference against which the boosted model's slope means something. Fitted on logs because holidays and seasons act proportionally; the back-transform carries a half-variance correction so the log fit does not introduce a bias of its own. |
| **Gradient boosting** | origin lags (0 to 168 h), rolling means, the same hour one and two weeks before the target, calendar, and the horizon itself | One model for all 168 horizons, with h as a feature, so the shape of the error-versus-horizon curve is something the model produces rather than something the setup imposes. |

No deep learning and no Prophet. Neither would earn its place on 41,640 hourly
observations, and both would cost the ability to say why a number came out the
way it did.
    """)

# --------------------------------------------------------------- 3. value ---
with tab3:
    st.subheader("From a percentage to a number someone can act on")
    st.markdown("""
    A MAPE does not tell anyone whether to buy the model. Put a price on the two
    directions of error and it does. Errors here are in MW held for one hour,
    which is MWh, so the two inputs below are euros per MWh of error.
    """)
    c1, c2, c3 = st.columns([1, 1, 2])
    products = {"Day-ahead (h = 24)": 24, "Two days out (h = 48)": 48,
                "Week ahead (h = 168)": 168}
    product = c1.selectbox("Forecast product", list(products), index=0)
    eur_under = c2.number_input("€ per MWh under-forecast", 0.0, 5000.0, 180.0,
                                10.0, help="Capacity that turned out to be short.")
    eur_over = c2.number_input("€ per MWh over-forecast", 0.0, 5000.0, 60.0, 10.0,
                               help="Capacity contracted and not used.")
    c3.markdown("""<div class='note'>One horizon at a time, on purpose. The
    backtest holds 168 forecasts for every hour, one per horizon, and summing
    across them would charge the same hour of error dozens of times over. A
    price belongs to a product: the day-ahead forecast is a different purchase
    from the week-ahead one.<br><br>
    The defaults make under-forecasting three times as expensive as
    over-forecasting. That ratio is a placeholder, not a market price. Move both
    fields.</div>""", unsafe_allow_html=True)

    bt = load_backtest()
    bt = bt[bt["h"] == products[product]]
    y = bt["y"].to_numpy(dtype=float)
    hours = len(y)
    HOURS_PER_YEAR = 8766.0

    def annual_cost(pred: np.ndarray) -> float:
        e = pred - y
        return float((np.maximum(-e, 0).sum() * eur_under
                      + np.maximum(e, 0).sum() * eur_over)
                     / hours * HOURS_PER_YEAR)

    cost = {m: annual_cost(bt[f"pred_{m}"].to_numpy(dtype=float)) for m in MODELS}
    saving = cost["seasonal_naive"] - cost["lgbm"]

    avoided = float(
        (np.abs(bt["pred_seasonal_naive"].to_numpy(dtype=float) - y).sum()
         - np.abs(bt["pred_lgbm"].to_numpy(dtype=float) - y).sum())
        / hours * HOURS_PER_YEAR)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Seasonal naive, cost of error",
              f"€ {cost['seasonal_naive'] / 1e6:,.0f} M / yr")
    k2.metric("Gradient boosting, cost of error",
              f"€ {cost['lgbm'] / 1e6:,.0f} M / yr")
    k3.metric("Difference", f"€ {saving / 1e6:,.0f} M / yr",
              f"{saving / cost['seasonal_naive']:.1%} lower")
    k4.metric("Forecast error avoided", f"{avoided / 1000:,.0f} GWh / yr",
              "no price attached", delta_color="off")

    ratios = np.array([1, 1.5, 2, 3, 4, 6, 8, 10], dtype=float)
    curve = []
    for r in ratios:
        under, over = r * eur_over, eur_over
        e_m = bt["pred_lgbm"].to_numpy(dtype=float) - y
        e_b = bt["pred_seasonal_naive"].to_numpy(dtype=float) - y
        cm = (np.maximum(-e_m, 0).sum() * under + np.maximum(e_m, 0).sum() * over)
        cb = (np.maximum(-e_b, 0).sum() * under + np.maximum(e_b, 0).sum() * over)
        curve.append((cb - cm) / hours * HOURS_PER_YEAR)
    fig = go.Figure(go.Scatter(x=ratios, y=curve, mode="lines+markers",
                               line=dict(color=ACCENT, width=2.6),
                               name="annual saving"))
    fig = layout(fig, "saving against seasonal naive (€/yr)",
                 "cost of under-forecasting ÷ cost of over-forecasting", height=320)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"""
    <div class="note">
    The saving is not a fixed number, it moves with how asymmetric the cost is.
    The more expensive under-forecasting becomes relative to over-forecasting,
    the more a model that leans slightly high is worth, which is why this line
    slopes at all.
    <br><br>
    The fourth figure carries no price at all, which is why it is there. Over a
    year the model removes that many GWh of absolute forecast error compared
    with seasonal naive, and that number stands whatever anyone decides an hour
    of error is worth. Everything to its left is that same fact with a price
    stapled on.
    <br><br>
    Read the levels with care. This is the entire Dutch grid at a mean load of
    {summary['load_mean'] / 1000:.1f} GW, so a {overall['lgbm']['mape']:.1f}%
    error is roughly {overall['lgbm']['mae']:,.0f} MW every hour of the year and
    any per-MWh price turns that into a large number by construction. It also
    charges every MWh of error at the full rate, which no settlement regime
    does. The figure that survives those objections is the
    <em>difference</em> between two forecasts scored on identical hours, not the
    level of either. And the cost function itself is settled with the business
    rather than chosen by whoever built the model, so treat both inputs as
    placeholders until someone with a budget has replaced them.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------- 4. monitoring ---
with tab4:
    st.subheader("What this would have looked like on a wall")
    wk = by_week()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wk["week"], y=wk["mape_seasonal_naive"],
                             name="Seasonal naive", mode="lines",
                             line=dict(color="#a8a29e", width=1.4)))
    fig.add_trace(go.Scatter(x=wk["week"], y=wk["mape_lgbm"],
                             name="Gradient boosting", mode="lines",
                             line=dict(color=ACCENT, width=2.4)))
    threshold = float(wk["mape_lgbm"].iloc[:52].mean() + 2 * wk["mape_lgbm"].iloc[:52].std())
    fig.add_hline(y=threshold, line=dict(color="#b45309", width=1.4, dash="dash"),
                  annotation_text=f"alert at {threshold:.1f}%",
                  annotation_position="top left")
    fig = layout(fig, "MAPE per week (%)", "week")
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure(go.Bar(x=wk["week"], y=wk["bias"],
                           marker_color=np.where(wk["bias"] > 0, ACCENT, "#b45309"),
                           name="weekly bias"))
    fig.add_hline(y=0, line=dict(color="#a8a29e", width=1))
    fig = layout(fig, "mean error per week (MW)", "week", height=280)
    st.plotly_chart(fig, use_container_width=True)

    flagged = wk[wk["mape_lgbm"] > threshold]
    in_2020 = int((flagged["week"].dt.year == 2020).sum())
    first_2020 = flagged[flagged["week"].dt.year == 2020]["week"].min()
    worst = wk.loc[wk["bias"].idxmin()]
    st.markdown(f"""
    <div class="warn">
    The alert line sits two standard deviations above the first year's weekly
    error, the kind of rule a team actually ships. It fires in {len(flagged)} of
    {len(wk)} weeks, and they are not scattered: {in_2020} of them are in 2020,
    running almost without a break from the week of
    {first_2020.strftime('%d %B %Y')} onwards. Demand fell roughly a tenth in a
    fortnight and every model trained on 2016 to 2019 carried on forecasting the
    country that no longer existed. The weekly bias turns positive and stays
    there, which is what separates a broken model from a noisy one: noise
    alternates sign, a regime change does not. A quarterly retrain is far too
    slow for that, and the answer is not a better model but a gate. Past the
    threshold, the forecast waits for a human.
    <br><br>
    The largest single miss points the other way, and at me. In the week of
    {worst['week'].strftime('%d %B %Y')} the model ran
    {abs(worst['bias']):,.0f} MW <em>under</em> realisation, the only sustained
    negative bias in the record. That was the hottest week in the whole series,
    peaking at 33 °C, with six of the five years' eight hottest August days in
    it. Load climbed to 13.3 GW against 11.5 GW the week before, and a model
    that is not allowed to look at temperature had no way to see it coming. The
    decision to leave weather out buys an honest horizon and costs exactly this,
    and a monitoring page that hid it would be worth nothing.
    </div>
    """, unsafe_allow_html=True)

    st.caption("Weeks above the alert threshold")
    st.dataframe(
        flagged.assign(week=flagged["week"].dt.strftime("%Y-%m-%d"))
        .rename(columns={"mape_lgbm": "model MAPE %",
                         "mape_seasonal_naive": "seasonal naive MAPE %",
                         "bias": "bias MW", "coverage": "80% coverage %"})
        .round(2), hide_index=True, use_container_width=True)

st.markdown("""
<div class="footer">
Ismail Arslan ·
<a href="https://ismailarslan.tech">ismailarslan.tech</a> ·
<a href="mailto:contact@ismailarslan.tech">contact@ismailarslan.tech</a> ·
<a href="https://linkedin.com/in/iqzarslan">linkedin.com/in/iqzarslan</a><br>
Load data: Open Power System Data, time series 2020-10-06 (CC-BY 4.0), originally
ENTSO-E Transparency. Temperature: Open-Meteo historical archive (CC-BY 4.0).
Educational portfolio project; not affiliated with any grid operator.
</div>
""", unsafe_allow_html=True)
