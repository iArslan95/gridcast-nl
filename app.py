"""GridCast: hourly Dutch electricity load, one to seven days ahead.

The app trains nothing. Everything it shows was produced by
scripts/build_artifacts.py: a rolling-origin backtest with a quarterly refit
over 2018-2020, written to parquet and committed. Here we only read and
aggregate. Interface copy is Dutch; code and comments stay English.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="GridCast: landelijke elektriciteitsvraag",
                   page_icon="⚡", layout="wide")

# Streamlit replaced use_container_width with width="stretch" in 1.49. The host
# resolves its own Streamlit version, so pick the spelling that exists rather
# than pinning the whole app to one side of that change.
_VERSION = tuple(int(p) for p in st.__version__.split(".")[:2])
WIDE = {"width": "stretch"} if _VERSION >= (1, 49) else {"use_container_width": True}

DATA = pathlib.Path(__file__).parent / "data" / "processed"
TZ = "Europe/Amsterdam"
ACCENT = "#4338ca"
HOURS_PER_YEAR = 8766.0

NAMES = {
    "lgbm": "Gradient boosting",
    "seasonal_naive": "Seizoensnaief",
    "hour_of_week": "Uur-van-de-week gemiddelde",
    "fourier": "Seizoens-Fourier",
    "persistence": "Persistentie",
}
STYLE = {
    "lgbm": dict(color=ACCENT, width=2.6, dash=None),
    "seasonal_naive": dict(color="#57534e", width=1.8, dash=None),
    "hour_of_week": dict(color="#a8a29e", width=1.4, dash="dot"),
    "fourier": dict(color="#0f766e", width=1.4, dash="dash"),
    "persistence": dict(color="#d6d3d1", width=1.2, dash="dot"),
}
MODELS = ["lgbm", "seasonal_naive", "hour_of_week", "fourier"]
MAANDEN = {1: "januari", 2: "februari", 3: "maart", 4: "april", 5: "mei",
           6: "juni", 7: "juli", 8: "augustus", 9: "september", 10: "oktober",
           11: "november", 12: "december"}

CSS = """
<style>
.block-container {padding-top: 1.4rem; max-width: 1500px;}
.hero {
  background: #ffffff;
  border: 1px solid #e7e5e4; border-left: 4px solid #4338ca;
  border-radius: 14px; padding: 26px 30px; margin-bottom: 18px;
}
.hero h1 {margin: 0; font-size: 1.75rem; color: #1c1917; letter-spacing: -0.01em;}
.hero h2 {margin: 4px 0 0; font-size: 1.02rem; font-weight: 600; color: #4338ca;}
.hero p {margin: 10px 0 0; color: #78716c; font-size: 0.98rem; max-width: 95ch;}
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


def nl(x: float, decimals: int = 0, unit: str = "") -> str:
    """Dutch number formatting: dot for thousands, comma for decimals."""
    s = f"{x:,.{decimals}f}".replace(",", " ").replace(".", ",")
    s = s.replace(" ", ".")
    return f"{s}{unit}"


# ---------------------------------------------------------------- loading ---
@st.cache_data(show_spinner="Backtest inlezen…")
def load_backtest() -> pd.DataFrame:
    bt = pd.read_parquet(DATA / "backtest.parquet")
    local = bt["target"].dt.tz_convert(TZ)
    bt["week"] = local.dt.tz_localize(None).dt.to_period("W").dt.start_time
    bt["seizoen"] = np.select(
        [local.dt.month.isin([12, 1, 2]), local.dt.month.isin([6, 7, 8])],
        ["winter", "zomer"], default="voor- en najaar")
    bt["dagdeel"] = np.where(local.dt.hour.between(7, 19), "piek 07-19", "dal")
    bt["dagsoort"] = bt["segment"].map(
        {"working day": "werkdag", "weekend": "weekend",
         "public holiday": "feestdag"})
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


@st.cache_data
def load_quantiles() -> pd.DataFrame:
    return pd.read_parquet(DATA / "residual_quantiles.parquet")


def mape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs((p - y) / y)) * 100)


@st.cache_data
def by_horizon(_v: str = "1") -> pd.DataFrame:
    bt = load_backtest()
    rows = []
    for h, d in bt.groupby("h"):
        row = {"h": int(h)}
        for m in MODELS:
            e = d[f"pred_{m}"] - d["y"]
            row[f"mape_{m}"] = float(np.mean(np.abs(e / d["y"])) * 100)
            row[f"bias_{m}"] = float(np.mean(e))
        s = d[d["lo80"].notna()]
        row["dekking"] = float(
            np.mean((s["y"] >= s["lo80"]) & (s["y"] <= s["hi80"])) * 100)
        rows.append(row)
    return pd.DataFrame(rows)


@st.cache_data
def by_week(_v: str = "1") -> pd.DataFrame:
    bt = load_backtest()
    rows = []
    for wk, d in bt.groupby("week"):
        rows.append({"week": wk,
                     "mape_lgbm": mape(d["y"], d["pred_lgbm"]),
                     "mape_seasonal_naive": mape(d["y"], d["pred_seasonal_naive"]),
                     "bias": float(np.mean(d["pred_lgbm"] - d["y"]))})
    return pd.DataFrame(rows)


@st.cache_data
def by_segment(column: str) -> pd.DataFrame:
    bt = load_backtest()
    rows = []
    for key, d in bt.groupby(column):
        row = {"segment": str(key), "uren": len(d)}
        for m in MODELS:
            row[NAMES[m]] = mape(d["y"], d[f"pred_{m}"])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("uren", ascending=False)


@st.cache_data
def exceedance(h: int, capacity: float) -> pd.DataFrame:
    """Turn each point forecast into a probability that the limit is passed.

    P(y > C) = P(residual > C - forecast), read from the empirical residual
    distribution of that horizon, built from folds that had already closed.
    Fold 1 has no such distribution and is left out, exactly as with the
    intervals.
    """
    bt = load_backtest()
    q = load_quantiles()
    d = bt[(bt["h"] == h) & (bt["fold"] > 1)][
        ["fold", "target", "y", "pred_lgbm", "pred_seasonal_naive"]].copy()

    kans = np.empty(len(d), dtype=float)
    folds = d["fold"].to_numpy()
    for fold in np.unique(folds):
        qq = q[(q["fold"] == fold) & (q["h"] == h)].sort_values("resid")
        mask = folds == fold
        nodig = capacity - d.loc[mask, "pred_lgbm"].to_numpy(float)
        cdf = np.interp(nodig, qq["resid"].to_numpy(float),
                        qq["q"].to_numpy(float), left=0.0, right=1.0)
        kans[mask] = 1.0 - cdf

    d["kans"] = kans
    d["overschreden"] = d["y"] > capacity
    return d


def alarm_stats(d: pd.DataFrame, alarm: np.ndarray) -> dict:
    raak = int((alarm & d["overschreden"]).sum())
    vals = int((alarm & ~d["overschreden"]).sum())
    gemist = int((~alarm & d["overschreden"]).sum())
    maanden = max((d["target"].max() - d["target"].min()).days / 30.44, 1e-9)
    return {"raak": raak, "vals": vals, "gemist": gemist,
            "recall": raak / max(raak + gemist, 1) * 100,
            "precisie": raak / max(raak + vals, 1) * 100,
            "vals_pm": vals / maanden}


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
reeks = meta["series"]
n_forecasts = int(sum(f["forecasts"] for f in meta["folds"]))

st.markdown(f"""
<div class="hero">
  <h1>⚡ GridCast</h1>
  <h2>Landelijke elektriciteitsvraag per uur, 1 tot 7 dagen vooruit</h2>
  <p>Voorspellen is het makkelijke deel. De vraag die telt is wanneer je je eigen
  getal nog mag geloven. Dat wordt hier gemeten met een rolling origin backtest
  over {meta['test_start'][:4]} tot 2020: {len(meta['folds'])}
  kwartaalhertrainingen, {nl(n_forecasts)} voorspellingen, elk gemaakt met
  uitsluitend data die op dat moment bestond. Naast elk cijfer staat de baseline
  die het moet verslaan.</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("MAPE model", nl(overall["lgbm"]["mape"], 2, "%"),
          nl(overall["lgbm"]["mape"] - overall["seasonal_naive"]["mape"], 2)
          + " pp t.o.v. seizoensnaief", delta_color="inverse")
c2.metric("MAPE seizoensnaief", nl(overall["seasonal_naive"]["mape"], 2, "%"),
          "de baseline", delta_color="off")
c3.metric("Gemiddelde absolute fout", nl(overall["lgbm"]["mae"], 0, " MW"),
          nl(overall["lgbm"]["bias"], 0, " MW bias"), delta_color="off")
c4.metric("80%-interval, werkelijke dekking", nl(meta["coverage80"], 1, "%"),
          nl(meta["coverage80"] - 80, 1) + " pp t.o.v. nominaal",
          delta_color="inverse")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Het probleem", "Model en backtest", "Van voorspelling naar besluit",
     "Monitoring"])

# ------------------------------------------------------------- 1. probleem ---
with tab1:
    st.markdown("""
    Te laag voorspellen betekent dat er te weinig capaciteit klaarstaat. Te hoog
    voorspellen betekent capaciteit betalen die ongebruikt blijft. Die twee
    fouten kosten niet hetzelfde, en daarom is een accuratessepercentage nog geen
    antwoord.
    """)

    st.subheader("De reeks, en waarom hij in 2016 begint")
    lv = load_monthly_levels()
    fig = go.Figure()
    for jaar in [c for c in lv.columns if c != "month"]:
        uit = str(jaar) == "2015"
        fig.add_trace(go.Scatter(
            x=lv["month"], y=lv[jaar], name=str(jaar), mode="lines+markers",
            line=dict(color="#b45309" if uit else ACCENT,
                      width=2.6 if uit else 1.4,
                      dash="dash" if uit else None),
            opacity=1.0 if uit else 0.45))
    fig = layout(fig, "gemiddelde belasting (MW)", "kalendermaand")
    fig.update_xaxes(tickmode="linear", dtick=1)
    st.plotly_chart(fig, **WIDE)
    st.markdown("""
    <div class="note">
    Elke maand van 2015 (amber) ligt 1,7 tot 2,0 GW onder dezelfde maand van elk
    later jaar, terwijl 2016 tot en met 2019 op elkaar liggen. Verbruik
    verspringt niet 16% in één januari om daarna vier jaar stil te staan. Dit is
    een wijziging in wat er gerapporteerd werd, niet in wat er verbruikt werd. De
    reeks start daarom op 1 januari 2016. Dat kost een jaar historie en levert
    een reeks op die overal hetzelfde meet.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("De vorm van een dag, en wat een feestdag ermee doet")
    s = load_series()
    s["dow"] = s["local"].dt.dayofweek
    s["hr"] = s["local"].dt.hour
    groepen = {
        "Werkdag": s[(s["dow"] < 5) & ~s["is_holiday"]],
        "Weekend": s[s["dow"] >= 5],
        "Feestdag": s[s["is_holiday"]],
    }
    fig = go.Figure()
    for (label, deel), kleur in zip(groepen.items(), [ACCENT, "#57534e", "#b45309"]):
        prof = deel.groupby("hr")["load"].mean()
        fig.add_trace(go.Scatter(x=prof.index, y=prof.values, name=label,
                                 mode="lines", line=dict(width=2.4, color=kleur)))
    fig = layout(fig, "gemiddelde belasting (MW)", "uur van de dag (lokale tijd)")
    fig.update_xaxes(dtick=3)
    st.plotly_chart(fig, **WIDE)
    werk = s[(s["dow"] < 5) & ~s["is_holiday"]]["load"].mean()
    feest = s[s["is_holiday"]]["load"].mean()
    st.markdown(f"""
    <div class="note">
    Een feestdag ligt {nl(abs(feest / werk - 1) * 100, 1)}% onder een werkdag en
    houdt de vlakkere vorm van het weekend. Dat is een kalenderfeit, jaren
    vooruit bekend, en precies daar verdient een model zijn meerwaarde boven
    "hetzelfde uur vorige week": vorige week was het een gewone donderdag, deze
    donderdag is het Hemelvaart.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Temperatuur, die dit model niet gebruikt")
    dag = (s.set_index("local").resample("D")
           .agg({"load": "mean", "temp_c": "mean", "dow": "first",
                 "is_holiday": "max"}).dropna())
    dag = dag[(dag["dow"] < 5) & ~dag["is_holiday"].astype(bool)]
    fig = go.Figure(go.Scattergl(
        x=dag["temp_c"], y=dag["load"], mode="markers", name="werkdag",
        marker=dict(size=5, color=ACCENT, opacity=0.35),
        hovertemplate="%{x:.1f} °C, %{y:,.0f} MW<extra></extra>"))
    bins = dag.groupby(pd.cut(dag["temp_c"], np.arange(-6, 30, 2)),
                       observed=True)["load"].mean()
    fig.add_trace(go.Scatter(x=[iv.mid for iv in bins.index], y=bins.values,
                             mode="lines+markers", name="gemiddelde per 2 °C",
                             line=dict(color="#b45309", width=2.6)))
    fig = layout(fig, "gemiddelde dagbelasting (MW)",
                 "gemiddelde dagtemperatuur (°C)")
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, **WIDE)
    st.markdown("""
    <div class="note">
    Duidelijk en U-vormig: verwarmen onder ruwweg 15 °C, koelen boven ruwweg
    20 °C, met een vlak dal ertussen. Het verklaart een groot deel van de
    dagelijkse variatie, en het zit bewust niet in het model. Temperatuur
    gebruiken op t+168 betekent een weersverwachting invoeren waarvan de eigen
    fout nergens in deze backtest gemeten is. Een model dat beoordeeld wordt op
    temperaturen die daadwerkelijk zijn opgetreden, rapporteert een accuratesse
    die niemand in productie terugziet. De eerlijke keuzes zijn hem weglaten, of
    een tweede foutenbegroting bouwen voor het weer. Deze demo doet het eerste,
    en de monitoringtab laat zien wat die keuze in augustus 2020 kostte.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Een gepubliceerde voorspelling die de controle niet overleeft"):
        st.markdown("""
        Dezelfde bron levert naast de realisaties ook een dag-vooruit
        voorspelling mee, en dat was de voor de hand liggende harde baseline voor
        dit project. Die houdt geen stand. Gescoord tegen de realisaties uit
        hetzelfde bestand:
        """)
        fc = s.dropna(subset=["tso_day_ahead"]).copy()
        fc["jaar"] = fc["local"].dt.year
        rijen = []
        for jaar, d in fc.groupby("jaar"):
            e = d["tso_day_ahead"] - d["load"]
            rijen.append({"jaar": int(jaar), "uren": len(d),
                          "MAPE %": mape(d["load"], d["tso_day_ahead"]),
                          "bias MW": float(e.mean()),
                          "bias %": float(e.mean() / d["load"].mean() * 100)})
        st.dataframe(pd.DataFrame(rijen).round(2), hide_index=True, **WIDE)
        st.markdown("""
        <div class="warn">
        Een systematische afwijking die drie jaar lang rond +5% ligt en dan
        omklapt naar min 14% is geen voorspelling die slechter werd, dat zijn
        twee kolommen op een verschillende grondslag. Op eerste kerstdag 2018
        staat er 19.423 MW voorspeld tegen 12.139 MW gerealiseerd. Hij wordt hier
        dus niet als baseline gebruikt, en staat op deze pagina om een andere
        reden: een gepubliceerd getal narekenen tegen de realisaties in hetzelfde
        bestand kost een middag, en die stap overslaan is hoe een heel project
        verankerd raakt aan iets dat nooit vergelijkbaar was.
        </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------ 2. backtest ---
with tab2:
    st.subheader("Hoe de accuratesse gemeten is")
    st.markdown("""
    Vooruit lopen, niet opzij. Het model wordt elk kwartaal opnieuw gefit op
    alles tot die knip en op niets erna, en voorspelt daarna elke zes uur 168 uur
    vooruit. Die voorspellingen worden één keer gescoord en nooit herzien. Er zit
    nergens in deze repo een willekeurige train/test-split: uren door elkaar
    husselen laat een model interpoleren tussen het uur ervoor en het uur erna,
    wat prachtig scoort en niets waard is.
    """)
    st.dataframe(pd.DataFrame(meta["folds"]).rename(columns={
        "trained through": "getraind t/m", "first origin": "eerste origin",
        "last target": "laatste doeluur", "forecasts": "voorspellingen"}),
        hide_index=True, **WIDE)
    st.markdown("""
    <div class="note">
    Trainingsrijen stoppen een volle 168 uur vóór elke knip, niet op de knip
    zelf. Een origin dichterbij dan de langste horizon zou een doeluur nodig
    hebben dat het model niet mag zien, en alleen die rijen weggooien zou de
    lange horizons stilletjes uit de trainingsset verwijderen.
    <code>tests/test_no_leakage.py</code> controleert de eigenschap in plaats van
    de bedoeling: het vervangt elke waarde ná de origin door onzin, bouwt de
    features opnieuw en eist dat ze identiek terugkomen.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Fout per horizon")
    h = by_horizon()
    fig = go.Figure()
    for m in MODELS:
        fig.add_trace(go.Scatter(x=h["h"], y=h[f"mape_{m}"], name=NAMES[m],
                                 mode="lines", line=dict(**STYLE[m])))
    fig = layout(fig, "MAPE (%)", "horizon h (uren vooruit)")
    fig.update_xaxes(dtick=24)
    st.plotly_chart(fig, **WIDE)
    st.markdown(f"""
    <div class="note">
    Het model loopt van {nl(h['mape_lgbm'].iloc[0], 2)}% één uur vooruit naar
    {nl(h['mape_lgbm'].iloc[-1], 2)}% een week vooruit. Dat is een kleine
    stijging, en het is het interessantste resultaat hier: op landelijk niveau is
    een week vooruit nauwelijks moeilijker dan een uur vooruit, omdat vrijwel
    alle informatie kalendervorm is die op beide momenten even goed bekend is.
    Het verklaart ook de twee vlakke lijnen. Seizoensnaief gebruikt op elke
    horizon dezelfde waarde en het Fourier-model gebruikt helemaal geen recente
    belasting, dus geen van beide heeft nog iets te verliezen naarmate de horizon
    groeit. Alleen het geboosterde model houdt recente informatie vast, en alleen
    dat heeft een lijn die loopt.
    <br><br>
    De voorspellingen worden om 00, 06, 12 en 18 UTC uitgegeven en niet één keer
    per dag. Bij één dagelijkse origin zou h = 24, 48 tot en met 168 allemaal op
    middernacht landen, het rustigste uur van de dag, en dan meet deze grafiek de
    klok in plaats van de horizon. Vier origins halen dat er niet helemaal uit:
    elke horizon landt nog steeds op vier vaste kloktijden en die set schuift met
    h, wat de kleine rimpeling in deze lijnen is. De zuivere vergelijking is h
    tegen h+24, waar de uren exact matchen en alleen de aanlooptijd verschilt. Zo
    gemeten kost een dag extra aanlooptijd het geboosterde model 0,04 pp en de
    twee vlakke modellen hoogstens 0,01 pp.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Bias, apart gehouden van accuratesse")
    fig = go.Figure()
    for m in MODELS:
        fig.add_trace(go.Scatter(x=h["h"], y=h[f"bias_{m}"], name=NAMES[m],
                                 mode="lines", line=dict(**STYLE[m])))
    fig.add_hline(y=0, line=dict(color="#a8a29e", width=1))
    fig = layout(fig, "gemiddelde fout (MW), positief = te hoog voorspeld",
                 "horizon h (uren vooruit)", height=320)
    fig.update_xaxes(dtick=24)
    st.plotly_chart(fig, **WIDE)
    st.markdown("""
    <div class="note">
    Twee modellen kunnen dezelfde MAE hebben en iets totaal verschillends
    betekenen. Fouten die rond nul schommelen zijn ruis, en daar houd je een
    reserve tegen aan. Fouten die elk uur dezelfde kant op leunen zijn een
    systematisch tekort, en dat repareer je in het model. Het Fourier-model leunt
    ruwweg 250 MW te hoog omdat het geen enkele manier heeft om te merken dat het
    niveau verschoven is.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Betekenen de intervallen wat ze beweren?")
    fig = go.Figure(go.Scatter(x=h["h"], y=h["dekking"], name="werkelijke dekking",
                               mode="lines", line=dict(color=ACCENT, width=2.6)))
    fig.add_hline(y=80, line=dict(color="#b45309", width=1.6, dash="dash"),
                  annotation_text="80% nominaal", annotation_position="top left")
    fig = layout(fig, "aandeel realisaties binnen de band (%)",
                 "horizon h (uren vooruit)", height=320)
    fig.update_yaxes(range=[50, 95])
    fig.update_xaxes(dtick=24)
    st.plotly_chart(fig, **WIDE)
    st.markdown(f"""
    <div class="warn">
    Een 80%-interval dat {nl(meta['coverage80'], 1)}% van de realisaties bevat is
    te smal, en dit is het getal dat ik als eerste bij een klant op tafel zou
    leggen. De banden zijn de empirische kwantielen van de fouten die het model
    in eerdere folds daadwerkelijk maakte. Ze beschrijven het verleden dus goed,
    en houden alleen stand zolang de toekomst zich blijft gedragen als dat
    verleden. In 2020 hield dat op, de dekking zakte naar 55%, en een band die
    niemand hercontroleert was gewoon op dezelfde breedte doorgetekend. De eerste
    fold heeft helemaal geen interval: op dat moment bestond er nog geen fout
    buiten de trainingsset om een breedte uit af te leiden, en er een lenen uit
    latere folds is in de toekomst kijken.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Waar het faalt, wat nuttiger is dan waar het slaagt")
    links, rechts = st.columns(2)
    with links:
        st.caption("MAPE per dagsoort")
        st.dataframe(by_segment("dagsoort").round(2), hide_index=True, **WIDE)
        st.caption("MAPE per seizoen")
        st.dataframe(by_segment("seizoen").round(2), hide_index=True, **WIDE)
    with rechts:
        st.caption("MAPE per dagdeel")
        st.dataframe(by_segment("dagdeel").round(2), hide_index=True, **WIDE)
        st.caption("MAPE per fold")
        bt = load_backtest()
        rijen = []
        for f, d in bt.groupby("fold"):
            rijen.append({"fold": int(f), "vanaf": str(d["target"].min().date()),
                          "Gradient boosting": mape(d["y"], d["pred_lgbm"]),
                          "Seizoensnaief": mape(d["y"], d["pred_seasonal_naive"])})
        st.dataframe(pd.DataFrame(rijen).round(2), hide_index=True, **WIDE)
    st.markdown("""
    <div class="note">
    Op feestdagen verdient het model zijn bestaan: het halveert ruwweg de fout
    van "hetzelfde uur vorige week", dat geen enkele manier heeft om te weten dat
    het vandaag geen normale donderdag is. Piekuren zijn twee keer zo moeilijk
    als de nacht, dus één kop-MAPE vleit vooral de uren waar niemand wakker van
    ligt. En in de laatste fold wordt het geboosterde model <em>verslagen</em>
    door seizoensnaief, wat het hele verhaal van 2020 is en de reden dat de
    monitoringtab bestaat.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("De drie modellen")
    st.markdown("""
| Model | Wat het ziet | Waarom het er is |
|---|---|---|
| **Seizoensnaief** | hetzelfde uur een week eerder | Op een reeks met zoveel weekstructuur is dit een echte concurrent, geen stroman. Wat dit niet verslaat, verdient het niet om te draaien. |
| **Seizoens-Fourier** | Fourier-termen voor de dag-, week- en jaarcyclus, feestdagvlaggen, een trage trend. Ridge op log-belasting, geen recente belasting | De zuivere kalenderblik. Zijn vlakke foutcurve is de referentie die de helling van het geboosterde model betekenis geeft. Gefit op logs omdat feestdagen en seizoenen proportioneel werken, met een halve-variantiecorrectie op de terugtransformatie zodat de logfit geen eigen bias introduceert. |
| **Gradient boosting** | origin-lags (0 tot 168 uur), voortschrijdende gemiddelden, hetzelfde uur één en twee weken vóór het doeluur, kalender, en de horizon zelf | Eén model voor alle 168 horizons met h als feature, zodat de vorm van de foutcurve iets is dat het model produceert in plaats van iets dat de opzet oplegt. |

Geen deep learning en geen Prophet. Geen van beide verdient een plek op 41.640
uurwaarnemingen, en allebei kosten ze het vermogen om uit te leggen waarom een
getal eruit kwam zoals het eruit kwam.
    """)

# -------------------------------------------------------------- 3. besluit ---
with tab3:
    st.subheader("De capaciteitsvraag")
    st.markdown("""
    Een netbeheerder vraagt zelden hoe nauwkeurig de voorspelling is. De vraag is
    of een grens overschreden wordt, hoe zeker dat is, en hoe lang van tevoren je
    het weet. Dat is een ander product dan een puntvoorspelling: je hebt een kans
    nodig, en dus een verdeling in plaats van één getal.
    """)

    k1, k2, k3 = st.columns([1, 1, 1])
    lead = k1.selectbox("Aanlooptijd", [24, 48, 72, 168], index=0,
                        format_func=lambda x: f"{x} uur ({x // 24} dag"
                                              + ("en)" if x // 24 > 1 else ")"))
    bt = load_backtest()
    y_lead = bt.loc[bt["h"] == lead, "y"]
    grens = k2.slider("Capaciteitsgrens (MW)",
                      int(y_lead.quantile(0.80) // 100 * 100),
                      int(y_lead.max() // 100 * 100),
                      int(y_lead.quantile(0.95) // 100 * 100), step=100)
    d = exceedance(lead, float(grens))
    k3.metric("Overschrijdingsuren in de backtest",
              f"{int(d['overschreden'].sum())} van {len(d)}",
              nl(d["overschreden"].mean() * 100, 1) + "% van de uren",
              delta_color="off")

    drempels = np.round(np.arange(0.02, 0.99, 0.02), 2)
    curve = [alarm_stats(d, (d["kans"] >= t).to_numpy()) for t in drempels]
    punt = alarm_stats(d, (d["pred_lgbm"] > grens).to_numpy())
    naief = alarm_stats(d, (d["pred_seasonal_naive"] > grens).to_numpy())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[c["vals_pm"] for c in curve], y=[c["recall"] for c in curve],
        mode="lines", name="kansdrempel, van hoog naar laag",
        line=dict(color=ACCENT, width=2.6),
        text=[f"drempel {t:.2f}".replace(".", ",") for t in drempels],
        hovertemplate="%{text}<br>%{y:.0f}% gevonden"
                      "<br>%{x:.1f} vals alarm per maand<extra></extra>"))
    fig.add_trace(go.Scatter(x=[punt["vals_pm"]], y=[punt["recall"]], mode="markers",
                             name="regel: puntvoorspelling boven de grens",
                             marker=dict(size=13, color="#b45309", symbol="diamond")))
    fig.add_trace(go.Scatter(x=[naief["vals_pm"]], y=[naief["recall"]], mode="markers",
                             name="regel: seizoensnaief boven de grens",
                             marker=dict(size=11, color="#a8a29e", symbol="square")))
    fig = layout(fig, "aandeel overschrijdingen gevonden (%)",
                 "vals alarm per maand", height=380)
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, **WIDE)
    st.markdown(f"""
    <div class="note">
    De puntvoorspelling geeft je één punt op deze grafiek en verder niets. Bij de
    gekozen grens vindt de regel "voorspelling boven de grens"
    {nl(punt['recall'], 0)}% van de overschrijdingen bij
    {nl(punt['vals_pm'], 1)} vals alarm per maand. De kansdrempel geeft je de
    hele curve, zodat je zelf kiest waar je gaat zitten: meer overschrijdingen
    vangen kost meer loze interventies, en dat is een gesprek met de operatie en
    niet met de modelleur. Ter vergelijking staat dezelfde regel op basis van
    seizoensnaief er ook in.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Klopt die kans ook?")
    kal = d.groupby(pd.cut(d["kans"], np.arange(0, 1.05, 0.1)),
                    observed=True).agg(voorspeld=("kans", "mean"),
                                       werkelijk=("overschreden", "mean"),
                                       n=("kans", "size")).dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                             line=dict(color="#b45309", width=1.4, dash="dash")))
    fig.add_trace(go.Scatter(
        x=kal["voorspeld"], y=kal["werkelijk"], mode="lines+markers",
        name="gemeten", line=dict(color=ACCENT, width=2.6),
        marker=dict(size=[float(min(6 + n / 40, 22)) for n in kal["n"]]),
        text=[f"{int(n)} uren" for n in kal["n"]],
        hovertemplate="voorspeld %{x:.0%}<br>werkelijk %{y:.0%}"
                      "<br>%{text}<extra></extra>"))
    fig = layout(fig, "werkelijk aandeel overschrijdingen", "voorspelde kans",
                 height=340)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, **WIDE)

    top = d[d["y"] >= d["y"].quantile(0.90)]
    bias_top = float((top["pred_lgbm"] - top["y"]).mean())
    st.markdown(f"""
    <div class="warn">
    Dit is de plek waar ik het model tegenspreek. Over de hele backtest is de
    bias {nl(overall['lgbm']['bias'], 0)} MW, praktisch nul. Maar in het hoogste
    deciel van de belasting, precies de uren die tegen een capaciteitsgrens aan
    liggen, voorspelt hetzelfde model gemiddeld {nl(abs(bias_top), 0)} MW
    <em>te laag</em>. Een kop-MAPE maakt dat volledig onzichtbaar, en het is de
    verkeerde richting: het model is het meest optimistisch op de momenten waarop
    optimisme het duurst is. Voor een landelijke reeks is dat een voetnoot. Voor
    een station met een harde grens is het de kern van de zaak, en het is de
    eerste correctie die ik zou aanbrengen: een verliesfunctie die onderschatting
    in de staart zwaarder beprijst, of een apart kwantielmodel voor het bovenste
    deel van de verdeling.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("De prijs van een fout")
    st.markdown("""
    Een MAPE vertelt niemand of het model gekocht moet worden. Zet een prijs op de
    twee richtingen van de fout en dat verandert. Fouten staan hier in MW
    gedurende één uur, dus in MWh.
    """)
    p1, p2, p3 = st.columns([1, 1, 2])
    producten = {"Dag vooruit (h = 24)": 24, "Twee dagen (h = 48)": 48,
                 "Week vooruit (h = 168)": 168}
    product = p1.selectbox("Voorspelproduct", list(producten), index=0)
    eur_onder = p2.number_input("€ per MWh te laag voorspeld", 0.0, 5000.0, 180.0, 10.0)
    eur_boven = p2.number_input("€ per MWh te hoog voorspeld", 0.0, 5000.0, 60.0, 10.0)
    p3.markdown("""<div class='note'>Eén horizon tegelijk, en met opzet. De
    backtest bevat 168 voorspellingen voor elk uur, één per horizon, en daar
    overheen sommeren zou dezelfde fout tientallen keren beprijzen. Een prijs
    hoort bij een product: de dag-vooruit voorspelling is een andere aankoop dan
    de week-vooruit voorspelling.</div>""", unsafe_allow_html=True)

    w = load_backtest()
    w = w[w["h"] == producten[product]]
    yv = w["y"].to_numpy(float)
    uren = len(yv)

    def jaarkosten(pred) -> float:
        e = np.asarray(pred, float) - yv
        return float((np.maximum(-e, 0).sum() * eur_onder
                      + np.maximum(e, 0).sum() * eur_boven) / uren * HOURS_PER_YEAR)

    kosten = {m: jaarkosten(w[f"pred_{m}"]) for m in MODELS}
    besparing = kosten["seasonal_naive"] - kosten["lgbm"]
    vermeden = float((np.abs(w["pred_seasonal_naive"].to_numpy(float) - yv).sum()
                      - np.abs(w["pred_lgbm"].to_numpy(float) - yv).sum())
                     / uren * HOURS_PER_YEAR)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Seizoensnaief, foutkosten",
              "€ " + nl(kosten["seasonal_naive"] / 1e6, 0, " mln/jr"))
    m2.metric("Gradient boosting, foutkosten",
              "€ " + nl(kosten["lgbm"] / 1e6, 0, " mln/jr"))
    m3.metric("Verschil", "€ " + nl(besparing / 1e6, 0, " mln/jr"),
              nl(besparing / kosten["seasonal_naive"] * 100, 1) + "% lager")
    m4.metric("Vermeden voorspelfout", nl(vermeden / 1000, 0, " GWh/jr"),
              "zonder prijskaartje", delta_color="off")
    st.markdown(f"""
    <div class="note">
    Het vierde getal draagt geen prijs, en daarom staat het er. Over een jaar
    neemt het model dat aantal GWh absolute voorspelfout weg ten opzichte van
    seizoensnaief, en dat blijft staan wat iemand ook besluit dat een uur fout
    waard is. Alles links daarvan is hetzelfde feit met een prijs eraan geniet.
    <br><br>
    Lees de niveaus met zorg. Dit is het hele Nederlandse net bij een gemiddelde
    belasting van {nl(reeks['load_mean'] / 1000, 1)} GW, dus een fout van
    {nl(overall['lgbm']['mape'], 1)}% is ruwweg {nl(overall['lgbm']['mae'], 0)} MW
    elk uur van het jaar, en elke prijs per MWh maakt daar per definitie een groot
    getal van. Het beprijst bovendien elke MWh fout tegen het volle tarief, wat
    geen enkel afrekenregime doet. Het cijfer dat die bezwaren overleeft is het
    <em>verschil</em> tussen twee voorspellingen op identieke uren, niet het
    niveau van een van beide. En de kostenfunctie zelf hoort met de business
    vastgesteld te worden, niet gekozen door degene die het model bouwde.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------- 4. monitoring ---
with tab4:
    st.subheader("Hoe dit aan de muur zou hangen")
    wk = by_week()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wk["week"], y=wk["mape_seasonal_naive"],
                             name="Seizoensnaief", mode="lines",
                             line=dict(color="#a8a29e", width=1.4)))
    fig.add_trace(go.Scatter(x=wk["week"], y=wk["mape_lgbm"],
                             name="Gradient boosting", mode="lines",
                             line=dict(color=ACCENT, width=2.4)))
    drempel = float(wk["mape_lgbm"].iloc[:52].mean()
                    + 2 * wk["mape_lgbm"].iloc[:52].std())
    fig.add_hline(y=drempel, line=dict(color="#b45309", width=1.4, dash="dash"),
                  annotation_text="alarm bij " + nl(drempel, 1) + "%",
                  annotation_position="top left")
    fig = layout(fig, "MAPE per week (%)", "week")
    st.plotly_chart(fig, **WIDE)

    fig = go.Figure(go.Bar(x=wk["week"], y=wk["bias"], name="bias per week",
                           marker_color=np.where(wk["bias"] > 0, ACCENT, "#b45309")))
    fig.add_hline(y=0, line=dict(color="#a8a29e", width=1))
    fig = layout(fig, "gemiddelde fout per week (MW)", "week", height=280)
    st.plotly_chart(fig, **WIDE)

    gemarkeerd = wk[wk["mape_lgbm"] > drempel]
    in_2020 = int((gemarkeerd["week"].dt.year == 2020).sum())
    eerste = gemarkeerd[gemarkeerd["week"].dt.year == 2020]["week"].min()
    slechtste = wk.loc[wk["bias"].idxmin()]
    st.markdown(f"""
    <div class="warn">
    De alarmlijn ligt twee standaarddeviaties boven de weekfout van het eerste
    jaar, het soort regel dat een team ook echt in productie zet. Hij gaat af in
    {len(gemarkeerd)} van de {len(wk)} weken, en die liggen niet verspreid:
    {in_2020} ervan vallen in 2020, vrijwel aaneengesloten vanaf de week van
    {eerste.day} {MAANDEN[eerste.month]} {eerste.year}. De vraag zakte in twee
    weken ruwweg een tiende, en elk model dat op 2016 tot en met 2019 getraind
    was bleef het land voorspellen dat niet meer bestond. De weekbias wordt
    positief en blijft dat, en dat is wat een kapot model onderscheidt van een
    luidruchtig model: ruis wisselt van teken, een regimebreuk niet. Een
    kwartaalhertraining is daar veel te traag voor, en het antwoord is geen beter
    model maar een poort. Boven de drempel wacht de voorspelling op een mens.
    <br><br>
    De grootste misser wijst de andere kant op, en naar mij. In de week van
    {slechtste['week'].day} {MAANDEN[slechtste['week'].month]}
    {slechtste['week'].year} liep het model {nl(abs(slechtste['bias']), 0)} MW
    <em>onder</em> de realisatie, de enige aanhoudende negatieve bias in de hele
    reeks. Dat was de heetste week van de hele periode, met een piek van 33 °C en
    zes van de acht warmste augustusdagen van vijf jaar erin. De belasting liep
    op naar 13,3 GW tegen 11,5 GW de week ervoor, en een model dat niet naar
    temperatuur mag kijken had geen enkele manier om dat te zien aankomen. De
    keuze om weer weg te laten koopt een eerlijke horizon en kost precies dit, en
    een monitoringpagina die dat verstopte zou niets waard zijn.
    </div>
    """, unsafe_allow_html=True)

    st.caption("Weken boven de alarmdrempel")
    st.dataframe(
        gemarkeerd.assign(week=gemarkeerd["week"].dt.strftime("%Y-%m-%d"))
        .rename(columns={"mape_lgbm": "MAPE model %",
                         "mape_seasonal_naive": "MAPE seizoensnaief %",
                         "bias": "bias MW"}).round(2),
        hide_index=True, **WIDE)

st.markdown("""
<div class="footer">
Ismail Arslan ·
<a href="https://ismailarslan.tech">ismailarslan.tech</a> ·
<a href="mailto:contact@ismailarslan.tech">contact@ismailarslan.tech</a> ·
<a href="https://linkedin.com/in/iqzarslan">linkedin.com/in/iqzarslan</a><br>
Belastingdata: Open Power System Data, time series 2020-10-06 (CC-BY 4.0),
oorspronkelijk ENTSO-E Transparency. Temperatuur: Open-Meteo historisch archief
(CC-BY 4.0). Educatief portfolioproject, niet verbonden aan enige netbeheerder.
</div>
""", unsafe_allow_html=True)
