"""GridCast: hourly Dutch electricity load, one to seven days ahead.

The app trains nothing. Everything it shows was produced by
scripts/build_artifacts.py: a rolling-origin backtest with a quarterly refit
over 2018-2020, plus a snapshot of the operators' capacity map, written to
parquet and committed. Here we only read and aggregate. Interface copy is
Dutch; code and comments stay English.

Navigation is a sidebar rather than tabs: with tabs Streamlit executes every
panel on every rerun, which for ten charts over 670k rows is a second of work
nobody asked for.

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
                   page_icon="⚡", layout="wide",
                   initial_sidebar_state="expanded")

# Streamlit replaced use_container_width with width="stretch" in 1.49. The host
# resolves its own Streamlit version, so pick the spelling that exists rather
# than pinning the whole app to one side of that change.
_VERSION = tuple(int(p) for p in st.__version__.split(".")[:2])
WIDE = {"width": "stretch"} if _VERSION >= (1, 49) else {"use_container_width": True}

DATA = pathlib.Path(__file__).parent / "data" / "processed"
TZ = "Europe/Amsterdam"

INK = "#1c1917"
MUTED = "#78716c"
LINE = "#e7e5e4"
ACCENT = "#4338ca"
WARM = "#b45309"
COOL = "#0f766e"

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
    "fourier": dict(color=COOL, width=1.4, dash="dash"),
    "persistence": dict(color="#d6d3d1", width=1.2, dash="dot"),
}
MODELS = ["lgbm", "seasonal_naive", "hour_of_week", "fourier"]
DAGEN = ["maandag", "dinsdag", "woensdag", "donderdag", "vrijdag",
         "zaterdag", "zondag"]
MAANDEN = {1: "januari", 2: "februari", 3: "maart", 4: "april", 5: "mei",
           6: "juni", 7: "juli", 8: "augustus", 9: "september", 10: "oktober",
           11: "november", 12: "december"}
STATUS_KLEUR = {
    "Nog niet ingekleurd": "#d6d3d1",
    "Beschikbaar": "#15803d",
    "Beperkt beschikbaar": "#eab308",
    "In onderzoek, met wachtrij": "#ea580c",
    "Tekort, met wachtrij": "#b91c1c",
}
STATUS_VOLGORDE = ["Beschikbaar", "Beperkt beschikbaar",
                   "In onderzoek, met wachtrij", "Tekort, met wachtrij",
                   "Nog niet ingekleurd"]

CSS = f"""
<style>
/* Deliberately no padding-top override. Streamlit reserves that space for its
   own fixed toolbar, the reserved amount differs per version, and trimming it
   put the first line of every section underneath the toolbar. */
.block-container {{padding-bottom: 3rem; max-width: 1480px;}}
[data-testid="stSidebar"] {{background: #ffffff; border-right: 1px solid {LINE};}}
[data-testid="stSidebar"] .block-container {{padding-top: 1.6rem;}}
.brand {{font-size: 1.28rem; font-weight: 700; color: {INK}; letter-spacing: -0.01em;}}
.brand span {{color: {ACCENT};}}
.brand-sub {{color: {MUTED}; font-size: 0.82rem; margin: 2px 0 18px; line-height: 1.45;}}
.side-note {{color: #a8a29e; font-size: 0.76rem; line-height: 1.5;
  border-top: 1px solid {LINE}; margin-top: 20px; padding-top: 14px;}}
/* The sidebar radio is navigation, not a form: hide the radio dots and style
   each option as a nav row, with the checked one as the active page. */
[data-testid="stSidebar"] [role="radiogroup"] {{gap: 2px;}}
[data-testid="stSidebar"] [role="radiogroup"] label {{
  display: flex; align-items: center; width: 100%; margin: 0;
  padding: 8px 12px; border-radius: 9px; cursor: pointer;
  transition: background 0.12s;}}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{
  display: none;}}
[data-testid="stSidebar"] [role="radiogroup"] label p {{
  font-size: 0.94rem; font-weight: 600; color: #57534e; margin: 0;}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{
  background: #f5f5f4;}}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
  background: #eef2ff;}}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {{
  color: {ACCENT};}}
.eyebrow {{text-transform: uppercase; letter-spacing: 0.09em; font-size: 0.72rem;
  font-weight: 700; color: {ACCENT}; margin-bottom: 6px;}}
h1.sec {{font-size: 1.72rem; font-weight: 700; color: {INK}; margin: 0 0 8px;
  letter-spacing: -0.015em; line-height: 1.2;}}
p.stand {{color: {MUTED}; font-size: 1.0rem; max-width: 88ch; margin: 0 0 22px;
  line-height: 1.6;}}
h2.sub {{font-size: 1.06rem; font-weight: 700; color: {INK};
  margin: 26px 0 2px; letter-spacing: -0.005em;}}
p.subnote {{color: {MUTED}; font-size: 0.9rem; margin: 0 0 10px; max-width: 92ch;}}
[data-testid="stMetric"] {{background: #ffffff; border: 1px solid {LINE};
  border-radius: 12px; padding: 15px 17px;}}
[data-testid="stMetricLabel"] {{color: {MUTED};}}
.note, .warn {{border-radius: 10px; padding: 14px 18px; margin: 4px 0 8px;
  font-size: 0.92rem; line-height: 1.62; max-width: 104ch;}}
.note {{background: #ffffff; border: 1px solid {LINE};
  border-left: 3px solid #a8a29e; color: #57534e;}}
.warn {{background: #fffbeb; border: 1px solid #fde68a;
  border-left: 3px solid {WARM}; color: #78350f;}}
.legend {{display: flex; flex-wrap: wrap; gap: 16px; margin: 2px 0 10px;
  font-size: 0.84rem; color: #57534e;}}
.legend i {{display: inline-block; width: 11px; height: 11px; border-radius: 3px;
  margin-right: 6px; vertical-align: middle;}}
.footer {{color: #a8a29e; font-size: 0.82rem; margin-top: 34px;
  border-top: 1px solid {LINE}; padding-top: 16px; line-height: 1.6;}}
.footer a {{color: #a8a29e;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def nl(x: float, decimals: int = 0, unit: str = "") -> str:
    """Dutch number formatting: dot for thousands, comma for decimals."""
    s = f"{x:,.{decimals}f}".replace(",", "~").replace(".", ",").replace("~", ".")
    return f"{s}{unit}"


def kop(eyebrow: str, titel: str, stand: str) -> None:
    st.markdown(f"<div class='eyebrow'>{eyebrow}</div>"
                f"<h1 class='sec'>{titel}</h1><p class='stand'>{stand}</p>",
                unsafe_allow_html=True)


def sub(titel: str, note: str = "") -> None:
    html = f"<h2 class='sub'>{titel}</h2>"
    if note:
        html += f"<p class='subnote'>{note}</p>"
    st.markdown(html, unsafe_allow_html=True)


def note(text: str, warn: bool = False) -> None:
    st.markdown(f"<div class='{'warn' if warn else 'note'}'>{text}</div>",
                unsafe_allow_html=True)


# ---------------------------------------------------------------- loading ---
def fp(name: str) -> str:
    """Fingerprint of a parquet file: size and modification time.

    Every cached function that reads a file takes this as an argument, so the
    cache key changes when the file does. Without it a deploy that ships new
    artifacts into a process that is only reloaded, not restarted, keeps
    serving the previous build's frames: the code expects a column the cached
    DataFrame has never heard of, and the app dies on a KeyError that looks
    like a data bug and is not one.
    """
    s = (DATA / name).stat()
    return f"{name}:{s.st_size}:{int(s.st_mtime)}"


@st.cache_data(show_spinner="Backtest inlezen…")
def _backtest(fingerprint: str) -> pd.DataFrame:
    bt = pd.read_parquet(DATA / "backtest.parquet")
    local = bt["target"].dt.tz_convert(TZ)
    bt["uur"] = local.dt.hour
    bt["week"] = local.dt.tz_localize(None).dt.to_period("W").dt.start_time
    bt["seizoen"] = np.select(
        [local.dt.month.isin([12, 1, 2]), local.dt.month.isin([6, 7, 8])],
        ["winter", "zomer"], default="voor- en najaar")
    bt["dagdeel"] = np.where(local.dt.hour.between(7, 19), "piek 07-19", "dal")
    bt["dagsoort"] = bt["segment"].map({"working day": "werkdag",
                                        "weekend": "weekend",
                                        "public holiday": "feestdag"})
    return bt


def load_backtest() -> pd.DataFrame:
    return _backtest(fp("backtest.parquet"))


@st.cache_data
def _series(fingerprint: str) -> pd.DataFrame:
    s = pd.read_parquet(DATA / "series.parquet")
    s = s.rename(columns={s.columns[0]: "utc_timestamp"})
    s["local"] = s["utc_timestamp"].dt.tz_convert(TZ)
    s["uur"] = s["local"].dt.hour
    s["dag"] = s["local"].dt.dayofweek
    s["maand"] = s["local"].dt.month
    s["jaar"] = s["local"].dt.year
    s["dagsoort"] = np.where(s["is_holiday"], "feestdag",
                             np.where(s["dag"] >= 5, "weekend", "werkdag"))
    s["seizoen"] = np.select(
        [s["maand"].isin([12, 1, 2]), s["maand"].isin([6, 7, 8])],
        ["winter", "zomer"], default="voor- en najaar")
    return s


def load_series() -> pd.DataFrame:
    return _series(fp("series.parquet"))


@st.cache_data
def _meta(fingerprint: str) -> dict:
    return json.loads((DATA / "meta.json").read_text(encoding="utf-8"))


def load_meta() -> dict:
    return _meta(fp("meta.json"))


@st.cache_data
def _parquet(fingerprint: str) -> pd.DataFrame:
    name = fingerprint.split(":", 1)[0]
    return pd.read_parquet(DATA / name)


def load_monthly_levels() -> pd.DataFrame:
    return _parquet(fp("monthly_levels.parquet"))


def load_quantiles() -> pd.DataFrame:
    return _parquet(fp("residual_quantiles.parquet"))


def load_capaciteit() -> pd.DataFrame:
    return _parquet(fp("capaciteit.parquet"))


def load_importances() -> pd.DataFrame:
    return _parquet(fp("importances.parquet"))


def mape(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs((p - y) / y)) * 100)


@st.cache_data
def _by_horizon(fingerprint: str) -> pd.DataFrame:
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


def by_horizon() -> pd.DataFrame:
    return _by_horizon(fp("backtest.parquet"))


@st.cache_data
def _by_week(fingerprint: str) -> pd.DataFrame:
    bt = load_backtest()
    return pd.DataFrame([
        {"week": wk,
         "mape_lgbm": mape(d["y"], d["pred_lgbm"]),
         "mape_seasonal_naive": mape(d["y"], d["pred_seasonal_naive"]),
         "bias": float(np.mean(d["pred_lgbm"] - d["y"]))}
        for wk, d in bt.groupby("week")])


def by_week() -> pd.DataFrame:
    return _by_week(fp("backtest.parquet"))


@st.cache_data
def _exceedance(h: int, capacity: float, fingerprint: str) -> pd.DataFrame:
    """Turn each point forecast into a probability that the limit is passed.

    P(y > C) = P(residual > C - forecast), read from the empirical residual
    distribution of that horizon, built from folds that had already closed.
    Fold 1 has no such distribution and is left out, as with the intervals.
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


def exceedance(h: int, capacity: float) -> pd.DataFrame:
    return _exceedance(h, capacity, fp("backtest.parquet"))


def alarm_stats(d: pd.DataFrame, alarm: np.ndarray) -> dict:
    raak = int((alarm & d["overschreden"]).sum())
    vals = int((alarm & ~d["overschreden"]).sum())
    gemist = int((~alarm & d["overschreden"]).sum())
    maanden = max((d["target"].max() - d["target"].min()).days / 30.44, 1e-9)
    return {"raak": raak, "vals": vals, "gemist": gemist,
            "recall": raak / max(raak + gemist, 1) * 100,
            "precisie": raak / max(raak + vals, 1) * 100,
            "vals_pm": vals / maanden}


def layout(fig: go.Figure, ytitle: str, xtitle: str, height: int = 360,
           legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, template="simple_white",
        margin=dict(l=8, r=8, t=28 if legend else 8, b=8),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                    font=dict(size=11)),
        hovermode="x unified",
        font=dict(color="#44403c", size=12, family="system-ui, sans-serif"),
        plot_bgcolor="#ffffff", paper_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_yaxes(title=ytitle, gridcolor="#f2f1f0", zeroline=False)
    fig.update_xaxes(title=xtitle)
    return fig


def legenda(paren) -> None:
    items = "".join(f"<span><i style='background:{c}'></i>{n}</span>"
                    for n, c in paren)
    st.markdown(f"<div class='legend'>{items}</div>", unsafe_allow_html=True)


meta = load_meta()
overall = meta["overall"]
reeks = meta["series"]
cap_meta = meta.get("capaciteit", {})
n_forecasts = int(sum(f["forecasts"] for f in meta["folds"]))

SECTIES = ["Overzicht", "Patronen in de vraag", "Het model",
           "Regio en congestie", "Waarde en besluit", "Monitoring"]

with st.sidebar:
    st.markdown("<div class='brand'>⚡ Grid<span>Cast</span></div>"
                "<div class='brand-sub'>Landelijke elektriciteitsvraag per uur,"
                " 1 tot 7 dagen vooruit</div>", unsafe_allow_html=True)
    sectie = st.radio("Sectie", SECTIES, label_visibility="collapsed")
    st.markdown(
        f"<div class='side-note'>Backtest {meta['test_start'][:4]} tot 2020"
        f"<br>{len(meta['folds'])} kwartaalhertrainingen"
        f"<br>{nl(n_forecasts)} voorspellingen"
        f"<br>Capaciteitskaart opgehaald {meta.get('capaciteit_opgehaald', '')}"
        "</div>", unsafe_allow_html=True)

# ------------------------------------------------------------- overzicht ---
if sectie == "Overzicht":
    kop("Portfolioproject",
        "Voorspellen is het makkelijke deel",
        "De vraag die telt is wanneer je je eigen getal nog mag geloven. Deze "
        "demo voorspelt de Nederlandse elektriciteitsvraag per uur tot 168 uur "
        "vooruit, en besteedt de meeste ruimte aan de plekken waar dat misgaat. "
        "Naast elk cijfer staat de baseline die het moet verslaan.")

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

    sub("Wat hier te zien is")
    a, b, c = st.columns(3)
    with a:
        note("<b>Patronen in de vraag.</b> De jaar-, week- en daglaag van "
             "elektriciteitsvraag, en wat temperatuur, feestdagen en het weekend "
             "ermee doen. Elk patroon dat je hier ziet moet een model kunnen "
             "leren, en elk patroon dat het mist zie je later terug in de fout.")
    with b:
        note("<b>Het model.</b> Eerst één voorspelling van dichtbij, uur voor "
             "uur tegen wat er echt gebeurde. Daarna het bewijs: de fout per "
             "horizon, de betrouwbaarheid van de onzekerheidsband, en waar het "
             "model daadwerkelijk op leunt.")
    with c:
        note("<b>Regio en congestie.</b> De voorspelling hier is landelijk, het "
             "probleem is dat niet. De capaciteitskaart van de netbeheerders "
             "laat zien waar het net vol zit, en waarom een landelijk gemiddelde "
             "niet de operationele vraag is.")

    sub("Drie resultaten die niet vleien",
        "Ze staan vooraan in plaats van in een voetnoot, en de selftest pint ze "
        "vast zodat een latere wijziging ze niet stilletjes kan laten verdwijnen.")
    note(
        f"<b>Het interval is te smal.</b> Een 80%-band die "
        f"{nl(meta['coverage80'], 1)}% van de realisaties bevat belooft meer "
        "zekerheid dan hij levert. In het voorjaar van 2020 zakte de dekking naar "
        "66%.<br><br>"
        "<b>Het model onderschat de piek.</b> Over de hele backtest is de bias "
        f"{nl(overall['lgbm']['bias'], 0)} MW, praktisch nul. In het hoogste "
        "deciel van de belasting voorspelt hetzelfde model ruim 120 MW te laag: "
        "het is het meest optimistisch precies waar optimisme het duurst is."
        "<br><br>"
        "<b>Het verliest van de baseline.</b> In de laatste fold, na de "
        "vraaguitval van 2020, is seizoensnaief nauwkeuriger dan het getrainde "
        "model.", warn=True)

# --------------------------------------------------------------- patronen ---
elif sectie == "Patronen in de vraag":
    s = load_series()
    kop("Verkenning", "Drie lagen, boven elkaar",
        "Elektriciteitsvraag is opgebouwd uit een jaarritme, een weekritme en "
        "een dagritme, met daar bovenop het weer en de kalender. Alles hieronder "
        "is beschrijvend: dit is wat er in de data zit, voordat er een model aan "
        "te pas komt.")

    sub("De jaarlaag, en waarom de reeks in 2016 begint",
        "Gemiddelde belasting per kalendermaand, per jaar. 2015 ligt er "
        "structureel onder en doet niet mee.")
    lv = load_monthly_levels()
    fig = go.Figure()
    for jaar in [c for c in lv.columns if c != "month"]:
        uit = str(jaar) == "2015"
        fig.add_trace(go.Scatter(
            x=lv["month"], y=lv[jaar], name=str(jaar), mode="lines+markers",
            line=dict(color=WARM if uit else ACCENT, width=2.8 if uit else 1.5,
                      dash="dash" if uit else None),
            opacity=1.0 if uit else 0.45))
    fig = layout(fig, "gemiddelde belasting (MW)", "kalendermaand", 340)
    fig.update_xaxes(tickmode="array", tickvals=list(range(1, 13)),
                     ticktext=[MAANDEN[m][:3] for m in range(1, 13)])
    st.plotly_chart(fig, **WIDE)
    note("Winter ligt ruwweg 15% boven zomer, met januari als top en augustus "
         "als dal. Dat is de jaarlaag, en hij is volledig voorspelbaar uit de "
         "kalender.<br><br>"
         "Elke maand van 2015 ligt 1,7 tot 2,0 GW onder dezelfde maand van elk "
         "later jaar, terwijl 2016 tot en met 2019 op elkaar liggen. Verbruik "
         "verspringt niet 16% in één januari om daarna vier jaar stil te staan. "
         "Dit is een wijziging in wat er gerapporteerd werd, niet in wat er "
         "verbruikt werd, en de reeks start daarom op 1 januari 2016.")

    sub("De week- en daglaag in één beeld",
        "Gemiddelde belasting per uur van de dag en dag van de week. Donker is "
        "hoog.")
    piv = (s.pivot_table(index="dag", columns="uur", values="load",
                         aggfunc="mean").reindex(range(7)))
    fig = go.Figure(go.Heatmap(
        z=piv.values, x=list(range(24)), y=DAGEN, colorscale="Blues",
        colorbar=dict(title="MW", thickness=12, outlinewidth=0),
        hovertemplate="%{y} %{x}:00<br>%{z:,.0f} MW<extra></extra>"))
    fig = layout(fig, "", "uur van de dag (lokale tijd)", 300, legend=False)
    fig.update_xaxes(dtick=2)
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, **WIDE)
    werkdag_piek = s[s["dagsoort"] == "werkdag"].groupby("uur")["load"].mean()
    weekend_piek = s[s["dagsoort"] == "weekend"].groupby("uur")["load"].mean()
    note(
        f"Het werkweekblok is meteen zichtbaar: maandag tot en met vrijdag "
        f"tussen 07:00 en 19:00 loopt tot {nl(werkdag_piek.max(), 0)} MW, terwijl "
        f"het weekend op {nl(weekend_piek.max(), 0)} MW blijft steken. De "
        f"nachtelijke bodem rond {int(werkdag_piek.idxmin())}:00 is de rustigste "
        "en best voorspelbare periode van de week, en juist dat maakt een "
        "kop-MAPE misleidend: die wordt voor een groot deel gedragen door uren "
        "waar weinig gebeurt.")

    sub("De dagvorm per dagsoort en per seizoen",
        "Dezelfde 24 uur, uitgesplitst. Links naar dagsoort, rechts naar seizoen.")
    links, rechts = st.columns(2)
    with links:
        fig = go.Figure()
        for label, kleur in zip(["werkdag", "weekend", "feestdag"],
                                [ACCENT, "#57534e", WARM]):
            prof = s[s["dagsoort"] == label].groupby("uur")["load"].mean()
            fig.add_trace(go.Scatter(x=prof.index, y=prof.values, name=label,
                                     mode="lines", line=dict(width=2.4, color=kleur)))
        fig = layout(fig, "gemiddelde belasting (MW)", "uur van de dag", 320)
        fig.update_xaxes(dtick=3)
        st.plotly_chart(fig, **WIDE)
    with rechts:
        fig = go.Figure()
        for label, kleur in zip(["winter", "voor- en najaar", "zomer"],
                                [ACCENT, "#a8a29e", WARM]):
            prof = s[s["seizoen"] == label].groupby("uur")["load"].mean()
            fig.add_trace(go.Scatter(x=prof.index, y=prof.values, name=label,
                                     mode="lines", line=dict(width=2.4, color=kleur)))
        fig = layout(fig, "gemiddelde belasting (MW)", "uur van de dag", 320)
        fig.update_xaxes(dtick=3)
        st.plotly_chart(fig, **WIDE)

    werk = s[s["dagsoort"] == "werkdag"]["load"].mean()
    weekend = s[s["dagsoort"] == "weekend"]["load"].mean()
    feest = s[s["dagsoort"] == "feestdag"]["load"].mean()
    note(
        f"Een weekenddag ligt {nl(abs(weekend / werk - 1) * 100, 1)}% onder een "
        f"werkdag, een feestdag {nl(abs(feest / werk - 1) * 100, 1)}%. Een "
        "feestdag gedraagt zich dus als een weekenddag die midden in de week "
        "valt, en dat is precies waar het verschil tussen een model en "
        "\"hetzelfde uur vorige week\" vandaan komt: vorige week was het een "
        "gewone donderdag.<br><br>"
        "In de winter is de avondpiek scherper en ligt het hele profiel hoger. "
        "De zomer is vlakker: de ochtendpiek blijft, de avondpiek verdwijnt "
        "grotendeels met het daglicht.")

    sub("Welke feestdag hoeveel scheelt",
        "Gemiddelde belasting op de dag zelf, afgezet tegen een gewone werkdag "
        "in dezelfde maand. Per feestdag bij naam, want Pasen, Hemelvaart en "
        "Pinksteren verschuiven elk jaar en een datum zegt dus niets. Alleen "
        "feestdagen die doordeweeks vielen, anders meet je het weekend mee.")
    fd = s[s["is_holiday"] & (s["dag"] < 5)].copy()
    fd["datum"] = fd["local"].dt.date
    per_dag = fd.groupby(["holiday_name", "datum"]).agg(
        load=("load", "mean"), maand=("maand", "first")).reset_index()
    normaal = s[s["dagsoort"] == "werkdag"].groupby("maand")["load"].mean()
    per_dag["afwijking"] = (per_dag["load"] / per_dag["maand"].map(normaal) - 1) * 100
    fdg = (per_dag.groupby("holiday_name")["afwijking"]
           .agg(gemiddeld="mean", laagste="min", hoogste="max", jaren="count")
           .sort_values("gemiddeld").reset_index())
    fig = go.Figure(go.Bar(
        x=fdg["gemiddeld"], y=fdg["holiday_name"], orientation="h",
        marker_color=[WARM if n < 3 else ACCENT for n in fdg["jaren"]],
        error_x=dict(type="data", symmetric=False,
                     array=fdg["hoogste"] - fdg["gemiddeld"],
                     arrayminus=fdg["gemiddeld"] - fdg["laagste"],
                     color="#a8a29e", thickness=1.2, width=4),
        customdata=np.stack([fdg["jaren"], fdg["laagste"], fdg["hoogste"]], axis=-1),
        hovertemplate="<b>%{y}</b><br>gemiddeld %{x:.1f}%"
                      "<br>%{customdata[0]} jaren, van %{customdata[1]:.1f}%"
                      " tot %{customdata[2]:.1f}%<extra></extra>"))
    fig = layout(fig, "", "afwijking t.o.v. een gewone werkdag in dezelfde maand (%)",
                 max(300, 42 * len(fdg)), legend=False)
    st.plotly_chart(fig, **WIDE)
    legenda([("gemeten over drie jaar of meer", ACCENT),
             ("minder dan drie jaar, dunne basis", WARM),
             ("grijze streep: laagste tot hoogste jaar", "#a8a29e")])
    note(
        "Nieuwjaarsdag en eerste kerstdag zijn de diepste dalen, rond een vijfde "
        "onder een gewone werkdag. Het uitschieteruur is <b>Goede Vrijdag</b>: die "
        "staat wel in de feestdagenlijst maar is in Nederland voor de meeste "
        "mensen geen vrije dag, en dat is precies wat de data zegt met "
        f"{nl(fdg.loc[fdg['holiday_name'] == 'Goede Vrijdag', 'gemiddeld'].iloc[0], 1)}% "
        "gemiddeld en in één jaar zelfs een hogere belasting dan normaal. Een "
        "model dat alle feestdagen als één vlag behandelt, leert dus een "
        "gemiddelde dat voor geen enkele feestdag klopt.<br><br>"
        "Bevrijdingsdag rust op één jaar en staat daarom in amber: die is alleen "
        "in een lustrumjaar een vrije dag, en binnen deze reeks is dat alleen "
        "2020. Eén waarneming is geen effect, en een balk die dat verzwijgt is "
        "een balk die liegt.")

    sub("Waar de reeks zelf grillig geworden is",
        "Hoeveel een uur afwijkt van hetzelfde uur een week eerder, per uur "
        "van de dag. Dit is een eigenschap van de data, er komt geen model aan "
        "te pas.")
    s_idx = s.set_index(pd.DatetimeIndex(s["utc_timestamp"]))
    wow = (s_idx["load"] - s_idx["load"].shift(168)).abs() / s_idx["load"] * 100
    gril = pd.DataFrame({"uur": s_idx["uur"], "jaar": s_idx["jaar"],
                         "ape": wow}).dropna()
    fig = go.Figure()
    for jaar, kleur, breedte in [(2016, "#d6d3d1", 1.6), (2018, "#a8a29e", 1.6),
                                 (2020, WARM, 2.8)]:
        prof = gril[gril["jaar"] == jaar].groupby("uur")["ape"].mean()
        fig.add_trace(go.Scatter(x=prof.index, y=prof.values, name=str(jaar),
                                 mode="lines",
                                 line=dict(color=kleur, width=breedte)))
    fig = layout(fig, "week-op-week verschil (% van de belasting)",
                 "uur van de dag (lokale tijd)", 320)
    fig.update_xaxes(dtick=2)
    st.plotly_chart(fig, **WIDE)
    note(
        "De nacht is in vijf jaar niets veranderd: rond 2,5% afwijking van "
        "dezelfde nacht een week eerder. De middag is ontploft: van 4,5% in "
        "2016 naar ruim 10% in 2020, en de groei zit exact in de daglichturen. "
        "Dat is <b>zon op dak</b>. De gemeten landelijke belasting is verbruik "
        "mínus opwek achter de meter, het Nederlandse zonvermogen is in deze "
        "periode ruwweg vervijfvoudigd, en daarmee werd het middaguur "
        "weersafhankelijk: een bewolkte week na een zonnige maakt \"zelfde uur "
        "vorige week\" waardeloos. Voor elk model geldt hetzelfde, dus dit is "
        "de plek waar de fout vandaan komt én de plek waar hij elk jaar "
        "groeit. Wie deze reeks vandaag serieus voorspelt, heeft instraling "
        "nodig; dat staat in de README als de eerlijke grens van deze demo.")

    sub("Temperatuur, die dit model bewust niet gebruikt",
        "Gemiddelde dagbelasting tegen gemiddelde dagtemperatuur, alleen "
        "werkdagen.")
    dag = (s.set_index("local").resample("D")
           .agg({"load": "mean", "temp_c": "mean", "dag": "first",
                 "is_holiday": "max"}).dropna())
    dag = dag[(dag["dag"] < 5) & ~dag["is_holiday"].astype(bool)]
    fig = go.Figure(go.Scattergl(
        x=dag["temp_c"], y=dag["load"], mode="markers", name="werkdag",
        marker=dict(size=5, color=ACCENT, opacity=0.3),
        hovertemplate="%{x:.1f} °C, %{y:,.0f} MW<extra></extra>"))
    bins = dag.groupby(pd.cut(dag["temp_c"], np.arange(-6, 30, 2)),
                       observed=True)["load"].mean()
    fig.add_trace(go.Scatter(x=[iv.mid for iv in bins.index], y=bins.values,
                             mode="lines+markers", name="gemiddelde per 2 °C",
                             line=dict(color=WARM, width=2.8)))
    fig = layout(fig, "gemiddelde dagbelasting (MW)",
                 "gemiddelde dagtemperatuur (°C)", 340)
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, **WIDE)
    note("Duidelijk en U-vormig: verwarmen onder ruwweg 15 °C, koelen boven "
         "ruwweg 20 °C, met een vlak dal ertussen. Het verklaart een groot deel "
         "van de dagelijkse variatie, en het zit er bewust niet in. Temperatuur "
         "gebruiken op t+168 betekent een weersverwachting invoeren waarvan de "
         "eigen fout nergens in deze backtest gemeten is. Een model dat "
         "beoordeeld wordt op temperaturen die daadwerkelijk zijn opgetreden "
         "rapporteert een accuratesse die niemand in productie terugziet. De "
         "eerlijke keuzes zijn hem weglaten of een tweede foutenbegroting bouwen "
         "voor het weer. Deze demo doet het eerste, en de monitoringsectie laat "
         "zien wat dat in augustus 2020 kostte.")

    with st.expander("Een gepubliceerde voorspelling die de controle niet overleeft"):
        fc = s.dropna(subset=["tso_day_ahead"]).copy()
        rijen = []
        for jaar, d in fc.groupby("jaar"):
            e = d["tso_day_ahead"] - d["load"]
            rijen.append({"jaar": int(jaar), "uren": len(d),
                          "MAPE %": mape(d["load"], d["tso_day_ahead"]),
                          "bias MW": float(e.mean()),
                          "bias %": float(e.mean() / d["load"].mean() * 100)})
        st.dataframe(pd.DataFrame(rijen).round(2), hide_index=True, **WIDE)
        note("Dezelfde bron levert naast de realisaties een dag-vooruit "
             "voorspelling mee, en dat was de voor de hand liggende harde "
             "baseline. Een systematische afwijking die drie jaar lang rond +5% "
             "ligt en dan omklapt naar min 14% is geen voorspelling die slechter "
             "werd, dat zijn twee kolommen op een verschillende grondslag. Op "
             "eerste kerstdag 2018 staat er 19.423 MW voorspeld tegen 12.139 MW "
             "gerealiseerd. Hij wordt hier dus niet als baseline gebruikt, en "
             "staat er om een andere reden: een gepubliceerd getal narekenen "
             "kost een middag, en die stap overslaan is hoe een heel project "
             "verankerd raakt aan iets dat nooit vergelijkbaar was.", warn=True)

# ------------------------------------------------------------- het model ---
elif sectie == "Het model":
    kop("Het model", "Eerst kijken, dan meten",
        "Eén voorspelling van dichtbij, precies zoals hij op het moment van "
        "afgifte bestond: 168 uur vooruit, met de onzekerheidsband, afgezet "
        "tegen wat er daarna werkelijk gebeurde. Daaronder het bewijs over de "
        "volle backtest. Kies een rustige week om te zien wat het model kan, "
        "en daarna maart 2020 om te zien waar het ophoudt.")

    bt = load_backtest()
    PRESETS = {
        "Gewone werkweek — mei 2019": "2019-05-13",
        "Hemelvaartsweek — mei 2019": "2019-05-27",
        "Kerstweek — december 2018": "2018-12-24",
        "Eerste lockdownweek — maart 2020": "2020-03-16",
        "Hittegolf — augustus 2020": "2020-08-07",
    }
    c1, c2 = st.columns([2, 1])
    keuze = c1.selectbox("Kies een week", list(PRESETS) + ["Kies zelf een datum"])
    cand = pd.DatetimeIndex(sorted(bt.loc[bt["origin"].dt.hour == 0,
                                          "origin"].unique()))
    if keuze == "Kies zelf een datum":
        gekozen = c2.date_input("Afgiftedatum (00:00 UTC)",
                                value=pd.Timestamp("2019-05-13").date(),
                                min_value=cand.min().date(),
                                max_value=cand.max().date())
        doel = pd.Timestamp(gekozen, tz="UTC")
    else:
        doel = pd.Timestamp(PRESETS[keuze], tz="UTC")
    o = cand[int(np.argmin(np.abs((cand - doel).to_numpy())))]
    w = bt[bt["origin"] == o].sort_values("h")

    heeft_band = w["lo80"].notna().any()
    fig = go.Figure()
    if heeft_band:
        fig.add_trace(go.Scatter(x=w["target"], y=w["hi80"], mode="lines",
                                 line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=w["target"], y=w["lo80"], mode="lines",
                                 line=dict(width=0), fill="tonexty",
                                 fillcolor="rgba(67, 56, 202, 0.13)",
                                 name="80%-band", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=w["target"], y=w["pred_seasonal_naive"],
                             name="seizoensnaief", mode="lines",
                             line=dict(color="#a8a29e", width=1.3, dash="dot")))
    fig.add_trace(go.Scatter(x=w["target"], y=w["y"], name="werkelijk",
                             mode="lines", line=dict(color=INK, width=2.0)))
    fig.add_trace(go.Scatter(x=w["target"], y=w["pred_lgbm"], name="voorspelling",
                             mode="lines", line=dict(color=ACCENT, width=2.6)))
    fig = layout(fig, "belasting (MW)", "", 420)
    fig.add_annotation(x=w["target"].iloc[0], y=1.02, yref="paper",
                       text="afgifte +1u", showarrow=False,
                       font=dict(size=10, color=MUTED))
    st.plotly_chart(fig, **WIDE)

    m_w = mape(w["y"], w["pred_lgbm"])
    m_sn = mape(w["y"], w["pred_seasonal_naive"])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Afgegeven op", o.strftime("%d-%m-%Y %H:%M UTC"))
    m2.metric("MAPE deze week", nl(m_w, 2, "%"),
              nl(m_w - overall["lgbm"]["mape"], 2) + " pp t.o.v. gemiddeld",
              delta_color="inverse")
    m3.metric("Seizoensnaief deze week", nl(m_sn, 2, "%"),
              "de baseline", delta_color="off")
    if heeft_band:
        dek = float(np.mean((w["y"] >= w["lo80"]) & (w["y"] <= w["hi80"])) * 100)
        m4.metric("In de 80%-band", nl(dek, 0, "%"),
                  "van deze 168 uren", delta_color="off")
    else:
        m4.metric("80%-band", "geen",
                  "eerste fold: nog geen fouthistorie", delta_color="off")

    note(
        "Waar je op moet letten. In een gewone week zit vrijwel alle winst in "
        "de <em>vorm</em>: het model raakt het dubbele piekpatroon van een "
        "werkdag en de vlakke zaterdag, terwijl seizoensnaief elke afwijking "
        "van vorige week letterlijk herhaalt, inclusief de fouten. In de "
        "kerstweek zie je het model de feestdagen omzeilen waar de baseline er "
        "vol in stapt. En in de lockdownweek van maart 2020 zie je beide het "
        "spoor bijster: de werkelijkheid zakt onder alles wat het model ooit "
        "geleerd heeft, de band houdt het niet, en precies dat moment is waar "
        "de monitoringsectie over gaat. Eén voorspelling per keer bekijken is "
        "geen bewijs, daarvoor zijn de andere secties; maar wie hier een paar "
        "weken doorklikt, weet daarna wat die gemiddelden betekenen.")

    sub("De fout per horizon, over de volle backtest",
        "Elke zes uur een nieuwe afgifte, 168 uur vooruit, elk kwartaal "
        "hertraind op alleen het verleden. Alle modellen op dezelfde uren "
        "gescoord.")
    h = by_horizon()
    # Each horizon lands on exactly four clock hours (origins run 00/06/12/18
    # UTC) and that set shifts with h. Because the error depends heavily on the
    # time of day, the raw per-h line carries a period-six sawtooth that is
    # sampling, not signal. Six consecutive horizons cover every clock-hour mix
    # exactly once, so a six-wide average removes the artefact exactly.
    glad = h.copy()
    for c in glad.columns:
        if c != "h":
            glad[c] = h[c].rolling(6, center=True, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h["h"], y=h["mape_lgbm"], name="per losse horizon",
                             mode="lines",
                             line=dict(color="#c7d2fe", width=1.0)))
    for m in ["lgbm", "seasonal_naive", "fourier"]:
        fig.add_trace(go.Scatter(x=glad["h"], y=glad[f"mape_{m}"], name=NAMES[m],
                                 mode="lines", line=dict(**STYLE[m])))
    fig = layout(fig, "MAPE (%)", "horizon h (uren vooruit)", 340)
    fig.update_xaxes(dtick=24)
    st.plotly_chart(fig, **WIDE)
    note(
        f"Het model loopt van {nl(glad['mape_lgbm'].iloc[0], 2)}% één uur "
        f"vooruit naar {nl(glad['mape_lgbm'].iloc[-1], 2)}% een week vooruit. "
        "Dat is een kleine stijging, en het is het interessantste resultaat "
        "hier: op landelijk niveau is een week vooruit nauwelijks moeilijker "
        "dan een uur vooruit, omdat vrijwel alle informatie kalendervorm is "
        "die op beide momenten even goed bekend is. Seizoensnaief gebruikt op "
        "elke horizon dezelfde waarde en het Fourier-model gebruikt helemaal "
        "geen recente belasting, dus die twee hebben niets te verliezen en "
        "lopen vlak. Alleen het geboosterde model houdt recente informatie "
        "vast, en alleen dat heeft een lijn die loopt.<br><br>"
        "Over de lichte lijn: per losse horizon zaagtandt de meting met "
        "periode zes. Dat is geen modelgedrag maar bemonstering: de afgiftes "
        "lopen om 00, 06, 12 en 18 UTC, dus elke horizon landt op vier "
        "kloktijden en die set verschuift met h, terwijl de fout sterk van "
        "het uur van de dag afhangt (zie Patronen). Zes opeenvolgende "
        "horizonnen dekken elke kloktijdmix precies één keer, dus de dikke "
        "lijnen middelen over zes horizonnen: dat verwijdert het artefact "
        "exact, zonder iets glad te strijken dat echt is.")

    sub("Betekenen de intervallen wat ze beweren?",
        "Van elke week: welk deel van de realisaties viel binnen de 80%-band "
        "die het model vooraf tekende. Bij een band die klopt, schommelt deze "
        "lijn rond de streepjeslijn.")
    band = bt[bt["lo80"].notna()]
    dek_week = (band.assign(raak=(band["y"] >= band["lo80"])
                            & (band["y"] <= band["hi80"]))
                .groupby("week")["raak"].mean() * 100)
    fig = go.Figure(go.Scatter(x=dek_week.index, y=dek_week.values,
                               name="werkelijke dekking", mode="lines",
                               line=dict(color=ACCENT, width=2.2)))
    fig.add_hline(y=80, line=dict(color=WARM, width=1.6, dash="dash"),
                  annotation_text="80% nominaal", annotation_position="top left")
    fig.add_vrect(x0="2020-03-15", x1="2020-07-01",
                  fillcolor="rgba(180, 83, 9, 0.07)", line_width=0,
                  annotation_text="lockdown", annotation_position="top left",
                  annotation_font=dict(size=10, color=MUTED))
    fig = layout(fig, "aandeel realisaties binnen de band (%)", "week", 320,
                 legend=False)
    st.plotly_chart(fig, **WIDE)
    dek_2020 = dek_week[dek_week.index >= "2020-03-15"]
    note(
        f"In rustige tijden schommelt de dekking net onder de 80: gemiddeld "
        f"{nl(meta['coverage80'], 1)}% over de hele backtest, dus de band is "
        "ook dan al een fractie te smal. Het echte verhaal begint in maart "
        f"2020: de dekking klapt naar een dieptepunt van "
        f"{nl(dek_2020.min(), 0)}% in een enkele week. De banden zijn de "
        "empirische kwantielen van de fouten die het model in eerdere folds "
        "daadwerkelijk maakte. Ze beschrijven het verleden dus goed, en houden "
        "precies zolang stand als de toekomst zich als dat verleden gedraagt. "
        "Toen het land in lockdown ging, bleef het model de oude wereld "
        "voorspellen met de oude zekerheid erbij, en dat tweede is het "
        "gevaarlijkst: een fout getal mét een te stellig interval. De remedie "
        "is niet een slimmere band maar een die zichzelf bijstelt: "
        "herkalibreren op een rollend venster, zodat een regimebreuk de band "
        "verbreedt in plaats van hem stilletjes ongeldig te maken. Langs de "
        "horizon is de dekking overigens vrijwel vlak, omdat de breedte per "
        "horizon apart is gefit; de tijd is de as waarop dit stuk kan gaan.",
        warn=True)

    sub("Onder de motorkap",
        "Wat de modellen zien, en waar het winnende model daadwerkelijk op "
        "leunt.")
    st.markdown("""
| Model | Wat het ziet | Waarom het er is |
|---|---|---|
| **Seizoensnaief** | hetzelfde uur een week eerder | Op een reeks met zoveel weekstructuur een echte concurrent, geen stroman. Wat dit niet verslaat, verdient het niet om te draaien. |
| **Seizoens-Fourier** | Fourier-termen voor dag-, week- en jaarcyclus, feestdagvlaggen, een trage trend; ridge op log-belasting, geen recente belasting | De zuivere kalenderblik. Zijn vlakke foutcurve geeft de helling van het geboosterde model betekenis. |
| **Gradient boosting** | origin-lags tot 168 uur, voortschrijdende gemiddelden, hetzelfde uur één en twee weken vóór het doeluur, kalender, en de horizon zelf | Eén model voor alle 168 horizons met h als feature, zodat de foutcurve iets is dat het model produceert in plaats van iets dat de opzet oplegt. |

Geen deep learning en geen Prophet: geen van beide verdient een plek op 41.640
uurwaarnemingen, en allebei kosten ze het vermogen om een getal uit te leggen.
    """)

    imp = load_importances().copy()

    def _groep(f: str) -> str:
        if f.startswith(("sin_d", "cos_d")):
            return "Fourier dagcyclus"
        if f.startswith(("sin_w", "cos_w")):
            return "Fourier weekcyclus"
        if f.startswith(("sin_y", "cos_y")):
            return "Fourier jaarcyclus"
        return {
            "tlag_168": "zelfde uur vorige week",
            "tlag_336": "zelfde uur twee weken terug",
            "is_holiday": "feestdagvlag",
            "is_yearend": "jaarwisseling",
            "is_weekend": "weekendvlag",
            "hour": "uur van de dag", "dow": "dag van de week",
            "doy": "dag van het jaar", "month": "maand",
            "h": "horizon", "h_day": "horizon in dagen",
            "oroll_24": "gemiddelde laatste 24 uur",
            "oroll_168": "gemiddelde laatste week",
            "odev_24_168": "recent niveau t.o.v. weekgemiddelde",
        }.get(f, "belasting rond afgifte (lags)" if f.startswith("olag") else f)

    imp["groep"] = imp["feature"].map(_groep)
    top = (imp.groupby("groep")["gain"].sum()
           .sort_values(ascending=True).tail(10))
    top = top / top.sum() * 100
    fig = go.Figure(go.Bar(
        x=top.values, y=top.index, orientation="h",
        marker_color=[ACCENT if g == "zelfde uur vorige week" else "#a8a29e"
                      for g in top.index],
        hovertemplate="%{y}: %{x:.1f}% van de gain<extra></extra>"))
    fig = layout(fig, "", "aandeel in de gain van het model (%)", 340,
                 legend=False)
    st.plotly_chart(fig, **WIDE)
    note(
        "Het model leunt overweldigend op <b>hetzelfde uur vorige week</b>, "
        "met de kalender en de feestdagvlag als correctie daarop. Dat is geen "
        "zwakte maar de verklaring van de hele demo: het model is in essentie "
        "een seizoensnaief die geleerd heeft wanneer vorige week de verkeerde "
        "referentie is. Twee kanttekeningen horen erbij. Gain-importances "
        "verdelen krediet willekeurig over gecorreleerde features, dus lees dit "
        "als een rangorde en geen exacte verdeling. En de feestdagvlag is één "
        "vlag voor alle feestdagen, terwijl de patronensectie laat zien dat "
        "Goede Vrijdag zich totaal anders gedraagt dan eerste kerstdag; een "
        "vlag per feestdag is de eerstvolgende verbetering die dit beeld "
        "aanwijst.")

    with st.expander("De folds, en waarom er geen train/test-split in deze repo zit"):
        st.dataframe(pd.DataFrame(meta["folds"]).rename(columns={
            "trained through": "getraind t/m", "first origin": "eerste origin",
            "last target": "laatste doeluur", "forecasts": "voorspellingen"}),
            hide_index=True, **WIDE)
        note("Uren door elkaar husselen laat een model interpoleren tussen het "
             "uur ervoor en het uur erna, wat prachtig scoort en niets waard is. "
             "Trainingsrijen stoppen hier een volle 168 uur vóór elke knip, niet "
             "op de knip zelf: een origin dichterbij dan de langste horizon zou "
             "een doeluur nodig hebben dat het model niet mag zien, en alleen "
             "die rijen weggooien zou de lange horizons stilletjes uit de "
             "trainingsset verwijderen. <code>tests/test_no_leakage.py</code> "
             "controleert de eigenschap in plaats van de bedoeling: het vervangt "
             "elke waarde ná de origin door onzin, bouwt de features opnieuw en "
             "eist dat ze identiek terugkomen.")

# --------------------------------------------------------------- regio ---
elif sectie == "Regio en congestie":
    cap = load_capaciteit()
    kop("Waar het probleem echt zit", "Landelijk gemiddeld is nergens",
        "De voorspelling in deze demo is landelijk en loopt tot 2020. Congestie "
        "is geen van beide: die is lokaal, en hij is nu. Hieronder staat de "
        "capaciteitskaart van de gezamenlijke netbeheerders, zodat het verschil "
        "tussen die twee vragen zichtbaar is in plaats van weggeschreven.")

    f1, f2 = st.columns([1, 1])
    beheerders = ["alle netbeheerders"] + sorted(
        b for b in cap["netbeheerder"].unique() if b and b != "onbekend")
    keuze = f1.selectbox("Netbeheerder", beheerders)
    richting = f2.radio("Richting", ["afname", "opwek"], horizontal=True,
                        help="Afname is verbruik, opwek is teruglevering.")
    kol = f"status_{richting}"
    wachtrij_kol = f"wachtrij_{richting}_mw"
    verzoek_kol = f"verzoeken_{richting}"

    sel = cap if keuze == "alle netbeheerders" else cap[cap["netbeheerder"] == keuze]
    # One row per supply area for anything counted: an area can be drawn as
    # several polygons and each carries the same attributes.
    ernst = {s: i for i, s in enumerate(
        ["Nog niet ingekleurd", "Beschikbaar", "Beperkt beschikbaar",
         "In onderzoek, met wachtrij", "Tekort, met wachtrij"])}
    gebieden = (sel.assign(_e=sel[kol].map(ernst)).sort_values("_e")
                .drop_duplicates("gebied_id", keep="last").drop(columns="_e"))
    tekort = gebieden[kol] == "Tekort, met wachtrij"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Voedingsgebieden", nl(len(gebieden)))
    m2.metric("Met een tekort", nl(int(tekort.sum())),
              nl(tekort.mean() * 100, 0) + "% van de gebieden", delta_color="off")
    m3.metric("Wachtrij", nl(gebieden[wachtrij_kol].sum(), 0, " MW"))
    m4.metric("Partijen in de wachtrij", nl(gebieden[verzoek_kol].sum(), 0))

    sub("De kaart",
        "Eén stip per getekend gebied, kleur is de status, grootte is de "
        "wachtrij in MW.")
    legenda([(s, STATUS_KLEUR[s]) for s in STATUS_VOLGORDE
             if s in set(sel[kol])])
    fig = go.Figure()
    for status in STATUS_VOLGORDE:
        deel = sel[sel[kol] == status]
        if deel.empty:
            continue
        fig.add_trace(go.Scattergeo(
            lon=deel["lon"], lat=deel["lat"], name=status, mode="markers",
            marker=dict(
                size=np.clip(6 + np.sqrt(deel[wachtrij_kol].clip(lower=0)) * 1.5,
                             6, 34),
                color=STATUS_KLEUR[status], opacity=0.78,
                line=dict(width=0.6, color="#ffffff")),
            customdata=np.stack([deel["gebied"], deel["netbeheerder"],
                                 deel[wachtrij_kol], deel[verzoek_kol]], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}"
                          "<br>wachtrij %{customdata[2]:,.0f} MW"
                          "<br>%{customdata[3]:,.0f} partijen<extra></extra>"))
    fig.update_geos(fitbounds="locations", resolution=50, projection_type="mercator",
                    showcountries=True, countrycolor="#d6d3d1",
                    showland=True, landcolor="#f7f6f5",
                    showocean=True, oceancolor="#eef2f6",
                    showlakes=False, showframe=False, coastlinecolor="#d6d3d1")
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=0, b=0),
                      showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="system-ui, sans-serif", size=12))
    st.plotly_chart(fig, **WIDE)

    g1, g2 = st.columns([3, 2])
    with g1:
        sub("De grootste wachtrijen")
        tab = (gebieden.nlargest(12, wachtrij_kol)[
            ["gebied", "netbeheerder", kol, wachtrij_kol, verzoek_kol]]
            .rename(columns={"gebied": "voedingsgebied", kol: "status",
                             wachtrij_kol: "wachtrij MW",
                             verzoek_kol: "partijen"}))
        st.dataframe(tab.round(0), hide_index=True, **WIDE)
    with g2:
        sub("Verdeling over de statussen")
        verdeling = gebieden[kol].value_counts().reindex(
            [s for s in STATUS_VOLGORDE if s in set(gebieden[kol])]).dropna()
        fig = go.Figure(go.Bar(
            x=verdeling.values, y=verdeling.index, orientation="h",
            marker_color=[STATUS_KLEUR[s] for s in verdeling.index],
            hovertemplate="%{y}: %{x} gebieden<extra></extra>"))
        fig = layout(fig, "", "aantal voedingsgebieden", 300, legend=False)
        st.plotly_chart(fig, **WIDE)

    landelijk = cap.assign(_e=cap["status_afname"].map(ernst)).sort_values("_e") \
        .drop_duplicates("gebied_id", keep="last")
    land_tekort = (landelijk["status_afname"] == "Tekort, met wachtrij").mean() * 100
    note(
        f"Landelijk zit {nl(land_tekort, 0)}% van de voedingsgebieden in een "
        "tekort aan transportcapaciteit voor afname. Dat is de reden dat dit "
        "onderwerp bestaat, en tegelijk de reden dat de landelijke voorspelling "
        "uit de rest van deze demo niet het antwoord is. Een landelijk "
        "uurgemiddelde van bijna 13 GW zegt niets over de vraag of een specifiek "
        "voedingsgebied vanmiddag om vijf uur boven zijn grens uitkomt. De "
        "methode draagt over, de getallen niet: op stationsniveau is de reeks "
        "grilliger, weegt één industriële aansluiting zwaar, en verschuift het "
        "doel van een gemiddelde naar een staart.")
    note(
        "<b>Hoe deze kaart en de rest van de demo op elkaar aansluiten.</b> Elk "
        "rood gebied hierboven is in het klein het probleem waar de sectie "
        "<em>Waarde en besluit</em> de gereedschappen voor laat zien: een harde "
        "grens, een vraagreeks die eromheen beweegt, en de vraag hoe lang van "
        "tevoren je een overschrijding ziet aankomen. Wat daar op de landelijke "
        "reeks staat, is wat ik voor één zo'n gebied zou bouwen: dezelfde "
        "leakagevrije backtest, dezelfde kansverdeling per horizon, dezelfde "
        "afweging tussen missen en loos uitrukken, maar dan op de reeks van dat "
        "gebied, met zijn eigen staartgedrag. De volgorde van aanpak staat vast: "
        "eerst de baseline op de gebiedsreeks, dan pas een model dat hem "
        "verslaat.")
    note(
        f"<b>Twee dingen over deze data die je moet weten voor je hem gebruikt.</b> "
        f"De kaart bestaat uit {nl(cap_meta.get('polygonen', len(cap)))} getekende "
        f"vlakken die samen {nl(cap_meta.get('gebieden', 0))} voedingsgebieden "
        "beschrijven, dus wie over de rijen sommeert telt dezelfde megawatts twee "
        f"of drie keer. En {nl(cap_meta.get('tegenstrijdig', 0))} gebieden dragen "
        "een verschillende status op verschillende vlakken. Hierboven telt steeds "
        "de zwaarste status per gebied, wat een keuze is en geen feit: het is de "
        "veilige kant voor een capaciteitsvraag. De statuscodes zelf staan nergens "
        "in de servicebeschrijving; hun betekenis komt uit de tekenregel van de "
        "uitgever zelf, en die keert de intuïtie om, want code 0 betekent niet "
        "\"onbekend\" maar \"ruimte beschikbaar\".", warn=True)

# ------------------------------------------------------ waarde en besluit ---
elif sectie == "Waarde en besluit":
    kop("Beslissen met een grens", "Zie je het op tijd aankomen?",
        "Een netbeheerder vraagt zelden hoe nauwkeurig een voorspelling is. De "
        "vraag is of een grens overschreden wordt, hoe zeker dat is, en of je "
        "er nog iets aan kunt doen. Deze pagina zet de voorspelling om in dat "
        "besluit.")

    note(
        "<b>Lees het als een dienstrooster.</b> Stel je een voedingsgebied voor "
        "dat een bepaald vermogen aankan. Elke dag draait het model, en soms "
        "zegt het: morgen om vijf uur wordt het krap. Dan stuur je iemand, of je "
        "belt een klant om terug te schakelen. Dat kost geld, dus je wilt niet "
        "voor elk klein risico uitrukken; maar een overschrijding missen kost "
        "meer.<br><br>"
        "Hieronder kies je drie dingen: hoe ver vooruit je kijkt, waar de grens "
        "ligt, en vanaf welke kans je in actie komt. De app rekent daarna op de "
        "echte backtest uit wat die keuze had opgeleverd: hoeveel "
        "overschrijdingen je had zien aankomen, hoeveel je had gemist, en hoe "
        "vaak je voor niets was uitgerukt.")

    k1, k2, k3 = st.columns(3)
    lead = k1.selectbox("Hoe ver vooruit kijk je?", [24, 48, 72, 168], index=0,
                        format_func=lambda x: f"{x} uur ({x // 24} dag"
                                              + ("en)" if x // 24 > 1 else ")"))
    bt = load_backtest()
    y_lead = bt.loc[bt["h"] == lead, "y"]
    grens = k2.slider("Wat kan het gebied aan? (MW)",
                      int(y_lead.quantile(0.80) // 100 * 100),
                      int(y_lead.max() // 100 * 100),
                      int(y_lead.quantile(0.95) // 100 * 100), step=100)
    drempel = k3.slider("Vanaf welke kans kom je in actie?", 5, 95, 30, 5,
                        format="%d%%")
    d = exceedance(lead, float(grens))
    nu = alarm_stats(d, (d["kans"] >= drempel / 100).to_numpy())
    n_over = int(d["overschreden"].sum())
    maanden = (d["target"].max() - d["target"].min()).days / 30.44

    sub("Wat die keuze had opgeleverd",
        f"Over {nl(maanden, 0)} maanden backtest, waarin de grens "
        f"{nl(n_over)} keer werd overschreden.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Op tijd gezien", nl(nu["raak"]),
              nl(nu["recall"], 0) + "% van alle overschrijdingen",
              delta_color="off")
    m2.metric("Gemist", nl(nu["gemist"]),
              nl(100 - nu["recall"], 0) + "% van alle overschrijdingen",
              delta_color="off")
    m3.metric("Vals alarm", nl(nu["vals"]),
              nl(nu["vals_pm"], 1) + " keer per maand", delta_color="off")
    m4.metric("Raak als je uitrukt", nl(nu["precisie"], 0, "%"),
              "van de alarmen was terecht", delta_color="off")

    fig = go.Figure(go.Bar(
        x=[nu["raak"], nu["gemist"], nu["vals"]],
        y=["op tijd gezien", "gemist", "vals alarm"], orientation="h",
        marker_color=["#15803d", "#b91c1c", "#a8a29e"],
        text=[nl(nu["raak"]), nl(nu["gemist"]), nl(nu["vals"])],
        textposition="outside",
        hovertemplate="%{y}: %{x} uren<extra></extra>"))
    fig = layout(fig, "", "aantal uren in de backtest", 240, legend=False)
    st.plotly_chart(fig, **WIDE)
    note(
        f"Met een drempel van {drempel}% ruk je uit zodra het model de kans op "
        f"overschrijding hoger inschat dan {drempel} op 100. Dat levert "
        f"{nl(nu['raak'])} van de {nl(n_over)} overschrijdingen op tijd op, "
        f"{nl(nu['gemist'])} glippen er doorheen, en je gaat "
        f"{nl(nu['vals_pm'], 1)} keer per maand voor niets. Zet de drempel lager "
        "en je mist minder maar rukt vaker voor niets uit; zet hem hoger en het "
        "omgekeerde. Er is geen goed antwoord zonder te weten wat een gemiste "
        "overschrijding kost tegenover een loze rit, en dat is een gesprek met "
        "de operatie en niet met de modelleur.")

    sub("Alle drempels tegelijk, en waarom een kans beter is dan een getal",
        "Elk punt op de lijn is één drempelinstelling. Rechtsboven is alles "
        "vangen tegen veel loos alarm, linksonder is bijna nooit uitrukken en "
        "veel missen.")
    drempels = np.round(np.arange(0.02, 0.99, 0.02), 2)
    curve = [alarm_stats(d, (d["kans"] >= t).to_numpy()) for t in drempels]
    punt = alarm_stats(d, (d["pred_lgbm"] > grens).to_numpy())
    naief = alarm_stats(d, (d["pred_seasonal_naive"] > grens).to_numpy())
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[c["vals_pm"] for c in curve], y=[c["recall"] for c in curve],
        mode="lines", name="alle kansdrempels", line=dict(color=ACCENT, width=2.8),
        text=[f"drempel {int(t * 100)}%" for t in drempels],
        hovertemplate="%{text}<br>%{y:.0f}% gevonden"
                      "<br>%{x:.1f} vals alarm per maand<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[nu["vals_pm"]], y=[nu["recall"]], mode="markers",
        name=f"jouw keuze: {drempel}%",
        marker=dict(size=17, color=ACCENT, symbol="circle",
                    line=dict(width=2.5, color="#ffffff"))))
    fig.add_trace(go.Scatter(x=[punt["vals_pm"]], y=[punt["recall"]], mode="markers",
                             name="zonder kans: alarm als de voorspelling boven de grens ligt",
                             marker=dict(size=14, color=WARM, symbol="diamond")))
    fig.add_trace(go.Scatter(x=[naief["vals_pm"]], y=[naief["recall"]], mode="markers",
                             name="idem, maar met seizoensnaief",
                             marker=dict(size=11, color="#a8a29e", symbol="square")))
    fig = layout(fig, "aandeel overschrijdingen op tijd gezien (%)",
                 "vals alarm per maand", 380)
    fig.update_layout(hovermode="closest")
    st.plotly_chart(fig, **WIDE)
    note(
        "Hier zit het hele punt van deze pagina. Wie alleen een puntvoorspelling "
        "heeft, kan maar één regel maken: alarm als het voorspelde getal boven de "
        f"grens ligt. Dat is de ruit, en die vindt {nl(punt['recall'], 0)}% van de "
        f"overschrijdingen bij {nl(punt['vals_pm'], 1)} vals alarm per maand. Je "
        "zit vast aan dat ene punt. Met een kans in plaats van een getal krijg je "
        "de hele lijn, en dan kies je zelf of je liever niets mist of liever niet "
        "voor niets uitrukt. Dezelfde regel op basis van seizoensnaief staat er "
        "als vierkant naast, zodat zichtbaar blijft wat het model daadwerkelijk "
        "toevoegt.")

    sub("Klopt die kans eigenlijk wel?",
        "Neem alle uren waarvan het model zei \"30% kans\". Als het model deugt, "
        "ging het in ongeveer 30 van de 100 van die uren ook echt mis. Dat is "
        "wat deze grafiek nakijkt: voorspelde kans op de horizontale as, het "
        "werkelijke aandeel op de verticale. Op de streepjeslijn klopt de kans "
        "precies; eronder belooft het model meer zekerheid dan het waarmaakt.")
    kal = d.groupby(pd.cut(d["kans"], np.arange(0, 1.05, 0.1)),
                    observed=True).agg(voorspeld=("kans", "mean"),
                                       werkelijk=("overschreden", "mean"),
                                       n=("kans", "size")).dropna()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                             line=dict(color=WARM, width=1.4, dash="dash")))
    fig.add_trace(go.Scatter(
        x=kal["voorspeld"], y=kal["werkelijk"], mode="lines+markers", name="gemeten",
        line=dict(color=ACCENT, width=2.8),
        marker=dict(size=[float(min(7 + n / 40, 22)) for n in kal["n"]]),
        text=[f"{int(n)} uren" for n in kal["n"]],
        hovertemplate="voorspeld %{x:.0%}<br>werkelijk %{y:.0%}"
                      "<br>%{text}<extra></extra>"))
    fig = layout(fig, "werkelijk aandeel overschrijdingen", "voorspelde kans", 340)
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(tickformat=".0%")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, **WIDE)

    top = d[d["y"] >= d["y"].quantile(0.90)]
    bias_top = float((top["pred_lgbm"] - top["y"]).mean())
    note(
        "Dit is de plek waar ik het model tegenspreek. Over de hele backtest is "
        f"de bias {nl(overall['lgbm']['bias'], 0)} MW, praktisch nul. Maar in het "
        "hoogste deciel van de belasting, precies de uren die tegen een "
        f"capaciteitsgrens aan liggen, voorspelt hetzelfde model gemiddeld "
        f"{nl(abs(bias_top), 0)} MW <em>te laag</em>. Een kop-MAPE maakt dat "
        "volledig onzichtbaar, en het is de verkeerde richting: het model is het "
        "meest optimistisch op de momenten waarop optimisme het duurst is. Voor "
        "een landelijke reeks is dat een voetnoot, voor een station met een harde "
        "grens de kern van de zaak. De eerste correctie die ik zou aanbrengen is "
        "een verliesfunctie die onderschatting in de staart zwaarder beprijst, of "
        "een apart kwantielmodel voor de bovenkant van de verdeling.", warn=True)

    sub("De prijs van een fout",
        "Fouten staan in MW gedurende één uur, dus in MWh. Eén horizon tegelijk, "
        "want de backtest bevat 168 voorspellingen per uur en daar overheen "
        "sommeren zou dezelfde fout tientallen keren beprijzen.")
    p1, p2, p3 = st.columns([1, 1, 2])
    producten = {"Dag vooruit (h = 24)": 24, "Twee dagen (h = 48)": 48,
                 "Week vooruit (h = 168)": 168}
    product = p1.selectbox("Voorspelproduct", list(producten), index=0)
    eur_onder = p2.number_input("€ per MWh te laag voorspeld", 0.0, 5000.0, 180.0, 10.0)
    eur_boven = p2.number_input("€ per MWh te hoog voorspeld", 0.0, 5000.0, 60.0, 10.0)
    p3.markdown("<div class='note'>De standaardwaarden maken te laag voorspellen "
                "drie keer zo duur als te hoog. Dat is een plaatshouder en geen "
                "marktprijs.</div>", unsafe_allow_html=True)

    w = bt[bt["h"] == producten[product]]
    yv = w["y"].to_numpy(float)
    uren = len(yv)

    def jaarkosten(pred) -> float:
        e = np.asarray(pred, float) - yv
        return float((np.maximum(-e, 0).sum() * eur_onder
                      + np.maximum(e, 0).sum() * eur_boven) / uren * 8766.0)

    kosten = {m: jaarkosten(w[f"pred_{m}"]) for m in MODELS}
    besparing = kosten["seasonal_naive"] - kosten["lgbm"]
    vermeden = float((np.abs(w["pred_seasonal_naive"].to_numpy(float) - yv).sum()
                      - np.abs(w["pred_lgbm"].to_numpy(float) - yv).sum())
                     / uren * 8766.0)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Seizoensnaief, foutkosten",
              "€ " + nl(kosten["seasonal_naive"] / 1e6, 0, " mln/jr"))
    m2.metric("Gradient boosting, foutkosten",
              "€ " + nl(kosten["lgbm"] / 1e6, 0, " mln/jr"))
    m3.metric("Verschil", "€ " + nl(besparing / 1e6, 0, " mln/jr"),
              nl(besparing / kosten["seasonal_naive"] * 100, 1) + "% lager")
    m4.metric("Vermeden voorspelfout", nl(vermeden / 1000, 0, " GWh/jr"),
              "zonder prijskaartje", delta_color="off")
    note(
        "Het vierde getal draagt geen prijs, en daarom staat het er. Over een "
        "jaar neemt het model dat aantal GWh absolute voorspelfout weg ten "
        "opzichte van seizoensnaief, en dat blijft staan wat iemand ook besluit "
        "dat een uur fout waard is.<br><br>"
        "Lees de niveaus met zorg. Dit is het hele Nederlandse net bij een "
        f"gemiddelde belasting van {nl(reeks['load_mean'] / 1000, 1)} GW, dus een "
        f"fout van {nl(overall['lgbm']['mape'], 1)}% is ruwweg "
        f"{nl(overall['lgbm']['mae'], 0)} MW elk uur van het jaar, en elke prijs "
        "per MWh maakt daar per definitie een groot getal van. Het beprijst "
        "bovendien elke MWh fout tegen het volle tarief, wat geen enkel "
        "afrekenregime doet. Het cijfer dat die bezwaren overleeft is het "
        "<em>verschil</em> tussen twee voorspellingen op identieke uren, niet het "
        "niveau van een van beide.")

# ----------------------------------------------------------- monitoring ---
elif sectie == "Monitoring":
    kop("De dag na livegang", "Hoe dit aan de muur zou hangen",
        "Een model dat vandaag klopt, klopt morgen niet vanzelf. Dit is de "
        "pagina die dat zichtbaar maakt, en het is de enige pagina waar deze "
        "demo laat zien wat er gebeurt als de werkelijkheid van regime wisselt.")

    wk = by_week()
    drempel = float(wk["mape_lgbm"].iloc[:52].mean()
                    + 2 * wk["mape_lgbm"].iloc[:52].std())
    gemarkeerd = wk[wk["mape_lgbm"] > drempel]
    in_2020 = int((gemarkeerd["week"].dt.year == 2020).sum())
    eerste = gemarkeerd[gemarkeerd["week"].dt.year == 2020]["week"].min()
    slechtste = wk.loc[wk["bias"].idxmin()]

    m1, m2, m3 = st.columns(3)
    m1.metric("Weken boven de alarmdrempel", f"{len(gemarkeerd)} van {len(wk)}")
    m2.metric("Daarvan in 2020", nl(in_2020),
              f"vanaf {eerste.day} {MAANDEN[eerste.month]}", delta_color="off")
    m3.metric("Grootste weekbias", nl(slechtste["bias"], 0, " MW"),
              f"week van {slechtste['week'].day} {MAANDEN[slechtste['week'].month]}"
              f" {slechtste['week'].year}", delta_color="off")

    sub("Weekfout, model tegen baseline")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wk["week"], y=wk["mape_seasonal_naive"],
                             name="Seizoensnaief", mode="lines",
                             line=dict(color="#a8a29e", width=1.4)))
    fig.add_trace(go.Scatter(x=wk["week"], y=wk["mape_lgbm"],
                             name="Gradient boosting", mode="lines",
                             line=dict(color=ACCENT, width=2.6)))
    fig.add_hline(y=drempel, line=dict(color=WARM, width=1.4, dash="dash"),
                  annotation_text="alarm bij " + nl(drempel, 1) + "%",
                  annotation_position="top left")
    fig = layout(fig, "MAPE per week (%)", "week", 340)
    st.plotly_chart(fig, **WIDE)

    sub("Weekbias, het teken dat een regimebreuk verraadt")
    fig = go.Figure(go.Bar(x=wk["week"], y=wk["bias"], name="bias per week",
                           marker_color=np.where(wk["bias"] > 0, ACCENT, WARM)))
    fig.add_hline(y=0, line=dict(color="#a8a29e", width=1))
    fig = layout(fig, "gemiddelde fout per week (MW)", "week", 280, legend=False)
    st.plotly_chart(fig, **WIDE)

    note(
        "De alarmlijn ligt twee standaarddeviaties boven de weekfout van het "
        f"eerste jaar, het soort regel dat een team ook echt in productie zet. "
        f"Hij gaat af in {len(gemarkeerd)} van de {len(wk)} weken, en die liggen "
        f"niet verspreid: {in_2020} ervan vallen in 2020, vrijwel aaneengesloten "
        f"vanaf de week van {eerste.day} {MAANDEN[eerste.month]} {eerste.year}. De "
        "vraag zakte in twee weken ruwweg een tiende, en elk model dat op 2016 tot "
        "en met 2019 getraind was bleef het land voorspellen dat niet meer "
        "bestond. De weekbias wordt positief en blijft dat, en dat is wat een "
        "kapot model onderscheidt van een luidruchtig model: ruis wisselt van "
        "teken, een regimebreuk niet. Een kwartaalhertraining is daar veel te "
        "traag voor, en het antwoord is geen beter model maar een poort. Boven de "
        "drempel wacht de voorspelling op een mens.<br><br>"
        "De grootste misser wijst de andere kant op, en naar mij. In de week van "
        f"{slechtste['week'].day} {MAANDEN[slechtste['week'].month]} "
        f"{slechtste['week'].year} liep het model {nl(abs(slechtste['bias']), 0)} "
        "MW <em>onder</em> de realisatie, de enige aanhoudende negatieve bias in "
        "de hele reeks. Dat was de heetste week van de hele periode, met een piek "
        "van 33 °C en zes van de acht warmste augustusdagen van vijf jaar erin. De "
        "belasting liep op naar 13,3 GW tegen 11,5 GW de week ervoor, en een model "
        "dat niet naar temperatuur mag kijken had geen enkele manier om dat te "
        "zien aankomen. De keuze om weer weg te laten koopt een eerlijke horizon "
        "en kost precies dit, en een monitoringpagina die dat verstopte zou niets "
        "waard zijn.", warn=True)

    sub("De grootste missers, stuk voor stuk",
        "De acht dagen waarop de dag-vooruit voorspelling er gemiddeld het "
        "verst naast zat, met erbij wat er die dag aan de hand was.")
    bt = load_backtest()
    s = load_series()
    d24 = bt[bt["h"] == 24].copy()
    d24["datum"] = d24["target"].dt.tz_convert(TZ).dt.date
    per_dag = d24.groupby("datum").agg(
        fout=("pred_lgbm", lambda x: 0.0), y=("y", "mean"),
        p=("pred_lgbm", "mean")).reset_index()
    per_dag["fout"] = per_dag["p"] - per_dag["y"]
    abs_fout = (d24.assign(ae=(d24["pred_lgbm"] - d24["y"]).abs())
                .groupby("datum")["ae"].mean())
    per_dag["abs_fout"] = per_dag["datum"].map(abs_fout)
    ctx = s.copy()
    ctx["datum"] = ctx["local"].dt.date
    per_ctx = ctx.groupby("datum").agg(
        temp=("temp_c", "mean"),
        feestdag=("holiday_name", lambda x: next((v for v in x if v), "")),
        dagsoort=("dagsoort", "first")).reset_index()
    ergste = (per_dag.merge(per_ctx, on="datum")
              .nlargest(8, "abs_fout"))

    def _context(r) -> str:
        if r["feestdag"]:
            return r["feestdag"]
        dt = pd.Timestamp(r["datum"])
        if pd.Timestamp("2020-03-15") <= dt <= pd.Timestamp("2020-06-01"):
            return "lockdown"
        if r["temp"] >= 22:
            return f"hittegolf ({r['temp']:.0f} °C daggemiddeld)"
        return r["dagsoort"]

    ergste["wat er speelde"] = ergste.apply(_context, axis=1)
    st.dataframe(
        ergste.assign(datum=ergste["datum"].astype(str))
        [["datum", "wat er speelde", "y", "p", "fout", "abs_fout"]]
        .rename(columns={"y": "werkelijk MW", "p": "voorspeld MW",
                         "fout": "bias MW", "abs_fout": "|fout| MW"})
        .round(0), hide_index=True, **WIDE)
    note(
        "Geen van deze dagen is een raadsel, en dat is het punt. De missers "
        "clusteren op drie verklaarbare situaties: de lockdownweken, de "
        "hittegolf, en feestdagconstellaties die zelden in de trainingsdata "
        "voorkomen. Een misser die je kunt benoemen is een werklijst; een "
        "misser die nergens bij hoort is pas echt zorgwekkend. Deze tabel is "
        "wat ik in productie elke maandag als eerste zou bekijken.")

    with st.expander("Weken boven de alarmdrempel"):
        st.dataframe(
            gemarkeerd.assign(week=gemarkeerd["week"].dt.strftime("%Y-%m-%d"))
            .rename(columns={"mape_lgbm": "MAPE model %",
                             "mape_seasonal_naive": "MAPE seizoensnaief %",
                             "bias": "bias MW"}).round(2),
            hide_index=True, **WIDE)

st.markdown(f"""
<div class="footer">
Ismail Arslan ·
<a href="https://ismailarslan.tech">ismailarslan.tech</a> ·
<a href="mailto:contact@ismailarslan.tech">contact@ismailarslan.tech</a> ·
<a href="https://linkedin.com/in/iqzarslan">linkedin.com/in/iqzarslan</a><br>
Belastingdata: Open Power System Data, time series 2020-10-06 (CC-BY 4.0),
oorspronkelijk ENTSO-E Transparency. Temperatuur: Open-Meteo historisch archief
(CC-BY 4.0). Capaciteitskaart elektriciteitsnet: Netbeheer Nederland en Esri
Nederland, opgehaald {meta.get('capaciteit_opgehaald', '')}, gebruikt onder de
Esri Nederland Terms of Use; alleen een afgeleide samenvatting is opgenomen, geen
vlakgeometrie. Educatief portfolioproject, niet verbonden aan enige netbeheerder.
</div>
""", unsafe_allow_html=True)
