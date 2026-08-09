"""Where the grid is actually full, from the operators' own capacity map.

The load series in this repo is national and ends in 2020. Congestion is
neither: it is local, and it is a live problem. This module pulls the public
Capaciteitskaart so the app can put the two side by side, which is the honest
way to say that a national average is not the operational question.

Source
------
Capaciteitskaart elektriciteitsnet, Netbeheer Nederland, published as an ArcGIS
feature service by Esri Nederland. One record per voedingsgebied (supply area)
with a status for consumption (`afname`) and for feed-in (`opwek`), the queue in
MW and the number of parties waiting.

The status codes carry no domain in the service metadata. Their meaning is taken
verbatim from the publisher's own renderer expression rather than guessed: note
that 0 means capacity is available, which is the opposite of what the numbering
suggests. `STATUS_EXPRESSION` records that expression so the mapping below can
be checked against its source.

Terms
-----
Esri Nederland Terms of Use, not an open licence. Only a derived summary is
written to data/processed: area name, operator, status, queue and a centroid.
The polygons themselves are not redistributed.
"""
from __future__ import annotations

import pandas as pd
import requests

SERVICE = ("https://services.arcgis.com/nSZVuSZjHpEZZbRo/arcgis/rest/services/"
           "Capaciteitskaart_elektriciteitsnet_v2_afname/FeatureServer/0/query")

# Copied from the service renderer (drawingInfo.renderer.valueExpression) so the
# mapping below can be audited against the publisher rather than trusted.
STATUS_EXPRESSION = """
If($feature.afname == 0 && $feature.voedingsgebied_naam != "0") return 'Transportcapaciteit beschikbaar zonder wachtrij'
If($feature.afname == 0 && $feature.voedingsgebied_naam == "0") return 'Kleur wordt later toegevoegd'
If($feature.afname == -1 && $feature.voedingsgebied_naam == "0") return 'Kleur wordt later toegevoegd'
If($feature.afname == 1) return 'Transportcapaciteit beperkt beschikbaar zonder wachtrij'
If($feature.afname == 2) return 'Gebied is in onderzoek met wachtrij'
If($feature.afname == 3) return 'Tekort aan transportcapaciteit met wachtrij'
""".strip()

STATUS = {
    0: "Beschikbaar",
    1: "Beperkt beschikbaar",
    2: "In onderzoek, met wachtrij",
    3: "Tekort, met wachtrij",
}
ONBEKEND = "Nog niet ingekleurd"
# Ordered worst last, so a sort puts the problem at the top of a chart.
VOLGORDE = [ONBEKEND, "Beschikbaar", "Beperkt beschikbaar",
            "In onderzoek, met wachtrij", "Tekort, met wachtrij"]
# The publisher draws "available" white, which disappears on a light map. Green
# reads the same way to anyone who has seen a traffic light; the labels are the
# publisher's own.
KLEUR = {
    ONBEKEND: "#d6d3d1",
    "Beschikbaar": "#15803d",
    "Beperkt beschikbaar": "#eab308",
    "In onderzoek, met wachtrij": "#ea580c",
    "Tekort, met wachtrij": "#b91c1c",
}

FIELDS = ["afname", "opwek", "voedingsgebied_naam", "voedingsgebied_id", "RNB",
          "unieke_verzoeken_afname", "wachtrij_afname",
          "unieke_verzoeken_invoeding", "wachtrij_invoeding"]


def _status(code, naam: str) -> str:
    if naam in (None, "", "0") or code is None or code < 0:
        return ONBEKEND
    return STATUS.get(int(code), ONBEKEND)


def fetch() -> pd.DataFrame:
    """One row per supply area, with a centroid instead of a polygon."""
    params = {
        "where": "1=1", "outFields": ",".join(FIELDS), "returnGeometry": "false",
        "returnCentroid": "true", "outSR": 4326, "f": "json",
        "resultRecordCount": 2000,
    }
    r = requests.get(SERVICE, params=params, timeout=180)
    r.raise_for_status()
    payload = r.json()
    if payload.get("exceededTransferLimit"):
        raise RuntimeError("capacity map paged; add an offset loop")

    rows = []
    for f in payload["features"]:
        a = f["attributes"]
        c = f.get("centroid") or {}
        if c.get("x") is None:
            continue
        naam = a.get("voedingsgebied_naam")
        rows.append({
            "gebied": naam,
            "gebied_id": str(a.get("voedingsgebied_id")),
            "netbeheerder": a.get("RNB") or "onbekend",
            "status_afname": _status(a.get("afname"), naam),
            "status_opwek": _status(a.get("opwek"), naam),
            "wachtrij_afname_mw": a.get("wachtrij_afname") or 0.0,
            "wachtrij_opwek_mw": a.get("wachtrij_invoeding") or 0.0,
            "verzoeken_afname": a.get("unieke_verzoeken_afname") or 0.0,
            "verzoeken_opwek": a.get("unieke_verzoeken_invoeding") or 0.0,
            "lon": float(c["x"]),
            "lat": float(c["y"]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("capacity map returned no usable records")
    return df


# Severity order for resolving an area drawn as several polygons. Worst wins:
# if any polygon of an area reports a shortage, the area has a shortage.
ERNST = {ONBEKEND: 0, "Beschikbaar": 1, "Beperkt beschikbaar": 2,
         "In onderzoek, met wachtrij": 3, "Tekort, met wachtrij": 4}


def per_area(df: pd.DataFrame) -> pd.DataFrame:
    """One row per supply area, for anything that is counted or summed.

    A supply area can be drawn as several polygons and each one carries the
    area's attributes, so summing the queue over rows counts the same megawatts
    two or three times: 927 polygons cover 571 areas. Totals are taken on this
    frame; the map keeps the polygon rows because each needs its own dot.

    Roughly a fifth of the areas carry a different status on different polygons.
    The most severe one wins, which is a choice and not a fact: it is the safe
    direction for a capacity question, and `conflicts()` reports how often it
    had to be made.
    """
    ranked = df.assign(_ernst=df["status_afname"].map(ERNST))
    return (ranked.sort_values("_ernst")
            .drop_duplicates("gebied_id", keep="last")
            .drop(columns="_ernst"))


def conflicts(df: pd.DataFrame) -> int:
    """Areas whose polygons disagree about their own status."""
    return int((df.groupby("gebied_id")["status_afname"].nunique() > 1).sum())


def summarise(df: pd.DataFrame) -> dict:
    polygonen = int(len(df))
    tegenstrijdig = conflicts(df)
    df = per_area(df)
    tekort = df["status_afname"] == "Tekort, met wachtrij"
    stedin = df["netbeheerder"].str.contains("Stedin", na=False)
    return {
        "polygonen": polygonen,
        "tegenstrijdig": tegenstrijdig,
        "gebieden": int(len(df)),
        "tekort": int(tekort.sum()),
        "tekort_pct": float(tekort.mean() * 100),
        "wachtrij_mw": float(df["wachtrij_afname_mw"].sum()),
        "verzoeken": int(df["verzoeken_afname"].sum()),
        "netbeheerders": df["netbeheerder"].value_counts().to_dict(),
        "stedin_gebieden": int(stedin.sum()),
        "stedin_tekort_pct": float(tekort[stedin].mean() * 100) if stedin.any() else 0.0,
    }
