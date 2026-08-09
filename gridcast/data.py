"""Sources, cleaning and the calendar. Build-time only: the app never calls this.

Two public sources, both keyless:

  load      Open Power System Data, time series package 2020-10-06, CC-BY 4.0.
            Column NL_load_actual_entsoe_transparency, originally ENTSO-E
            Transparency. Hourly, UTC, 2015-01-01 .. 2020-09-30.
  weather   Open-Meteo historical archive (ERA5), CC-BY 4.0. Hourly 2 m
            temperature at De Bilt. Used for exploration only, never as a
            model feature — see README.

The series is cut at 2016-01-01. Every month of 2015 sits 1.7 to 2.0 GW below
the same month of 2016 while 2016 through 2019 are near identical, which is a
reporting change rather than demand. `level_break_evidence()` returns the
numbers the app shows to justify that cut.
"""
from __future__ import annotations

import pathlib

import holidays
import numpy as np
import pandas as pd
import requests

OPSD_URL = ("https://data.open-power-system-data.org/time_series/2020-10-06/"
            "time_series_60min_singleindex.csv")
OPSD_COLUMNS = {
    "NL_load_actual_entsoe_transparency": "load",
    "NL_load_forecast_entsoe_transparency": "tso_day_ahead",
    "NL_solar_generation_actual": "solar",
    "NL_wind_generation_actual": "wind",
}
METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
DE_BILT = (52.10, 5.18)

START = pd.Timestamp("2016-01-01", tz="UTC")
END = pd.Timestamp("2020-09-30 23:00", tz="UTC")
TZ = "Europe/Amsterdam"

RAW = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw"


def _cached(name: str, fetch) -> pd.DataFrame:
    """Download once into data/raw (gitignored), reuse afterwards."""
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / name
    if not path.exists():
        fetch(path)
    return path


def download_opsd(path: pathlib.Path) -> None:
    with requests.get(OPSD_URL, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(path, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1 << 20):
                fh.write(chunk)


def load_raw_opsd() -> pd.DataFrame:
    """Full NL slice, uncut, so the app can show the 2015 break itself."""
    path = _cached("time_series_60min_singleindex.csv", download_opsd)
    df = pd.read_csv(path, usecols=["utc_timestamp", *OPSD_COLUMNS],
                     parse_dates=["utc_timestamp"], index_col="utc_timestamp")
    return df.rename(columns=OPSD_COLUMNS).sort_index()


def load_weather(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    name = f"open_meteo_{start.date()}_{end.date()}.csv"

    def fetch(path):
        params = {
            "latitude": DE_BILT[0], "longitude": DE_BILT[1],
            "start_date": str(start.date()), "end_date": str(end.date()),
            "hourly": "temperature_2m", "timezone": "UTC",
        }
        r = requests.get(METEO_URL, params=params, timeout=300)
        r.raise_for_status()
        pd.DataFrame(r.json()["hourly"]).to_csv(path, index=False)

    path = _cached(name, fetch)
    w = pd.read_csv(path, parse_dates=["time"])
    w["time"] = pd.to_datetime(w["time"], utc=True)
    return w.set_index("time")["temperature_2m"].rename("temp_c")


def level_break_evidence(raw: pd.DataFrame) -> pd.DataFrame:
    """Mean load per calendar month per year, the table behind the 2016 cut."""
    idx = raw.index.tz_convert(TZ)
    out = (raw["load"].groupby([idx.year, idx.month]).mean()
           .unstack(0).round(0))
    out.index.name = "month"
    out.columns.name = "year"
    return out


def calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar facts of each timestamp, in local time. No load involved, so
    this is knowable arbitrarily far ahead — unlike weather."""
    local = index.tz_convert(TZ)
    nl = holidays.country_holidays("NL", years=range(local.year.min(),
                                                     local.year.max() + 1))
    dates = local.date
    is_holiday = np.fromiter((d in nl for d in dates), dtype=bool,
                             count=len(dates))
    # The name, not just the flag. Easter, Ascension and Whitsun move every
    # year, so a date cannot identify which holiday it was, and grouping by
    # date splits one holiday across as many bars as there are years.
    holiday_name = np.array([nl.get(d, "") for d in dates], dtype=object)
    hour = np.asarray(local.hour)
    dow = np.asarray(local.dayofweek)
    month = np.asarray(local.month)
    day = np.asarray(local.day)
    return pd.DataFrame({
        "hour": hour,
        "dow": dow,
        "month": month,
        "doy": np.asarray(local.dayofyear),
        "is_weekend": dow >= 5,
        "is_holiday": is_holiday,
        "holiday_name": holiday_name,
        # The days between Christmas and New Year behave like nothing else in
        # the year and are not all public holidays.
        "is_yearend": ((month == 12) & (day >= 24)) | ((month == 1) & (day <= 2)),
    }, index=index)


def build_series() -> pd.DataFrame:
    """The one hourly frame everything downstream uses: gap-free, UTC, cut at
    the level break, with calendar and temperature joined on."""
    raw = load_raw_opsd()
    df = raw.loc[START:END].copy()

    expected = pd.date_range(START, END, freq="H", tz="UTC")
    if not df.index.equals(expected):
        raise ValueError(f"load series is not gap-free hourly: "
                         f"{len(df)} rows, expected {len(expected)}")
    if df["load"].isna().any():
        raise ValueError("load has missing values after the cut")

    df = df.join(load_weather(START, END))
    df = df.join(calendar(df.index))
    return df


def summarise(df: pd.DataFrame) -> dict:
    """Numbers the README and the app quote, computed rather than typed."""
    local = df.index.tz_convert(TZ)
    work = df.loc[(local.dayofweek < 5) & ~df["is_holiday"], "load"]
    hol = df.loc[df["is_holiday"], "load"]
    y2019 = df.loc["2019"]
    l2019 = y2019.index.tz_convert(TZ)
    by_hour = y2019.groupby(l2019.hour)["load"].mean()
    return {
        "start": str(df.index.min()),
        "end": str(df.index.max()),
        "hours": int(len(df)),
        "load_mean": float(df["load"].mean()),
        "load_min": float(df["load"].min()),
        "load_max": float(df["load"].max()),
        "holiday_gap_pct": float(hol.mean() / work.mean() - 1) * 100,
        "trough_hour": int(by_hour.idxmin()),
        "trough_mw": float(by_hour.min()),
        "peak_hour": int(by_hour.idxmax()),
        "peak_mw": float(by_hour.max()),
    }
