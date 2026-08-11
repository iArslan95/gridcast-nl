"""GridCast assistant: a grounded LLM chat over the committed backtest.

Architecture: context injection, same as the other demos in this portfolio.
Every question is answered by an LLM (Groq, Llama 3.3 70B) that receives a
serialized digest of the artifacts — scores, folds, findings, the capacity
snapshot — in its system prompt. The app trains nothing and the assistant
invents nothing: what is not in the digest, it is told to say it cannot know.
The API key lives in Streamlit secrets, never in the repo; without a key the
app runs fine and the panel explains how to enable it.
"""
from __future__ import annotations

import json
import os
import re
import time

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_HISTORY_TURNS = 8
MAX_USER_MESSAGES = 25

SYSTEM_PROMPT = """\
You are the GridCast assistant, embedded in GridCast — a forecasting demo
about Dutch national electricity load, built by Ismail Arslan as a portfolio
project for freelance data-science work in the energy sector. The interface
is Dutch; answer in the language of the user's latest message (Dutch in,
Dutch out; English in, English out).

WHAT THE PROJECT IS
- Hourly national load for the Netherlands, forecast 1 to 168 hours ahead.
- Scored by a rolling-origin backtest over 2018-2020: quarterly refits, a
  forecast issued every six hours, each made only from data that existed at
  that moment. No random train/test split exists anywhere in the repository.
- Three models, all defensible: seasonal naive (same hour last week), a
  seasonal Fourier ridge on log load (calendar only), and LightGBM gradient
  boosting (direct multi-horizon, horizon as a feature). No deep learning,
  no Prophet, on purpose: neither earns its place on 41,640 hourly points.
- The app trains nothing at runtime; it reads committed parquet artifacts.
- Weather is deliberately NOT a feature: using realised temperature in a
  backtest reports an accuracy nobody reproduces in operation, because real
  forecasting would have to use a weather forecast with its own error.
- Sources: Open Power System Data (load, CC-BY 4.0, originally ENTSO-E
  Transparency, series ends 30 September 2020), Open-Meteo (temperature,
  exploration only), the Dutch grid operators' public capacity map
  (congestion snapshot). The euro figures in the value panel are
  illustrative; the cost of error is settled with the business.

HOW TO ANSWER
- Ground every number in the DIGEST block below. If something is not in it,
  say you cannot see that in the data rather than inventing it.
- The demo's honest findings are its selling point. Do not soften them: the
  80% interval is too narrow, the model under-forecasts the top decile, it
  loses to seasonal naive after the 2020 regime break, and the August 2020
  heatwave miss is the price of leaving weather out. Explain them as
  deliberate, pinned by the selftest.
- Be concise: under 150 words unless asked for depth. Sober tone, no
  exclamation marks, no em dashes. Numbers with units (MW, MWh, %, EUR).
- Small arithmetic on digest numbers is fine; show it briefly.
- Stay on topic: this demo, forecasting, energy. Politely steer anything
  else back.
"""


def get_api_key():
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY")


def build_context(meta: dict, lines: list) -> str:
    """The digest the model is grounded in: meta.json plus computed findings."""
    out = ["DIGEST", "",
           "meta.json (scores are over the full backtest, all horizons):",
           json.dumps({k: meta[k] for k in ("test_start", "overall",
                                            "coverage80", "folds")
                       if k in meta}, default=str)]
    out.append("")
    out.extend(lines)
    return "\n".join(out)


SUGGESTED = [
    "Waarom dekt het 80%-interval maar 76%?",
    "Wat ging er mis in maart en augustus 2020?",
    "Wat zou je hiervan meenemen naar een netbeheerder?",
]


def _post_with_retry(api_key, payload):
    """POST to Groq, retrying on free-tier 429s (the response says how long
    to wait) and transient 5xx errors before giving up with a clear message."""
    for attempt in range(3):
        resp = requests.post(GROQ_URL, json=payload, stream=True, timeout=60,
                             headers={"Authorization": f"Bearer {api_key}"})
        if resp.status_code == 200:
            return resp
        status, detail = resp.status_code, resp.text[:200]
        if attempt < 2 and status == 429:
            m = re.search(r"try again in ([0-9.]+)s", resp.text)
            try:
                wait = float(resp.headers.get("retry-after") or
                             (m.group(1) if m else 3.0))
            except ValueError:
                wait = 3.0
            resp.close()
            time.sleep(min(wait + 0.4, 9.0))
            continue
        if attempt < 2 and status >= 500:
            resp.close()
            time.sleep(1.5)
            continue
        resp.close()
        if status == 429:
            raise RuntimeError("de gratis Groq-laag zit aan zijn limiet en "
                               "bleef bezet na retries; wacht een halve "
                               "minuut en probeer opnieuw.")
        raise RuntimeError(f"Groq API {status}: {detail}")
    raise RuntimeError("Groq API unavailable after retries.")


def stream_reply(api_key, context, history):
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + context}]
        + history[-MAX_HISTORY_TURNS:]
    )
    payload = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 700,
        "stream": True,
    }
    with _post_with_retry(api_key, payload) as resp:
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            delta = json.loads(data)["choices"][0].get("delta", {})
            chunk = delta.get("content")
            if chunk:
                yield chunk
