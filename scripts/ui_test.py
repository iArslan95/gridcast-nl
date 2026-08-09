"""Headless UI test via streamlit.testing: walk every section of the app and
drive the controls that change what is computed.

Navigation is a sidebar radio, so unlike tabs only the selected section runs.
That is the point of the sidebar, and it means this test has to visit each one
rather than trusting a single clean run.

    python scripts/ui_test.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

SECTIES = ["Overzicht", "Patronen in de vraag", "Voorspellen per segment",
           "Regio en congestie", "Waarde en besluit", "Monitoring"]
# Charts expected per section. A section that silently loses a chart is a
# regression the eye would miss.
CHARTS = {"Overzicht": 0, "Patronen in de vraag": 6,
          "Voorspellen per segment": 4, "Regio en congestie": 2,
          "Waarde en besluit": 2, "Monitoring": 2}


def run(at: AppTest, sectie: str) -> AppTest:
    at.sidebar.radio[0].set_value(sectie)
    at.run()
    assert not at.exception, f"'{sectie}' raised: {at.exception}"
    n = len(at.get("plotly_chart"))
    assert n == CHARTS[sectie], f"'{sectie}': expected {CHARTS[sectie]} charts, got {n}"
    return at


def main():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception, f"app raised on first run: {at.exception}"
    assert len(at.sidebar.radio) == 1, "sidebar navigation should render"
    assert at.sidebar.radio[0].options == SECTIES, at.sidebar.radio[0].options

    labels = [m.label for m in at.metric]
    assert "MAPE model" in labels and "MAPE seizoensnaief" in labels, labels

    for sectie in SECTIES:
        run(at, sectie)

    # The capacity map has to survive a different operator and direction.
    run(at, "Regio en congestie")
    at.selectbox[0].select("Stedin")
    at.run()
    assert not at.exception, f"map raised for a single operator: {at.exception}"
    at.radio[0].set_value("opwek")
    at.run()
    assert not at.exception, f"map raised for feed-in: {at.exception}"
    stedin = [m.value for m in at.metric]
    assert stedin, "operator filter should still produce metrics"

    # The decision panel has to survive a longer lead time, a higher limit and a
    # dearer error.
    run(at, "Waarde en besluit")
    at.selectbox[0].select(168)
    at.run()
    assert not at.exception, f"decision panel raised on a 7-day lead: {at.exception}"
    at.slider[0].set_value(int(at.slider[0].value) + 400)
    at.run()
    assert not at.exception, f"decision panel raised on a higher limit: {at.exception}"
    at.number_input[0].set_value(600.0)
    at.run()
    assert not at.exception, f"decision panel raised on a new price: {at.exception}"

    print("sections:", len(SECTIES), "| charts:",
          " ".join(f"{s.split()[0]}={CHARTS[s]}" for s in SECTIES))
    print("metrics on Waarde en besluit:",
          " | ".join(f"{m.label}={m.value}" for m in at.metric))
    print("UI TEST OK")


if __name__ == "__main__":
    main()
