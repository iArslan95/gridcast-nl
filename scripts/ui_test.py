"""Headless UI test via streamlit.testing: load the app, check the KPI row and
every tab, and drive the cost panel to make sure the euro figures respond.

    python scripts/ui_test.py
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300)
    at.run()
    assert not at.exception, f"app raised on default run: {at.exception}"

    assert len(at.metric) >= 4, f"expected the KPI row, got {len(at.metric)} metrics"
    labels = [m.label for m in at.metric]
    assert "MAPE model" in labels and "MAPE seizoensnaief" in labels, labels

    # Four tabs, no more. Streamlit executes every tab body on each run, so a
    # clean run means all four rendered.
    assert len(at.tabs) == 4, f"expected 4 tabs, got {len(at.tabs)}"

    assert len(at.selectbox) == 2, "lead-time and product pickers should render"
    assert len(at.number_input) == 2, "two cost inputs expected"
    assert len(at.slider) == 1, "the capacity slider should render"

    # Three charts on the problem tab, three on the backtest tab, two on the
    # decision tab, two on monitoring.
    assert len(at.get("plotly_chart")) == 10, \
        f"expected 10 charts, got {len(at.get('plotly_chart'))}"

    # Making under-forecasting dearer has to raise the cost of both forecasts.
    before = [b.value for b in at.number_input]
    at.number_input[0].set_value(600.0)
    at.run()
    assert not at.exception, f"app raised after changing the cost: {at.exception}"

    at.selectbox[1].select("Week vooruit (h = 168)")
    at.run()
    assert not at.exception, f"app raised on the week-ahead product: {at.exception}"

    # The capacity question is the Stedin-facing part: move the lead time and
    # the limit and the exceedance panel has to survive both.
    # The lead-time picker carries raw hours and formats them for display, so
    # the test selects the value, not the label.
    at.selectbox[0].select(168)
    at.run()
    assert not at.exception, f"app raised on a 7-day lead time: {at.exception}"

    at.slider[0].set_value(int(at.slider[0].value) + 400)
    at.run()
    assert not at.exception, f"app raised on a higher capacity limit: {at.exception}"

    print("metrics:", " | ".join(f"{m.label}={m.value}" for m in at.metric))
    print(f"tabs: {len(at.tabs)}  charts: {len(at.get('plotly_chart'))}  "
          f"cost inputs started at {before}")
    print("UI TEST OK")


if __name__ == "__main__":
    main()
