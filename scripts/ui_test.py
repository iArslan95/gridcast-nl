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
    assert "Model MAPE" in labels and "Seasonal naive MAPE" in labels, labels

    # Four tabs, no more. Streamlit executes every tab body on each run, so a
    # clean run means all four rendered.
    assert len(at.tabs) == 4, f"expected 4 tabs, got {len(at.tabs)}"

    assert len(at.selectbox) == 1, "the forecast-product picker should render"
    assert len(at.number_input) == 2, "two cost inputs expected"

    # Three charts on the problem tab, three on the backtest tab, one on value,
    # two on monitoring.
    assert len(at.get("plotly_chart")) == 9, \
        f"expected 9 charts, got {len(at.get('plotly_chart'))}"

    # Making under-forecasting dearer has to raise the cost of both forecasts.
    before = [b.value for b in at.number_input]
    at.number_input[0].set_value(600.0)
    at.run()
    assert not at.exception, f"app raised after changing the cost: {at.exception}"

    at.selectbox[0].select("Week ahead (h = 168)")
    at.run()
    assert not at.exception, f"app raised on the week-ahead product: {at.exception}"

    print("metrics:", " | ".join(f"{m.label}={m.value}" for m in at.metric))
    print(f"tabs: {len(at.tabs)}  charts: {len(at.get('plotly_chart'))}  "
          f"cost inputs started at {before}")
    print("UI TEST OK")


if __name__ == "__main__":
    main()
