import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from matplotlib.axes import Axes
from PySide6.QtWidgets import QApplication, QComboBox

from app.ui.main_window import AnalysisPage


class _MainWindowStub:
    def __init__(self, dataframe):
        self.current_dataset = dataframe


def _make_page(dataframe):
    app = QApplication.instance() or QApplication([])
    page = AnalysisPage(_MainWindowStub(dataframe))
    return app, page


def test_histogram_uses_loaded_dataframe_values(monkeypatch):
    dataframe = pd.DataFrame({"measurement": [2, 2, 4, 8]})
    app, page = _make_page(dataframe)
    captured = {}

    def capture_hist(self, values, *args, **kwargs):
        captured["values"] = list(values)

    monkeypatch.setattr(Axes, "hist", capture_hist)
    page._visualization_dataframe = dataframe
    page._build_visualization_panel()
    page.chart_type.setCurrentIndex(page.chart_type.findData("histogram"))

    assert captured["values"] == dataframe["measurement"].tolist()
    page.close()
    app.processEvents()


def test_categorical_chart_handles_many_categories_and_actual_counts():
    dataframe = pd.DataFrame({"group": [f"group-{i % 30}" for i in range(90)]})
    app, page = _make_page(dataframe)
    page._visualization_dataframe = dataframe
    page._build_visualization_panel()
    page.chart_type.setCurrentIndex(page.chart_type.findData("bar"))

    axis = page.chart_figure.axes[0]
    assert len(axis.patches) == page.MAX_CATEGORIES
    assert sum(patch.get_height() for patch in axis.patches) == 60
    page.close()
    app.processEvents()


def test_incompatible_columns_show_clear_empty_state():
    dataframe = pd.DataFrame({"constant": [1, 1, 1], "label": ["only", "only", "only"]})
    app, page = _make_page(dataframe)
    page._visualization_dataframe = dataframe
    page._visualization_correlations = pd.DataFrame()
    page._visualization_missing_values = [
        {"column": "constant", "missing_count": 0}
    ]
    page._build_visualization_panel()

    assert page.chart_column.count() == 0
    assert "Tidak ada kolom numerik" in page.chart_message.text()

    page.chart_type.setCurrentIndex(page.chart_type.findData("correlation"))
    assert "sedikitnya dua kolom numerik" in page.chart_message.text()

    page.chart_type.setCurrentIndex(page.chart_type.findData("missing"))
    assert "Tidak ada nilai kosong" in page.chart_message.text()
    assert not page.chart_canvas.isVisible()
    page.close()
    app.processEvents()
