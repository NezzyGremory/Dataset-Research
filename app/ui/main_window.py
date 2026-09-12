from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QPixmap, QDesktopServices
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QMessageBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QProgressBar,
    QHeaderView,
)

from app.analyzer.loader import DatasetLoader
from app.analyzer.profiler import DatasetProfiler
from app.analyzer.statistics import DatasetStatistics
from app.analyzer.missing_values import MissingValueAnalyzer
from app.analyzer.duplicates import DuplicateAnalyzer
from app.analyzer.outliers import OutlierAnalyzer
from app.analyzer.correlations import CorrelationAnalyzer
from app.analyzer.fingerprint import DatasetFingerprint
from app.ml.task_detector import MLTaskDetector
from app.ml.method_recommender import MethodRecommender
from app.research.intelligence import ResearchIntelligenceEngine


def _resource_path(relative_path: str | Path) -> Path:
    """Return a resource path that works in development and PyInstaller builds."""
    relative = Path(relative_path)

    if getattr(sys, "frozen", False):
        # PyInstaller --onefile extracts bundled data under _MEIPASS.
        return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)) / "app" / "ui" / relative

    return Path(__file__).resolve().parent / relative


class StatCard(QFrame):

    def __init__(
        self,
        title,
        value="—",
        description="",
    ):
        super().__init__()

        self.setObjectName(
            "statCard"
        )

        self.setMinimumWidth(
            105
        )

        layout = QVBoxLayout(
            self
        )

        layout.setContentsMargins(
            14,
            11,
            14,
            11,
        )

        layout.setSpacing(
            1
        )

        title_label = QLabel(
            title
        )

        title_label.setObjectName(
            "statTitle"
        )

        self.value_label = QLabel(
            str(value)
        )

        self.value_label.setObjectName(
            "statValue"
        )

        description_label = QLabel(
            description
        )

        description_label.setObjectName(
            "statDescription"
        )

        layout.addWidget(
            title_label
        )

        layout.addWidget(
            self.value_label
        )

        layout.addWidget(
            description_label
        )

    def set_value(
        self,
        value,
    ):

        self.value_label.setText(
            str(value)
        )

class QuickActionCard(QFrame):

    def __init__(
        self,
        icon,
        title,
        description,
        callback,
        enabled=True,
    ):
        super().__init__()

        self.setObjectName(
            "quickActionCard"
        )

        self.setMinimumHeight(
            132
        )

        layout = QVBoxLayout(
            self
        )

        layout.setContentsMargins(
            16,
            14,
            16,
            14,
        )

        layout.setSpacing(
            5
        )

        # -------------------------------------------------
        # TOP
        # -------------------------------------------------

        top_layout = QHBoxLayout()

        top_layout.setSpacing(
            10
        )

        icon_label = QLabel(
            icon
        )

        icon_label.setObjectName(
            "quickActionIcon"
        )

        icon_label.setFixedSize(
            32,
            32
        )

        icon_label.setAlignment(
            Qt.AlignCenter
        )

        top_layout.addWidget(
            icon_label
        )

        title_label = QLabel(
            title
        )

        title_label.setObjectName(
            "quickActionTitle"
        )

        top_layout.addWidget(
            title_label
        )

        top_layout.addStretch()

        layout.addLayout(
            top_layout
        )

        # -------------------------------------------------
        # DESCRIPTION
        # -------------------------------------------------

        description_label = QLabel(
            description
        )

        description_label.setObjectName(
            "quickActionDescription"
        )

        description_label.setWordWrap(
            True
        )

        layout.addWidget(
            description_label
        )

        layout.addStretch()

        # -------------------------------------------------
        # BUTTON
        # -------------------------------------------------

        action_button = QPushButton(
            "Open  →"
        )

        action_button.setObjectName(
            "quickActionButton"
        )

        action_button.setCursor(
            Qt.PointingHandCursor
        )

        action_button.setEnabled(
            enabled
        )

        if callback is not None:

            action_button.clicked.connect(
                callback
            )

        layout.addWidget(
            action_button
        )

class PipelineStage(QFrame):

    def __init__(
        self,
        number,
        title,
        status="WAITING",
    ):
        super().__init__()

        self.setObjectName(
            "pipelineStageCard"
        )

        self.setMinimumHeight(
            76
        )

        layout = QVBoxLayout(
            self
        )

        layout.setContentsMargins(
            11,
            9,
            11,
            9,
        )

        layout.setSpacing(
            2
        )

        # -------------------------------------------------
        # NUMBER
        # -------------------------------------------------

        number_label = QLabel(
            f"{number:02d}"
        )

        number_label.setObjectName(
            "pipelineNumber"
        )

        layout.addWidget(
            number_label
        )

        # -------------------------------------------------
        # TITLE
        # -------------------------------------------------

        title_label = QLabel(
            title
        )

        title_label.setObjectName(
            "pipelineTitle"
        )

        title_label.setWordWrap(
            True
        )

        layout.addWidget(
            title_label
        )

        # -------------------------------------------------
        # STATUS
        # -------------------------------------------------

        self.status_label = QLabel(
            status
        )

        self.status_label.setObjectName(
            "pipelineStatus"
        )

        layout.addWidget(
            self.status_label
        )

    def set_status(
        self,
        status,
        state="waiting",
    ):

        self.status_label.setText(
            status
        )

        self.setProperty(
            "state",
            state
        )

        self.status_label.setProperty(
            "state",
            state
        )

        style = self.style()

        style.unpolish(
            self
        )

        style.polish(
            self
        )

        self.update()

class MiniLineChart(QFrame):
    """Compact line chart used by the dashboard research snapshot."""

    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("dashboardChartCard")
        self._title = title
        self._subtitle = subtitle
        self._labels = []
        self._values = []
        self.setMinimumHeight(170)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_data(self, labels, values):
        self._labels = [str(x) for x in labels]
        self._values = [float(x) for x in values]
        self.update()

    def clear_data(self):
        self._labels = []
        self._values = []
        self.update()

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont
        from PySide6.QtCore import QRectF, QPointF

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#FFFFFF"))

        painter.setPen(QColor("#17304E"))
        title_font = QFont(self.font())
        title_font.setBold(True)
        title_font.setPointSize(10)
        painter.setFont(title_font)
        painter.drawText(16, 22, self._title)

        painter.setPen(QColor("#7C8BA0"))
        sub_font = QFont(self.font())
        sub_font.setPointSize(8)
        painter.setFont(sub_font)
        painter.drawText(16, 37, self._subtitle)

        chart = QRectF(18, 50, max(40, self.width() - 34), max(70, self.height() - 72))
        painter.setPen(QPen(QColor("#E8EEF6"), 1))
        for frac in (0.0, 0.5, 1.0):
            y = chart.top() + chart.height() * frac
            painter.drawLine(QPointF(chart.left(), y), QPointF(chart.right(), y))

        if not self._values:
            painter.setPen(QColor("#A0ADBD"))
            painter.drawText(chart, Qt.AlignCenter, "Run Academic Research to populate this chart")
            painter.end()
            return

        vmax = max(self._values) or 1.0
        n = len(self._values)
        points = []
        for i, value in enumerate(self._values):
            x = chart.left() if n == 1 else chart.left() + (chart.width() * i / (n - 1))
            y = chart.bottom() - (value / vmax) * chart.height()
            points.append(QPointF(x, y))

        pen = QPen(QColor("#4F8CFF"), 2)
        painter.setPen(pen)
        for a, b in zip(points, points[1:]):
            painter.drawLine(a, b)

        painter.setBrush(QBrush(QColor("#4F8CFF")))
        painter.setPen(Qt.NoPen)
        for point in points:
            painter.drawEllipse(point, 3.5, 3.5)

        painter.setPen(QColor("#7C8BA0"))
        painter.setFont(sub_font)
        for i, label in enumerate(self._labels):
            if n == 1:
                x = chart.left()
            else:
                x = chart.left() + chart.width() * i / (n - 1)
            text = label
            rect = QRectF(x - 28, chart.bottom() + 3, 56, 15)
            painter.drawText(rect, Qt.AlignHCenter | Qt.AlignTop, text)

        painter.end()


class MiniBarChart(QFrame):
    """Compact vertical bar chart used by the dashboard research snapshot."""

    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("dashboardChartCard")
        self._title = title
        self._subtitle = subtitle
        self._labels = []
        self._values = []
        self.setMinimumHeight(170)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_data(self, labels, values):
        self._labels = [str(x) for x in labels]
        self._values = [float(x) for x in values]
        self.update()

    def clear_data(self):
        self._labels = []
        self._values = []
        self.update()

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QFont
        from PySide6.QtCore import QRectF

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#FFFFFF"))

        painter.setPen(QColor("#17304E"))
        title_font = QFont(self.font())
        title_font.setBold(True)
        title_font.setPointSize(10)
        painter.setFont(title_font)
        painter.drawText(16, 22, self._title)

        painter.setPen(QColor("#7C8BA0"))
        sub_font = QFont(self.font())
        sub_font.setPointSize(8)
        painter.setFont(sub_font)
        painter.drawText(16, 37, self._subtitle)

        chart = QRectF(18, 50, max(40, self.width() - 34), max(70, self.height() - 72))
        painter.setPen(QPen(QColor("#E8EEF6"), 1))
        painter.drawLine(chart.left(), chart.bottom(), chart.right(), chart.bottom())

        if not self._values:
            painter.setPen(QColor("#A0ADBD"))
            painter.drawText(chart, Qt.AlignCenter, "No research data yet")
            painter.end()
            return

        vmax = max(self._values) or 1.0
        count = len(self._values)
        slot = chart.width() / max(1, count)
        bar_width = max(12.0, min(40.0, slot * 0.58))

        painter.setBrush(QBrush(QColor("#77A8FF")))
        painter.setPen(Qt.NoPen)
        painter.setFont(sub_font)

        for i, (label, value) in enumerate(zip(self._labels, self._values)):
            bar_h = (value / vmax) * (chart.height() - 12)
            x = chart.left() + slot * i + (slot - bar_width) / 2
            y = chart.bottom() - bar_h
            painter.drawRoundedRect(QRectF(x, y, bar_width, bar_h), 4, 4)

            painter.setPen(QColor("#718096"))
            label_rect = QRectF(x - 12, chart.bottom() + 3, bar_width + 24, 24)
            painter.drawText(label_rect, Qt.AlignHCenter | Qt.AlignTop, label[:10])
            painter.setPen(Qt.NoPen)

        painter.end()


class DashboardPage(QWidget):

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._build_ui()
        self.update_research_result({})

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setObjectName("dashboardScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(28, 22, 28, 28)
        layout.setSpacing(16)

        # HERO --------------------------------------------------
        hero = QFrame()
        hero.setObjectName("heroCard")
        hero.setMinimumHeight(215)

        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(24, 20, 24, 20)
        hero_layout.setSpacing(20)

        # --------------------------------------------------
        # HERO BRAND BLOCK
        # --------------------------------------------------
        hero_brand = QHBoxLayout()
        hero_brand.setSpacing(16)

        hero_logo = QLabel()
        hero_logo.setObjectName("heroLogo")
        hero_logo.setFixedSize(132, 132)
        hero_logo.setAlignment(Qt.AlignCenter)

        logo_path = _resource_path("assets/logo.jpg")
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            if not pixmap.isNull():
                hero_logo.setPixmap(
                    pixmap.scaled(
                        132,
                        132,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )
        else:
            hero_logo.setText("DR")

        hero_text = QVBoxLayout()
        hero_text.setSpacing(7)

        hero_title = QLabel("Dataset Research")
        hero_title.setObjectName("heroTitle")

        hero_description = QLabel(
            "Analyze your dataset, discover ML methods, and find\n"
            "relevant academic research in one workspace."
        )
        hero_description.setObjectName("heroDescription")
        hero_description.setWordWrap(True)

        start_button = QPushButton("+  Upload Dataset")
        start_button.setObjectName("heroButton")
        start_button.setFixedHeight(38)
        start_button.setMaximumWidth(170)
        start_button.setCursor(Qt.PointingHandCursor)
        start_button.clicked.connect(self.main_window.open_upload_page)

        hero_text.addWidget(hero_title)
        hero_text.addWidget(hero_description)
        hero_text.addWidget(start_button)
        hero_text.addStretch()

        hero_brand.addWidget(hero_logo)
        hero_brand.addLayout(hero_text, 1)
        hero_layout.addLayout(hero_brand, 1)

        visual = QFrame()
        visual.setObjectName("heroVisual")
        visual.setFixedWidth(270)
        visual_layout = QVBoxLayout(visual)
        visual_layout.setContentsMargins(16, 13, 16, 13)
        visual_layout.setSpacing(4)

        visual_title = QLabel("RESEARCH WORKFLOW")
        visual_title.setObjectName("heroVisualTitle")
        visual_layout.addWidget(visual_title)

        for number, title in [
            ("01", "Dataset"),
            ("02", "Analysis"),
            ("03", "ML Intelligence"),
            ("04", "Academic Research"),
        ]:
            row = QHBoxLayout()
            row.setSpacing(9)
            number_label = QLabel(number)
            number_label.setObjectName("heroWorkflowNumber")
            number_label.setFixedWidth(22)
            title_label = QLabel(title)
            title_label.setObjectName("heroWorkflowTitle")
            row.addWidget(number_label)
            row.addWidget(title_label)
            row.addStretch()
            visual_layout.addLayout(row)

        visual_layout.addStretch()
        hero_layout.addWidget(visual)
        layout.addWidget(hero)

        # QUICK ACTIONS -----------------------------------------
        quick_header = QHBoxLayout()
        quick_title = QLabel("Quick Actions")
        quick_title.setObjectName("sectionTitle")
        quick_description = QLabel("Start your research workflow")
        quick_description.setObjectName("sectionDescription")
        quick_header.addWidget(quick_title)
        quick_header.addSpacing(8)
        quick_header.addWidget(quick_description)
        quick_header.addStretch()
        layout.addLayout(quick_header)

        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(12)
        actions_layout.addWidget(QuickActionCard(
            "↑", "Upload Dataset",
            "Import a dataset and start a new project.",
            self.main_window.open_upload_page,
        ))
        actions_layout.addWidget(QuickActionCard(
            "◇", "Analyze Dataset",
            "Explore statistics, missing values and outliers.",
            self.main_window.open_analysis_page,
        ))
        actions_layout.addWidget(QuickActionCard(
            "✦", "ML Intelligence",
            "Discover suitable machine learning methods.",
            self.main_window.open_ml_page,
        ))
        actions_layout.addWidget(QuickActionCard(
            "◎", "Academic Research",
            "Find related papers and research opportunities.",
            self.main_window.open_research_page,
        ))
        layout.addLayout(actions_layout)

        # CURRENT DATASET ---------------------------------------
        dataset_header = QHBoxLayout()
        dataset_title = QLabel("Current Dataset")
        dataset_title.setObjectName("sectionTitle")
        self.dataset_status_badge = QLabel("NO DATASET")
        self.dataset_status_badge.setObjectName("datasetStatusBadge")
        dataset_header.addWidget(dataset_title)
        dataset_header.addStretch()
        dataset_header.addWidget(self.dataset_status_badge)
        layout.addLayout(dataset_header)

        dataset_card = QFrame()
        dataset_card.setObjectName("currentDatasetCard")
        dataset_card.setMinimumHeight(92)
        dataset_layout = QHBoxLayout(dataset_card)
        dataset_layout.setContentsMargins(18, 14, 18, 14)
        dataset_layout.setSpacing(16)

        info = QVBoxLayout()
        info.setSpacing(3)
        self.dataset_name_label = QLabel("No dataset loaded")
        self.dataset_name_label.setObjectName("datasetName")
        self.dataset_description_label = QLabel(
            "Upload a dataset to begin your research workflow."
        )
        self.dataset_description_label.setObjectName("datasetDescription")
        self.dataset_description_label.setWordWrap(True)
        info.addWidget(self.dataset_name_label)
        info.addWidget(self.dataset_description_label)
        info.addStretch()
        dataset_layout.addLayout(info, 1)

        stats = QHBoxLayout()
        stats.setSpacing(8)
        self.rows_card = StatCard("ROWS", "—", "Records")
        self.columns_card = StatCard("COLUMNS", "—", "Features")
        stats.addWidget(self.rows_card)
        stats.addWidget(self.columns_card)
        dataset_layout.addLayout(stats)
        layout.addWidget(dataset_card)

        # RESEARCH SNAPSHOT -------------------------------------
        snapshot_header = QHBoxLayout()
        snapshot_title = QLabel("Research Snapshot")
        snapshot_title.setObjectName("sectionTitle")
        snapshot_desc = QLabel("Visual summary of your academic research")
        snapshot_desc.setObjectName("sectionDescription")
        snapshot_header.addWidget(snapshot_title)
        snapshot_header.addSpacing(8)
        snapshot_header.addWidget(snapshot_desc)
        snapshot_header.addStretch()
        layout.addLayout(snapshot_header)

        chart_row = QHBoxLayout()
        chart_row.setSpacing(12)
        self.publication_chart = MiniLineChart(
            "Publication Trend",
            "Papers by publication year",
        )
        self.source_chart = MiniBarChart(
            "Paper Sources",
            "Distribution of academic sources",
        )
        self.metric_chart = MiniBarChart(
            "Research Metrics",
            "Relative magnitude of key outputs",
        )
        chart_row.addWidget(self.publication_chart, 1)
        chart_row.addWidget(self.source_chart, 1)
        chart_row.addWidget(self.metric_chart, 1)
        layout.addLayout(chart_row)

        # PIPELINE ----------------------------------------------
        pipeline_header = QHBoxLayout()
        pipeline_title = QLabel("Research Pipeline")
        pipeline_title.setObjectName("sectionTitle")
        pipeline_description = QLabel("Track your research progress")
        pipeline_description.setObjectName("sectionDescription")
        pipeline_header.addWidget(pipeline_title)
        pipeline_header.addSpacing(8)
        pipeline_header.addWidget(pipeline_description)
        pipeline_header.addStretch()
        layout.addLayout(pipeline_header)

        pipeline_card = QFrame()
        pipeline_card.setObjectName("pipelineCard")
        pipeline_layout = QHBoxLayout(pipeline_card)
        pipeline_layout.setContentsMargins(10, 10, 10, 10)
        pipeline_layout.setSpacing(4)

        pipeline_data = [
            ("Dataset", "Upload"),
            ("Analysis", "Waiting"),
            ("ML Intelligence", "Waiting"),
            ("Academic Research", "Waiting"),
            ("Landscape", "V2"),
            ("Research Gap", "V2"),
            ("Report", "V2"),
        ]
        self.pipeline_stages = []
        for index, (title, status) in enumerate(pipeline_data):
            stage = PipelineStage(index + 1, title, status.upper())
            self.pipeline_stages.append(stage)
            pipeline_layout.addWidget(stage, 1)
            if index < len(pipeline_data) - 1:
                connector = QLabel("›")
                connector.setObjectName("pipelineConnector")
                connector.setAlignment(Qt.AlignCenter)
                pipeline_layout.addWidget(connector)
        layout.addWidget(pipeline_card)

        layout.addStretch()
        scroll.setWidget(container)
        outer.addWidget(scroll)

    def update_dataset(self, dataframe, filename):
        if dataframe is None:
            return

        self.dataset_name_label.setText(filename)
        self.dataset_description_label.setText(
            "Dataset loaded successfully. Run analysis to generate dataset intelligence."
        )
        self.dataset_status_badge.setText("● DATASET LOADED")
        self.dataset_status_badge.setProperty("state", "success")
        style = self.dataset_status_badge.style()
        style.unpolish(self.dataset_status_badge)
        style.polish(self.dataset_status_badge)
        self.dataset_status_badge.update()

        self.rows_card.set_value(f"{len(dataframe):,}")
        self.columns_card.set_value(f"{len(dataframe.columns):,}")

        if len(self.pipeline_stages) >= 1:
            self.pipeline_stages[0].set_status("READY", "success")
        if len(self.pipeline_stages) >= 2:
            self.pipeline_stages[1].set_status("NEXT", "active")
        for stage in self.pipeline_stages[2:4]:
            stage.set_status("WAITING", "waiting")
        for stage in self.pipeline_stages[4:]:
            stage.set_status("READY", "future")

        self.update_research_result({})

    def update_research_result(self, result):
        result = result if isinstance(result, dict) else {}
        papers = result.get("papers", [])
        if not isinstance(papers, list):
            papers = []

        # Publication years.
        year_counts = {}
        for paper in papers:
            if not isinstance(paper, dict):
                continue
            year = paper.get("year")
            try:
                year = int(year)
            except (TypeError, ValueError):
                continue
            if 1900 <= year <= 2100:
                year_counts[year] = year_counts.get(year, 0) + 1
        years = sorted(year_counts)
        if years:
            self.publication_chart.set_data(
                years,
                [year_counts[y] for y in years],
            )
        else:
            self.publication_chart.clear_data()

        # Sources: prefer explicit search summary, otherwise paper source.
        source_counts = {}
        search_info = result.get("search", {})
        if isinstance(search_info, dict):
            sources = search_info.get("sources", {})
            if isinstance(sources, dict):
                for source, count in sources.items():
                    try:
                        source_counts[str(source)] = int(count)
                    except (TypeError, ValueError):
                        pass
        if not source_counts:
            for paper in papers:
                if not isinstance(paper, dict):
                    continue
                source = paper.get("source") or "Unknown"
                source = str(source)
                source_counts[source] = source_counts.get(source, 0) + 1
        top_sources = sorted(source_counts.items(), key=lambda item: item[1], reverse=True)[:5]
        if top_sources:
            self.source_chart.set_data(
                [name for name, _ in top_sources],
                [value for _, value in top_sources],
            )
        else:
            self.source_chart.clear_data()

        # Key metrics as a compact bar chart.
        summary = result.get("summary", {}) if isinstance(result.get("summary", {}), dict) else {}
        gaps = result.get("gaps", {}) if isinstance(result.get("gaps", {}), dict) else {}
        gap_summary = gaps.get("summary", {}) if isinstance(gaps.get("summary", {}), dict) else {}
        keyword_result = result.get("keywords", {}) if isinstance(result.get("keywords", {}), dict) else {}
        keywords = keyword_result.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [keywords]
        keyword_count = len(keywords) if isinstance(keywords, list) else 0
        paper_count = len(papers)
        gap_count = gap_summary.get("gap_count", 0) or 0
        relevance = summary.get("top_relevance_score") or 0
        metrics = [paper_count, keyword_count, gap_count, float(relevance)]
        labels = ["Papers", "Keywords", "Gaps", "Top Score"]
        self.metric_chart.set_data(labels, metrics if any(metrics) else [])

        # Pipeline status.
        if papers:
            if len(self.pipeline_stages) >= 4:
                self.pipeline_stages[3].set_status("READY", "success")
            if len(self.pipeline_stages) >= 5:
                self.pipeline_stages[4].set_status("READY", "success")
            if len(self.pipeline_stages) >= 6:
                self.pipeline_stages[5].set_status("READY", "success")
            if len(self.pipeline_stages) >= 7:
                self.pipeline_stages[6].set_status("READY", "success")

    def reset_dataset(self):
        self.dataset_name_label.setText("No dataset loaded")
        self.dataset_description_label.setText(
            "Upload a dataset to begin your research workflow."
        )
        self.dataset_status_badge.setText("NO DATASET")
        self.dataset_status_badge.setProperty("state", "waiting")
        style = self.dataset_status_badge.style()
        style.unpolish(self.dataset_status_badge)
        style.polish(self.dataset_status_badge)
        self.dataset_status_badge.update()
        self.rows_card.set_value("—")
        self.columns_card.set_value("—")
        for index, stage in enumerate(self.pipeline_stages):
            if index == 0:
                stage.set_status("UPLOAD", "active")
            elif index < 4:
                stage.set_status("WAITING", "waiting")
            else:
                stage.set_status("V2", "future")
        self.update_research_result({})

class UploadPage(QWidget):
    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window

        self.build_ui()

    def build_ui(self):

        layout = QVBoxLayout(self)

        layout.setContentsMargins(
            35, 30, 35, 30
        )

        layout.setSpacing(20)

        # =================================================
        # HEADER
        # =================================================

        title = QLabel(
            "Upload Dataset"
        )

        title.setObjectName(
            "pageTitle"
        )

        subtitle = QLabel(
            "Upload a CSV dataset to begin "
            "your academic research analysis."
        )

        subtitle.setObjectName(
            "pageSubtitle"
        )

        subtitle.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(subtitle)

        # =================================================
        # UPLOAD CARD
        # =================================================

        upload_card = QFrame()

        upload_card.setObjectName(
            "uploadCard"
        )

        upload_layout = QVBoxLayout(
            upload_card
        )

        upload_layout.setContentsMargins(
            40, 40, 40, 40
        )

        upload_layout.setSpacing(15)

        icon_label = QLabel(
            "CSV DATASET"
        )

        icon_label.setObjectName(
            "uploadIcon"
        )

        icon_label.setAlignment(
            Qt.AlignCenter
        )

        upload_layout.addWidget(
            icon_label
        )

        upload_title = QLabel(
            "Choose your dataset"
        )

        upload_title.setObjectName(
            "cardTitle"
        )

        upload_title.setAlignment(
            Qt.AlignCenter
        )

        upload_layout.addWidget(
            upload_title
        )

        upload_description = QLabel(
            "Supported format: CSV\n"
            "The original dataset will never be modified."
        )

        upload_description.setObjectName(
            "cardDescription"
        )

        upload_description.setAlignment(
            Qt.AlignCenter
        )

        upload_layout.addWidget(
            upload_description
        )

        button_layout = QHBoxLayout()

        button_layout.addStretch()

        self.upload_button = QPushButton(
            "Select CSV Dataset"
        )

        self.upload_button.setObjectName(
            "primaryButton"
        )

        self.upload_button.setMinimumHeight(
            45
        )

        self.upload_button.setMinimumWidth(
            200
        )

        self.upload_button.clicked.connect(
            self.select_dataset
        )

        button_layout.addWidget(
            self.upload_button
        )

        button_layout.addStretch()

        upload_layout.addLayout(
            button_layout
        )

        layout.addWidget(
            upload_card
        )

        # =================================================
        # FILE INFORMATION
        # =================================================

        info_card = QFrame()

        info_card.setObjectName(
            "contentCard"
        )

        info_layout = QVBoxLayout(
            info_card
        )

        info_layout.setContentsMargins(
            22, 20, 22, 20
        )

        info_title = QLabel(
            "Dataset Information"
        )

        info_title.setObjectName(
            "sectionTitle"
        )

        self.info_label = QLabel(
            "No dataset selected."
        )

        self.info_label.setObjectName(
            "cardDescription"
        )

        self.info_label.setWordWrap(
            True
        )

        info_layout.addWidget(
            info_title
        )

        info_layout.addWidget(
            self.info_label
        )

        layout.addWidget(
            info_card
        )

        layout.addStretch()

    # =====================================================
    # SELECT DATASET
    # =====================================================

    def select_dataset(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset",
            "",
            "CSV Files (*.csv)"
        )

        if not file_path:
            return

        try:

            dataframe = (
                self.main_window.loader.load_csv(
                    file_path
                )
            )

            self.main_window.current_dataset = (
                dataframe
            )

            self.main_window.current_file_path = (
                file_path
            )

            filename = Path(
                file_path
            ).name

            file_size = (
                Path(file_path).stat().st_size
                / 1024
            )

            self.info_label.setText(
                f"<b>File:</b> {filename}<br>"
                f"<b>Size:</b> {file_size:.2f} KB<br>"
                f"<b>Rows:</b> {len(dataframe):,}<br>"
                f"<b>Columns:</b> "
                f"{len(dataframe.columns):,}"
            )

            self.main_window.dashboard.update_dataset(
                dataframe,
                filename
            )

            self.main_window.update_dataset_state()

            self.main_window.open_analysis_page()

        except Exception as error:

            QMessageBox.critical(
                self,
                "Upload Error",
                "Failed to load dataset.\n\n"
                f"{error}"
            )

class AnalysisPage(QWidget):

    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window

        self.build_ui()

    def build_ui(self):

        root_layout = QVBoxLayout(self)

        root_layout.setContentsMargins(
            35, 30, 35, 30
        )

        root_layout.setSpacing(20)

        # =================================================
        # HEADER
        # =================================================

        header_layout = QHBoxLayout()

        title_layout = QVBoxLayout()

        title = QLabel(
            "Dataset Analysis"
        )

        title.setObjectName(
            "pageTitle"
        )

        subtitle = QLabel(
            "Understand the structure, quality, "
            "statistics, and relationships within your dataset."
        )

        subtitle.setObjectName(
            "pageSubtitle"
        )

        subtitle.setWordWrap(True)

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        header_layout.addLayout(
            title_layout
        )

        header_layout.addStretch()

        self.analyze_button = QPushButton(
            "Run Analysis"
        )

        self.analyze_button.setObjectName(
            "primaryButton"
        )

        self.analyze_button.setMinimumHeight(
            42
        )

        self.analyze_button.clicked.connect(
            self.run_analysis
        )

        header_layout.addWidget(
            self.analyze_button
        )

        root_layout.addLayout(
            header_layout
        )

        # =================================================
        # DATASET STATUS
        # =================================================

        self.dataset_label = QLabel(
            "No dataset loaded."
        )

        self.dataset_label.setObjectName(
            "contentCard"
        )

        root_layout.addWidget(
            self.dataset_label
        )

        # =================================================
        # RESULT AREA
        # =================================================

        scroll = QScrollArea()

        scroll.setWidgetResizable(
            True
        )

        scroll.setFrameShape(
            QFrame.NoFrame
        )

        container = QWidget()

        self.result_layout = QVBoxLayout(
            container
        )

        self.result_layout.setSpacing(
            15
        )

        scroll.setWidget(
            container
        )

        root_layout.addWidget(
            scroll
        )

        self.show_empty_state()

    # =====================================================
    # EMPTY STATE
    # =====================================================

    def show_empty_state(self):

        self.clear_results()

        label = QLabel(
            "No analysis results yet.\n\n"
            "Upload a dataset and click "
            "\"Run Analysis\"."
        )

        label.setObjectName(
            "emptyState"
        )

        label.setWordWrap(True)

        self.result_layout.addWidget(
            label
        )

        self.result_layout.addStretch()

    # =====================================================
    # CLEAR
    # =====================================================

    def clear_results(self):

        while self.result_layout.count():

            item = self.result_layout.takeAt(0)

            widget = item.widget()

            if widget:
                widget.deleteLater()

    # =====================================================
    # RUN ANALYSIS
    # =====================================================

    def run_analysis(self):

        dataframe = (
            self.main_window.current_dataset
        )

        if dataframe is None:

            self.show_empty_state()

            return

        self.analyze_button.setEnabled(
            False
        )

        self.analyze_button.setText(
            "Analyzing..."
        )

        try:

            result = (
                self.main_window.run_dataset_analysis()
            )

            self.display_results(
                result
            )

        except Exception as error:

            self.clear_results()

            error_label = QLabel(
                "Analysis failed.\n\n"
                f"{error}"
            )

            error_label.setObjectName(
                "errorState"
            )

            error_label.setWordWrap(
                True
            )

            self.result_layout.addWidget(
                error_label
            )

        finally:

            self.analyze_button.setEnabled(
                True
            )

            self.analyze_button.setText(
                "Run Analysis"
            )

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    def display_results(self, result):

        self.clear_results()

        profile = result.get(
            "profile"
        )

        statistics = result.get(
            "statistics"
        )

        missing_values = result.get(
            "missing_values"
        )

        duplicates = result.get(
            "duplicates"
        )

        outliers = result.get(
            "outliers"
        )

        correlations = result.get(
            "correlations"
        )

        fingerprint = result.get(
            "fingerprint"
        )

        # =================================================
        # OVERVIEW
        # =================================================

        self.add_section(
            "Dataset Overview",
            str(profile)
        )

        # =================================================
        # STATISTICS
        # =================================================

        self.add_section(
            "Column Statistics",
            self.format_collection(
                statistics
            )
        )

        # =================================================
        # MISSING
        # =================================================

        self.add_section(
            "Missing Values",
            self.format_collection(
                missing_values
            )
        )

        # =================================================
        # DUPLICATES
        # =================================================

        self.add_section(
            "Duplicates",
            str(duplicates)
        )

        # =================================================
        # OUTLIERS
        # =================================================

        self.add_section(
            "Outliers",
            self.format_collection(
                outliers
            )
        )

        # =================================================
        # CORRELATIONS
        # =================================================

        self.add_section(
            "Correlation Analysis",
            str(correlations)
        )

        # =================================================
        # FINGERPRINT
        # =================================================

        if fingerprint:

            fingerprint_text = (
                f"SHA-256:\n"
                f"{fingerprint.get('fingerprint', '—')}\n\n"
                f"Representation:\n"
                f"{fingerprint.get('representation', '—')}"
            )

            self.add_section(
                "Dataset Fingerprint",
                fingerprint_text
            )

        self.result_layout.addStretch()

    # =====================================================
    # SECTION
    # =====================================================

    def add_section(
        self,
        title,
        content
    ):

        card = QFrame()

        card.setObjectName(
            "contentCard"
        )

        layout = QVBoxLayout(
            card
        )

        layout.setContentsMargins(
            22, 20, 22, 20
        )

        layout.setSpacing(
            10
        )

        title_label = QLabel(
            title
        )

        title_label.setObjectName(
            "sectionTitle"
        )

        content_label = QLabel(
            str(content)
        )

        content_label.setObjectName(
            "analysisContent"
        )

        content_label.setWordWrap(
            True
        )

        layout.addWidget(
            title_label
        )

        layout.addWidget(
            content_label
        )

        self.result_layout.addWidget(
            card
        )

    # =====================================================
    # FORMAT COLLECTION
    # =====================================================

    @staticmethod
    def format_collection(
        collection
    ):

        if collection is None:
            return "No data."

        if isinstance(
            collection,
            (list, tuple)
        ):

            if not collection:
                return "No data."

            return "\n\n".join(
                str(item)
                for item in collection
            )

        return str(collection)

    # =====================================================
    # UPDATE DATASET
    # =====================================================

    def update_dataset(
        self,
        dataframe,
        filename
    ):

        self.dataset_label.setText(
            f"<b>{filename}</b> — "
            f"{len(dataframe):,} rows × "
            f"{len(dataframe.columns):,} columns"
        )

        self.show_empty_state()

class InfoCard(QFrame):

    def __init__(
        self,
        title: str,
        value: str = "-",
        subtitle: str = "",
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        self.setObjectName("contentCard")

        self.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Fixed,
        )

        layout = QVBoxLayout(self)

        layout.setContentsMargins(
            20,
            18,
            20,
            18,
        )

        layout.setSpacing(6)

        title_label = QLabel(
            title.upper()
        )

        title_label.setObjectName(
            "cardLabel"
        )

        self.value_label = QLabel(
            value
        )

        self.value_label.setObjectName(
            "cardValue"
        )

        self.value_label.setWordWrap(
            True
        )

        self.subtitle_label = QLabel(
            subtitle
        )

        self.subtitle_label.setObjectName(
            "cardDescription"
        )

        self.subtitle_label.setWordWrap(
            True
        )

        layout.addWidget(
            title_label
        )

        layout.addWidget(
            self.value_label
        )

        layout.addWidget(
            self.subtitle_label
        )

    def set_value(
        self,
        value: str,
    ):
        self.value_label.setText(
            value
        )

    def set_subtitle(
        self,
        text: str,
    ):
        self.subtitle_label.setText(
            text
        )

class TaskCard(QFrame):

    def __init__(
        self,
        task: dict[str, Any],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        self.setObjectName(
            "contentCard"
        )

        layout = QVBoxLayout(
            self
        )

        layout.setContentsMargins(
            20,
            18,
            20,
            18,
        )

        layout.setSpacing(
            10
        )

        header = QHBoxLayout()

        label = (
            task.get("label")
            or task.get("task")
            or "Unknown Task"
        )

        title = QLabel(
            str(label)
        )

        title.setObjectName(
            "sectionTitle"
        )

        score = task.get(
            "confidence",
            task.get("score", 0),
        )

        confidence = QLabel(
            f"{score}% confidence"
        )

        confidence.setObjectName(
            "scoreBadge"
        )

        header.addWidget(
            title
        )

        header.addStretch()

        header.addWidget(
            confidence
        )

        layout.addLayout(
            header
        )

        target = task.get(
            "target"
        )

        if target:

            target_label = QLabel(
                f"Target: {target}"
            )

            target_label.setObjectName(
                "cardDescription"
            )

            layout.addWidget(
                target_label
            )

        status = task.get(
            "status",
            "ESTIMATION",
        )

        status_label = QLabel(
            f"Status: {status}"
        )

        status_label.setObjectName(
            "statusLabel"
        )

        layout.addWidget(
            status_label
        )

        reasons = task.get(
            "reasons",
            [],
        )

        if reasons:

            reasons_title = QLabel(
                "Why this task?"
            )

            reasons_title.setObjectName(
                "cardLabel"
            )

            layout.addWidget(
                reasons_title
            )

            for reason in reasons:

                reason_label = QLabel(
                    f"• {reason}"
                )

                reason_label.setObjectName(
                    "cardDescription"
                )

                reason_label.setWordWrap(
                    True
                )

                layout.addWidget(
                    reason_label
                )

class MethodCard(QFrame):

    def __init__(
        self,
        method: dict[str, Any],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        self.method = method

        self.setObjectName(
            "contentCard"
        )

        layout = QVBoxLayout(
            self
        )

        layout.setContentsMargins(
            20,
            18,
            20,
            18,
        )

        layout.setSpacing(
            8
        )

        # --------------------------------------------------
        # HEADER
        # --------------------------------------------------

        header = QHBoxLayout()

        title = QLabel(
            str(
                method.get(
                    "method",
                    "Unknown Method",
                )
            )
        )

        title.setObjectName(
            "sectionTitle"
        )

        score = method.get(
            "score",
            0,
        )

        score_label = QLabel(
            f"{score}% match"
        )

        score_label.setObjectName(
            "scoreBadge"
        )

        header.addWidget(
            title
        )

        header.addStretch()

        header.addWidget(
            score_label
        )

        layout.addLayout(
            header
        )

        # --------------------------------------------------
        # CATEGORY
        # --------------------------------------------------

        category = QLabel(
            f"Category: "
            f"{method.get('category', '-')}"
        )

        category.setObjectName(
            "cardDescription"
        )

        layout.addWidget(
            category
        )

        # --------------------------------------------------
        # DESCRIPTION
        # --------------------------------------------------

        description = QLabel(
            str(
                method.get(
                    "description",
                    "",
                )
            )
        )

        description.setObjectName(
            "cardDescription"
        )

        description.setWordWrap(
            True
        )

        layout.addWidget(
            description
        )

        # --------------------------------------------------
        # REASONS
        # --------------------------------------------------

        reasons = method.get(
            "reasons",
            [],
        )

        if reasons:

            reasons_title = QLabel(
                "Why recommended?"
            )

            reasons_title.setObjectName(
                "cardLabel"
            )

            layout.addWidget(
                reasons_title
            )

            for reason in reasons[:5]:

                reason_label = QLabel(
                    f"• {reason}"
                )

                reason_label.setObjectName(
                    "cardDescription"
                )

                reason_label.setWordWrap(
                    True
                )

                layout.addWidget(
                    reason_label
                )

        # --------------------------------------------------
        # QUICK INFO
        # --------------------------------------------------

        info_layout = QHBoxLayout()

        interpretability = method.get(
            "interpretability",
            "-",
        )

        scaling = method.get(
            "scaling_required",
            False,
        )

        nonlinear = method.get(
            "nonlinear",
            False,
        )

        info_layout.addWidget(
            QLabel(
                f"Interpretability: "
                f"{interpretability}"
            )
        )

        info_layout.addWidget(
            QLabel(
                f"Scaling: "
                f"{'Required' if scaling else 'Not required'}"
            )
        )

        info_layout.addWidget(
            QLabel(
                f"Nonlinear: "
                f"{'Yes' if nonlinear else 'No'}"
            )
        )

        info_layout.addStretch()

        layout.addLayout(
            info_layout
        )

class MethodDetailCard(QFrame):

    def __init__(
        self,
        method: dict[str, Any],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        self.setObjectName(
            "contentCard"
        )

        layout = QVBoxLayout(
            self
        )

        layout.setContentsMargins(
            24,
            22,
            24,
            22,
        )

        layout.setSpacing(
            12
        )

        # --------------------------------------------------
        # TITLE
        # --------------------------------------------------

        title = QLabel(
            method.get(
                "method",
                "Method Details",
            )
        )

        title.setObjectName(
            "sectionTitle"
        )

        layout.addWidget(
            title
        )

        # --------------------------------------------------
        # DESCRIPTION
        # --------------------------------------------------

        description_title = QLabel(
            "Description"
        )

        description_title.setObjectName(
            "cardLabel"
        )

        layout.addWidget(
            description_title
        )

        description = QLabel(
            method.get(
                "description",
                "-",
            )
        )

        description.setObjectName(
            "cardDescription"
        )

        description.setWordWrap(
            True
        )

        layout.addWidget(
            description
        )

        # --------------------------------------------------
        # STRENGTHS
        # --------------------------------------------------

        strengths_title = QLabel(
            "Strengths"
        )

        strengths_title.setObjectName(
            "cardLabel"
        )

        layout.addWidget(
            strengths_title
        )

        for item in method.get(
            "strengths",
            [],
        ):

            label = QLabel(
                f"• {item}"
            )

            label.setObjectName(
                "cardDescription"
            )

            label.setWordWrap(
                True
            )

            layout.addWidget(
                label
            )

        # --------------------------------------------------
        # LIMITATIONS
        # --------------------------------------------------

        limitations_title = QLabel(
            "Limitations"
        )

        limitations_title.setObjectName(
            "cardLabel"
        )

        layout.addWidget(
            limitations_title
        )

        for item in method.get(
            "limitations",
            [],
        ):

            label = QLabel(
                f"• {item}"
            )

            label.setObjectName(
                "cardDescription"
            )

            label.setWordWrap(
                True
            )

            layout.addWidget(
                label
            )

        # --------------------------------------------------
        # PREPROCESSING
        # --------------------------------------------------

        preprocessing_title = QLabel(
            "Preprocessing"
        )

        preprocessing_title.setObjectName(
            "cardLabel"
        )

        layout.addWidget(
            preprocessing_title
        )

        for item in method.get(
            "preprocessing",
            [],
        ):

            label = QLabel(
                f"• {item}"
            )

            label.setObjectName(
                "cardDescription"
            )

            label.setWordWrap(
                True
            )

            layout.addWidget(
                label
            )

        # --------------------------------------------------
        # META
        # --------------------------------------------------

        meta = QLabel(
            "Interpretability: "
            f"{method.get('interpretability', '-')}"
            "\n"
            "Feature Scaling: "
            f"{'Required' if method.get('scaling_required') else 'Not required'}"
            "\n"
            "Nonlinear: "
            f"{'Yes' if method.get('nonlinear') else 'No'}"
            "\n"
            "Small Data: "
            f"{'Suitable' if method.get('small_data') else 'Limited'}"
            "\n"
            "Large Data: "
            f"{'Suitable' if method.get('large_data') else 'Limited'}"
        )

        meta.setObjectName(
            "cardDescription"
        )

        meta.setWordWrap(
            True
        )

        layout.addWidget(
            meta
        )

class MLIntelligencePage(QWidget):

    """
    Halaman ML Intelligence.

    Pipeline:

        Dataset
            ↓
        Fingerprint
            ↓
        ML Task Detection
            ↓
        Method Recommendation

    Semua hasil tetap bersifat estimasi/rekomendasi.
    """

    def __init__(
        self,
        main_window,
    ):
        super().__init__()

        self.main_window = main_window

        self.detector = (
            MLTaskDetector()
        )

        self.recommender = (
            MethodRecommender()
        )

        self.last_task_result = {}
        self.last_method_result = {}

        self.build_ui()

    # ======================================================
    # UI
    # ======================================================

    def build_ui(self):

        root_layout = QVBoxLayout(
            self
        )

        root_layout.setContentsMargins(
            30,
            25,
            30,
            25,
        )

        root_layout.setSpacing(
            18
        )

        # --------------------------------------------------
        # HEADER
        # --------------------------------------------------

        header = QHBoxLayout()

        title_layout = QVBoxLayout()

        title_layout.setSpacing(
            4
        )

        title = QLabel(
            "ML Intelligence"
        )

        title.setObjectName(
            "pageTitle"
        )

        description = QLabel(
            "Deteksi task machine learning "
            "dan rekomendasi metode berdasarkan "
            "struktur serta karakteristik dataset."
        )

        description.setObjectName(
            "pageDescription"
        )

        description.setWordWrap(
            True
        )

        title_layout.addWidget(
            title
        )

        title_layout.addWidget(
            description
        )

        header.addLayout(
            title_layout
        )

        header.addStretch()

        self.refresh_button = QPushButton(
            "Analyze ML"
        )

        self.refresh_button.setObjectName(
            "primaryButton"
        )

        self.refresh_button.clicked.connect(
            self.run_detection
        )

        header.addWidget(
            self.refresh_button
        )

        root_layout.addLayout(
            header
        )

        # --------------------------------------------------
        # SCROLL
        # --------------------------------------------------

        scroll = QScrollArea()

        scroll.setWidgetResizable(
            True
        )

        scroll.setFrameShape(
            QFrame.NoFrame
        )

        self.content = QWidget()

        self.content_layout = QVBoxLayout(
            self.content
        )

        self.content_layout.setContentsMargins(
            0,
            5,
            10,
            20,
        )

        self.content_layout.setSpacing(
            16
        )

        scroll.setWidget(
            self.content
        )

        root_layout.addWidget(
            scroll
        )

        self.show_empty_state()

    # ======================================================
    # EMPTY STATE
    # ======================================================

    def show_empty_state(self):

        self.clear_content()

        card = QFrame()

        card.setObjectName(
            "contentCard"
        )

        layout = QVBoxLayout(
            card
        )

        layout.setContentsMargins(
            30,
            40,
            30,
            40,
        )

        layout.setSpacing(
            10
        )

        title = QLabel(
            "ML Intelligence belum tersedia"
        )

        title.setObjectName(
            "sectionTitle"
        )

        title.setAlignment(
            Qt.AlignCenter
        )

        description = QLabel(
            "Upload dan analisis dataset terlebih dahulu "
            "untuk mendapatkan estimasi task machine learning "
            "dan rekomendasi metode."
        )

        description.setObjectName(
            "cardDescription"
        )

        description.setAlignment(
            Qt.AlignCenter
        )

        description.setWordWrap(
            True
        )

        layout.addWidget(
            title
        )

        layout.addWidget(
            description
        )

        self.content_layout.addWidget(
            card
        )

        self.content_layout.addStretch()

    # ======================================================
    # DETECTION
    # ======================================================

    def run_detection(self):

        analysis_result = getattr(
            self.main_window,
            "analysis_result",
            {},
        )

        if not analysis_result:

            self.show_empty_state()

            return

        fingerprint = analysis_result.get(
            "fingerprint"
        )

        if not fingerprint:

            self.show_empty_state()

            return

        try:

            # ------------------------------------------------
            # ML TASK DETECTION
            # ------------------------------------------------

            result = self.detector.detect(
                fingerprint
            )

            self.last_task_result = result

            # ------------------------------------------------
            # DATASET METADATA
            # ------------------------------------------------

            dataset_info = self.extract_dataset_info(
                analysis_result,
                fingerprint,
            )

            # ------------------------------------------------
            # ADD DATASET INFORMATION
            # ------------------------------------------------

            result["dataset"] = dataset_info

            # ------------------------------------------------
            # METHOD RECOMMENDATION
            # ------------------------------------------------

            method_result = (
                self.recommender.recommend(
                    result
                )
            )

            self.last_method_result = (
                method_result
            )

            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            self.display_result(
                result,
                method_result,
            )

        except Exception as exc:

            self.show_error(
                "Gagal melakukan ML Intelligence:\n"
                f"{exc}"
            )

    # ======================================================
    # DATASET INFORMATION
    # ======================================================

    def extract_dataset_info(
        self,
        analysis_result: dict[str, Any],
        fingerprint: dict[str, Any],
    ) -> dict[str, Any]:

        dataset = {}

        dataframe = getattr(
            self.main_window,
            "current_dataset",
            None,
        )

        # --------------------------------------------------
        # PRIMARY SOURCE: DATAFRAME
        # --------------------------------------------------

        if dataframe is not None:

            try:

                dataset["rows"] = int(
                    dataframe.shape[0]
                )

                dataset["columns"] = int(
                    dataframe.shape[1]
                )

                numeric_features = len(
                    dataframe.select_dtypes(
                        include="number"
                    ).columns
                )

                dataset[
                    "numeric_features"
                ] = numeric_features

            except Exception:
                pass

        # --------------------------------------------------
        # FALLBACK: FINGERPRINT
        # --------------------------------------------------

        representation = fingerprint.get(
            "representation",
            fingerprint,
        )

        characteristics = (
            representation.get(
                "characteristics",
                {},
            )
        )

        if not dataset.get("rows"):

            dataset["rows"] = int(
                representation.get(
                    "rows",
                    fingerprint.get(
                        "rows",
                        0,
                    ),
                )
                or 0
            )

        if not dataset.get("columns"):

            dataset["columns"] = int(
                representation.get(
                    "columns",
                    fingerprint.get(
                        "columns",
                        0,
                    ),
                )
                or 0
            )

        if not dataset.get(
            "numeric_features"
        ):

            dataset[
                "numeric_features"
            ] = int(
                characteristics.get(
                    "numeric_count",
                    0,
                )
                or 0
            )

        return dataset

    # ======================================================
    # DISPLAY RESULT
    # ======================================================

    def display_result(
        self,
        result: dict[str, Any],
        method_result: dict[str, Any],
    ):

        self.clear_content()

        status = result.get(
            "status",
            "ESTIMATION",
        )

        primary = result.get(
            "primary_task"
        )

        tasks = result.get(
            "tasks",
            [],
        )

        dataset = result.get(
            "dataset",
            {},
        )

        recommendations = method_result.get(
            "recommendations",
            [],
        )

        # ==================================================
        # DATASET SUMMARY
        # ==================================================

        dataset_title = QLabel(
            "Dataset Intelligence"
        )

        dataset_title.setObjectName(
            "sectionTitle"
        )

        self.content_layout.addWidget(
            dataset_title
        )

        dataset_cards = QHBoxLayout()

        dataset_cards.setSpacing(
            14
        )

        dataset_cards.addWidget(
            InfoCard(
                "Rows",
                str(
                    dataset.get(
                        "rows",
                        0,
                    )
                ),
                "Jumlah observasi dalam dataset.",
            )
        )

        dataset_cards.addWidget(
            InfoCard(
                "Columns",
                str(
                    dataset.get(
                        "columns",
                        0,
                    )
                ),
                "Jumlah kolom yang dianalisis.",
            )
        )

        dataset_cards.addWidget(
            InfoCard(
                "Numeric Features",
                str(
                    dataset.get(
                        "numeric_features",
                        0,
                    )
                ),
                "Jumlah fitur numerik.",
            )
        )

        self.content_layout.addLayout(
            dataset_cards
        )

        # ==================================================
        # PRIMARY TASK
        # ==================================================

        if primary:

            primary_label = primary.get(
                "label",
                primary.get(
                    "task",
                    "-",
                ),
            )

            confidence = primary.get(
                "confidence",
                primary.get(
                    "score",
                    0,
                ),
            )

            target = primary.get(
                "target"
            )

            cards_layout = QHBoxLayout()

            cards_layout.setSpacing(
                14
            )

            cards_layout.addWidget(
                InfoCard(
                    "Primary ML Task",
                    str(
                        primary_label
                    ),
                    f"Confidence: {confidence}%",
                )
            )

            cards_layout.addWidget(
                InfoCard(
                    "Target",
                    str(target)
                    if target
                    else "None",
                    "Target candidate yang digunakan detector.",
                )
            )

            cards_layout.addWidget(
                InfoCard(
                    "Confidence",
                    f"{confidence}%",
                    "Nilai ini merupakan estimasi heuristik.",
                )
            )

            self.content_layout.addLayout(
                cards_layout
            )

        # ==================================================
        # STATUS
        # ==================================================

        status_card = QFrame()

        status_card.setObjectName(
            "contentCard"
        )

        status_layout = QVBoxLayout(
            status_card
        )

        status_layout.setContentsMargins(
            20,
            18,
            20,
            18,
        )

        status_layout.setSpacing(
            8
        )

        status_title = QLabel(
            "Detection Status"
        )

        status_title.setObjectName(
            "cardLabel"
        )

        status_value = QLabel(
            str(status)
        )

        status_value.setObjectName(
            "statusLabel"
        )

        message = QLabel(
            result.get(
                "message",
                "Task merupakan estimasi berdasarkan dataset.",
            )
        )

        message.setObjectName(
            "cardDescription"
        )

        message.setWordWrap(
            True
        )

        status_layout.addWidget(
            status_title
        )

        status_layout.addWidget(
            status_value
        )

        status_layout.addWidget(
            message
        )

        self.content_layout.addWidget(
            status_card
        )

        # ==================================================
        # PRIMARY TASK ANALYSIS
        # ==================================================

        if primary:

            primary_section = QLabel(
                "Primary Task Analysis"
            )

            primary_section.setObjectName(
                "sectionTitle"
            )

            self.content_layout.addWidget(
                primary_section
            )

            self.content_layout.addWidget(
                TaskCard(
                    primary
                )
            )

        # ==================================================
        # METHOD RECOMMENDATION
        # ==================================================

        recommendation_title = QLabel(
            "Recommended ML Methods"
        )

        recommendation_title.setObjectName(
            "sectionTitle"
        )

        self.content_layout.addWidget(
            recommendation_title
        )

        recommendation_description = QLabel(
            "Ranking berikut merupakan recommendation score "
            "berdasarkan ML task dan karakteristik dataset. "
            "Score ini bukan accuracy atau hasil training model."
        )

        recommendation_description.setObjectName(
            "cardDescription"
        )

        recommendation_description.setWordWrap(
            True
        )

        self.content_layout.addWidget(
            recommendation_description
        )

        # --------------------------------------------------
        # METHOD CARDS
        # --------------------------------------------------

        if recommendations:

            for index, method in enumerate(
                recommendations,
                start=1,
            ):

                rank_label = QLabel(
                    f"#{index}"
                )

                rank_label.setObjectName(
                    "cardLabel"
                )

                self.content_layout.addWidget(
                    rank_label
                )

                self.content_layout.addWidget(
                    MethodCard(
                        method
                    )
                )

            # ------------------------------------------------
            # TOP METHOD DETAIL
            # ------------------------------------------------

            best_method = recommendations[0]

            detail_title = QLabel(
                "Top Recommended Method"
            )

            detail_title.setObjectName(
                "sectionTitle"
            )

            self.content_layout.addWidget(
                detail_title
            )

            self.content_layout.addWidget(
                MethodDetailCard(
                    best_method
                )
            )

        else:

            empty_methods = QLabel(
                "Tidak ada metode yang dapat direkomendasikan "
                "untuk task yang terdeteksi."
            )

            empty_methods.setObjectName(
                "cardDescription"
            )

            empty_methods.setWordWrap(
                True
            )

            self.content_layout.addWidget(
                empty_methods
            )

        # ==================================================
        # OTHER POSSIBLE TASKS
        # ==================================================

        tasks_section = QLabel(
            f"Possible ML Tasks ({len(tasks)})"
        )

        tasks_section.setObjectName(
            "sectionTitle"
        )

        self.content_layout.addWidget(
            tasks_section
        )

        if tasks:

            for task in tasks:

                self.content_layout.addWidget(
                    TaskCard(
                        task
                    )
                )

        else:

            empty = QLabel(
                "Tidak ada task ML yang berhasil dideteksi."
            )

            empty.setObjectName(
                "cardDescription"
            )

            self.content_layout.addWidget(
                empty
            )

        # ==================================================
        # DISCLAIMER
        # ==================================================

        disclaimer = QFrame()

        disclaimer.setObjectName(
            "contentCard"
        )

        disclaimer_layout = QVBoxLayout(
            disclaimer
        )

        disclaimer_layout.setContentsMargins(
            20,
            18,
            20,
            18,
        )

        disclaimer_title = QLabel(
            "Research Note"
        )

        disclaimer_title.setObjectName(
            "cardLabel"
        )

        disclaimer_text = QLabel(
            "ML task detection dan method recommendation "
            "merupakan estimasi berbasis heuristik serta "
            "knowledge base. Performa aktual metode harus "
            "divalidasi melalui eksperimen, cross-validation, "
            "dan metric evaluasi yang sesuai."
        )

        disclaimer_text.setObjectName(
            "cardDescription"
        )

        disclaimer_text.setWordWrap(
            True
        )

        disclaimer_layout.addWidget(
            disclaimer_title
        )

        disclaimer_layout.addWidget(
            disclaimer_text
        )

        self.content_layout.addWidget(
            disclaimer
        )

        self.content_layout.addStretch()

    # ======================================================
    # ERROR
    # ======================================================

    def show_error(
        self,
        message: str,
    ):

        self.clear_content()

        card = QFrame()

        card.setObjectName(
            "contentCard"
        )

        layout = QVBoxLayout(
            card
        )

        layout.setContentsMargins(
            25,
            25,
            25,
            25,
        )

        layout.setSpacing(
            10
        )

        title = QLabel(
            "ML Intelligence Error"
        )

        title.setObjectName(
            "sectionTitle"
        )

        error = QLabel(
            message
        )

        error.setObjectName(
            "cardDescription"
        )

        error.setWordWrap(
            True
        )

        layout.addWidget(
            title
        )

        layout.addWidget(
            error
        )

        self.content_layout.addWidget(
            card
        )

        self.content_layout.addStretch()

    # ======================================================
    # UPDATE DATASET
    # ======================================================

    def update_dataset(self):

        if getattr(
            self.main_window,
            "analysis_result",
            {},
        ):

            self.run_detection()

        else:

            self.show_empty_state()

    # ======================================================
    # CLEAR CONTENT
    # ======================================================

    def clear_content(self):

        while self.content_layout.count():

            item = (
                self.content_layout.takeAt(0)
            )

            widget = item.widget()

            if widget is not None:

                widget.deleteLater()

            elif item.layout() is not None:

                self.clear_layout(
                    item.layout()
                )

    # ======================================================
    # CLEAR LAYOUT
    # ======================================================

    def clear_layout(
        self,
        layout,
    ):

        while layout.count():

            item = layout.takeAt(0)

            widget = item.widget()

            if widget is not None:

                widget.deleteLater()

            elif item.layout() is not None:

                self.clear_layout(
                    item.layout()
                )

class ResearchPage(QWidget):
    """
    Academic Research workspace.

    Menampilkan:
    - Research Overview
    - Search Queries
    - Ranked Papers
    - Dataset Usage Confidence
    - Research Landscape
    - Research Trend
    - Research Gap
    """

    def __init__(self, main_window):
        super().__init__()

        self.main_window = main_window

        self.engine = ResearchIntelligenceEngine()
        self.task_detector = MLTaskDetector()

        self.research_result: Dict[str, Any] = {}

        self._build_ui()

    # =========================================================
    # UI
    # =========================================================

    def _build_ui(self):

        root = QVBoxLayout(self)

        root.setContentsMargins(
            30,
            25,
            30,
            25,
        )

        root.setSpacing(18)

        # -----------------------------------------------------
        # HEADER
        # -----------------------------------------------------

        header_layout = QHBoxLayout()

        title_layout = QVBoxLayout()

        title = QLabel(
            "Academic Research"
        )

        title.setObjectName(
            "pageTitle"
        )

        subtitle = QLabel(
            "Research intelligence based on your dataset"
        )

        subtitle.setObjectName(
            "pageSubtitle"
        )

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        header_layout.addLayout(
            title_layout
        )

        header_layout.addStretch()

        self.run_button = QPushButton(
            "Run Academic Research"
        )

        self.run_button.setObjectName(
            "primaryButton"
        )

        self.run_button.setMinimumHeight(
            42
        )

        self.run_button.setCursor(
            Qt.PointingHandCursor
        )

        self.run_button.clicked.connect(
            self.run_research
        )

        header_layout.addWidget(
            self.run_button
        )

        root.addLayout(
            header_layout
        )

        # -----------------------------------------------------
        # STATUS
        # -----------------------------------------------------

        self.status_frame = QFrame()

        self.status_frame.setObjectName(
            "researchStatus"
        )

        status_layout = QVBoxLayout(
            self.status_frame
        )

        status_layout.setContentsMargins(
            18,
            14,
            18,
            14,
        )

        self.dataset_label = QLabel(
            "Dataset: No dataset loaded"
        )

        self.dataset_label.setObjectName(
            "statusLabel"
        )

        self.analysis_label = QLabel(
            "Analysis: Not available"
        )

        self.analysis_label.setObjectName(
            "statusLabel"
        )

        status_layout.addWidget(
            self.dataset_label
        )

        status_layout.addWidget(
            self.analysis_label
        )

        root.addWidget(
            self.status_frame
        )

        # -----------------------------------------------------
        # PROGRESS
        # -----------------------------------------------------

        self.progress = QProgressBar()

        self.progress.setRange(
            0,
            0,
        )

        self.progress.setVisible(
            False
        )

        root.addWidget(
            self.progress
        )

        # -----------------------------------------------------
        # TABS
        # -----------------------------------------------------

        self.tabs = QTabWidget()

        self.tabs.setDocumentMode(
            True
        )

        self.overview_tab = (
            self._create_overview_tab()
        )

        self.queries_tab = (
            self._create_queries_tab()
        )

        self.papers_tab = (
            self._create_papers_tab()
        )

        self.landscape_tab = (
            self._create_landscape_tab()
        )

        self.trend_tab = (
            self._create_trend_tab()
        )

        self.gap_tab = (
            self._create_gap_tab()
        )

        self.tabs.addTab(
            self.overview_tab,
            "Overview",
        )

        self.tabs.addTab(
            self.queries_tab,
            "Search Queries",
        )

        self.tabs.addTab(
            self.papers_tab,
            "Ranked Papers",
        )

        self.tabs.addTab(
            self.landscape_tab,
            "Landscape",
        )

        self.tabs.addTab(
            self.trend_tab,
            "Trend",
        )

        self.tabs.addTab(
            self.gap_tab,
            "Research Gap",
        )

        root.addWidget(
            self.tabs,
            1,
        )

        # Initial state
        self.show_empty_state()

    # =========================================================
    # TAB FACTORIES
    # =========================================================

    def _create_overview_tab(self):

        widget = QWidget()

        layout = QVBoxLayout(
            widget
        )

        layout.setContentsMargins(
            5,
            15,
            5,
            5,
        )

        layout.setSpacing(18)

        # -----------------------------------------------------
        # SUMMARY CARDS
        # -----------------------------------------------------

        cards = QHBoxLayout()

        self.papers_card = (
            self._create_metric_card(
                "Papers Found",
                "0",
            )
        )

        self.score_card = (
            self._create_metric_card(
                "Top Relevance",
                "-",
            )
        )

        self.domain_card = (
            self._create_metric_card(
                "Research Domain",
                "-",
            )
        )

        self.gap_card = (
            self._create_metric_card(
                "Potential Gaps",
                "0",
            )
        )

        cards.addWidget(
            self.papers_card
        )

        cards.addWidget(
            self.score_card
        )

        cards.addWidget(
            self.domain_card
        )

        cards.addWidget(
            self.gap_card
        )

        layout.addLayout(
            cards
        )

        # -----------------------------------------------------
        # INTELLIGENCE
        # -----------------------------------------------------

        intelligence_frame = QFrame()

        intelligence_frame.setObjectName(
            "researchCard"
        )

        intelligence_layout = QVBoxLayout(
            intelligence_frame
        )

        intelligence_title = QLabel(
            "Research Intelligence"
        )

        intelligence_title.setObjectName(
            "sectionTitle"
        )

        intelligence_layout.addWidget(
            intelligence_title
        )

        self.keyword_text = QLabel(
            "Keywords: -"
        )

        self.domain_text = QLabel(
            "Domain: -"
        )

        self.task_text = QLabel(
            "ML Task: -"
        )

        self.method_text = QLabel(
            "Dominant Method: -"
        )

        self.direction_text = QLabel(
            "Research Direction: -"
        )

        self.latest_year_text = QLabel(
            "Latest Publication: -"
        )

        for label in (
            self.keyword_text,
            self.domain_text,
            self.task_text,
            self.method_text,
            self.direction_text,
            self.latest_year_text,
        ):
            label.setWordWrap(True)

            intelligence_layout.addWidget(
                label
            )

        layout.addWidget(
            intelligence_frame
        )

        # -----------------------------------------------------
        # TOP PAPER
        # -----------------------------------------------------

        top_frame = QFrame()

        top_frame.setObjectName(
            "researchCard"
        )

        top_layout = QVBoxLayout(
            top_frame
        )

        top_title = QLabel(
            "Top Ranked Paper"
        )

        top_title.setObjectName(
            "sectionTitle"
        )

        self.top_paper_title = QLabel(
            "No research has been run yet."
        )

        self.top_paper_title.setWordWrap(
            True
        )

        self.top_paper_info = QLabel(
            ""
        )

        self.top_paper_info.setWordWrap(
            True
        )

        top_layout.addWidget(
            top_title
        )

        top_layout.addWidget(
            self.top_paper_title
        )

        top_layout.addWidget(
            self.top_paper_info
        )

        layout.addWidget(
            top_frame
        )

        layout.addStretch()

        return widget

    # ---------------------------------------------------------

    def _create_queries_tab(self):

        widget = QWidget()

        layout = QVBoxLayout(
            widget
        )

        title = QLabel(
            "Generated Academic Search Queries"
        )

        title.setObjectName(
            "sectionTitle"
        )

        description = QLabel(
            "Queries generated from the detected dataset "
            "keywords, research domain, and ML task."
        )

        description.setWordWrap(
            True
        )

        self.query_list = QListWidget()

        layout.addWidget(
            title
        )

        layout.addWidget(
            description
        )

        layout.addWidget(
            self.query_list,
            1,
        )

        return widget

    # ---------------------------------------------------------

    def _create_papers_tab(self):

        widget = QWidget()

        layout = QVBoxLayout(
            widget
        )

        title = QLabel(
            "Ranked Academic Papers"
        )

        title.setObjectName(
            "sectionTitle"
        )

        description = QLabel(
            "Ranking uses the configured relevance model. "
            "Dataset Usage Confidence is reported separately "
            "from relevance."
        )

        description.setWordWrap(
            True
        )

        self.paper_table = QTableWidget()

        self.paper_table.setColumnCount(
            7
        )

        self.paper_table.setHorizontalHeaderLabels(
            [
                "#",
                "Paper",
                "Year",
                "Relevance",
                "Dataset",
                "Schema",
                "Usage Confidence",
            ]
        )

        self.paper_table.setEditTriggers(
            QTableWidget.NoEditTriggers
        )

        self.paper_table.setSelectionBehavior(
            QTableWidget.SelectRows
        )

        self.paper_table.setAlternatingRowColors(
            True
        )

        header = (
            self.paper_table.horizontalHeader()
        )

        header.setSectionResizeMode(
            0,
            QHeaderView.ResizeToContents,
        )

        header.setSectionResizeMode(
            1,
            QHeaderView.Stretch,
        )

        header.setSectionResizeMode(
            2,
            QHeaderView.ResizeToContents,
        )

        header.setSectionResizeMode(
            3,
            QHeaderView.ResizeToContents,
        )

        header.setSectionResizeMode(
            4,
            QHeaderView.ResizeToContents,
        )

        header.setSectionResizeMode(
            5,
            QHeaderView.ResizeToContents,
        )

        header.setSectionResizeMode(
            6,
            QHeaderView.ResizeToContents,
        )

        self.paper_table.itemSelectionChanged.connect(
            self._show_selected_paper
        )

        self.paper_detail = QTextEdit()

        self.paper_detail.setReadOnly(
            True
        )

        self.paper_detail.setMaximumHeight(
            190
        )

        layout.addWidget(
            title
        )

        layout.addWidget(
            description
        )

        layout.addWidget(
            self.paper_table,
            1,
        )

        layout.addWidget(
            self.paper_detail
        )

        return widget

    # ---------------------------------------------------------

    def _create_landscape_tab(self):

        widget = QWidget()

        layout = QVBoxLayout(
            widget
        )

        title = QLabel(
            "Research Landscape"
        )

        title.setObjectName(
            "sectionTitle"
        )

        self.landscape_text = QTextEdit()

        self.landscape_text.setReadOnly(
            True
        )

        layout.addWidget(
            title
        )

        layout.addWidget(
            self.landscape_text,
            1,
        )

        return widget

    # ---------------------------------------------------------

    def _create_trend_tab(self):

        widget = QWidget()

        layout = QVBoxLayout(
            widget
        )

        title = QLabel(
            "Research Trend"
        )

        title.setObjectName(
            "sectionTitle"
        )

        self.trend_text = QTextEdit()

        self.trend_text.setReadOnly(
            True
        )

        layout.addWidget(
            title
        )

        layout.addWidget(
            self.trend_text,
            1,
        )

        return widget

    # ---------------------------------------------------------

    def _create_gap_tab(self):

        widget = QWidget()

        layout = QVBoxLayout(
            widget
        )

        title = QLabel(
            "Potential Research Gap"
        )

        title.setObjectName(
            "sectionTitle"
        )

        description = QLabel(
            "These are potential research directions inferred "
            "from the analyzed literature. They are not claims "
            "that a research gap has been definitively proven."
        )

        description.setWordWrap(
            True
        )

        self.gap_text = QTextEdit()

        self.gap_text.setReadOnly(
            True
        )

        layout.addWidget(
            title
        )

        layout.addWidget(
            description
        )

        layout.addWidget(
            self.gap_text,
            1,
        )

        return widget

    # =========================================================
    # METRIC CARD
    # =========================================================

    @staticmethod
    def _create_metric_card(
        title: str,
        value: str,
    ):

        frame = QFrame()

        frame.setObjectName(
            "metricCard"
        )

        frame.setMinimumHeight(
            105
        )

        layout = QVBoxLayout(
            frame
        )

        title_label = QLabel(
            title
        )

        title_label.setObjectName(
            "metricTitle"
        )

        value_label = QLabel(
            value
        )

        value_label.setObjectName(
            "metricValue"
        )

        value_label.setWordWrap(
            True
        )

        layout.addWidget(
            title_label
        )

        layout.addWidget(
            value_label
        )

        frame.value_label = value_label

        return frame

    # =========================================================
    # EMPTY STATE
    # =========================================================

    def show_empty_state(self):

        self.research_result = {}

        self.run_button.setEnabled(
            True
        )

        self.progress.setVisible(
            False
        )

        self.dataset_label.setText(
            "Dataset: No dataset loaded"
        )

        self.analysis_label.setText(
            "Analysis: Not available"
        )

        self._set_metric(
            self.papers_card,
            "0",
        )

        self._set_metric(
            self.score_card,
            "-",
        )

        self._set_metric(
            self.domain_card,
            "-",
        )

        self._set_metric(
            self.gap_card,
            "0",
        )

        self.keyword_text.setText(
            "Keywords: -"
        )

        self.domain_text.setText(
            "Domain: -"
        )

        self.task_text.setText(
            "ML Task: -"
        )

        self.method_text.setText(
            "Dominant Method: -"
        )

        self.direction_text.setText(
            "Research Direction: -"
        )

        self.latest_year_text.setText(
            "Latest Publication: -"
        )

        self.top_paper_title.setText(
            "No research has been run yet."
        )

        self.top_paper_info.setText(
            ""
        )

        self.query_list.clear()

        self.paper_table.setRowCount(
            0
        )

        self.paper_detail.clear()

        self.landscape_text.clear()

        self.trend_text.clear()

        self.gap_text.clear()

        if hasattr(self.main_window, "papers_tool_page"):
            self.main_window.papers_tool_page.show_empty_state()

    # =========================================================
    # RUN RESEARCH
    # =========================================================

    def run_research(self):

        dataframe = (
            self.main_window.current_dataset
        )

        analysis_result = (
            self.main_window.analysis_result
        )

        if dataframe is None:

            QMessageBox.warning(
                self,
                "Dataset Required",
                "Load a dataset before running academic research.",
            )

            return

        if not analysis_result:

            QMessageBox.warning(
                self,
                "Analysis Required",
                "Run Dataset Analysis first before starting Academic Research.",
            )

            return

        fingerprint = (
            analysis_result.get(
                "fingerprint"
            )
        )

        if not fingerprint:

            QMessageBox.warning(
                self,
                "Fingerprint Required",
                "Dataset fingerprint is not available. "
                "Run Dataset Analysis again.",
            )

            return

        self.run_button.setEnabled(
            False
        )

        self.progress.setVisible(
            True
        )

        self.run_button.setText(
            "Researching..."
        )

        try:

            ml_result = (
                self._build_ml_result(
                    dataframe,
                    fingerprint,
                )
            )

            result = (
                self.engine.analyze(
                    dataframe=dataframe,
                    fingerprint=fingerprint,
                    ml_result=ml_result,
                    search_limit=20,
                    max_queries=5,
                )
            )

            if result.get(
                "status"
            ) != "SUCCESS":

                error = result.get(
                    "error",
                    "Unknown research error.",
                )

                QMessageBox.critical(
                    self,
                    "Research Error",
                    str(error),
                )

                return

            self.research_result = result

            self.main_window.dashboard.update_research_result(result)

            self.main_window.papers_tool_page.update_from_result(
                result
            )

            self._update_status(
                dataframe,
                analysis_result,
            )

            self._populate_result(
                result,
                ml_result,
            )

            self.tabs.setCurrentWidget(
                self.overview_tab
            )

        except Exception as exc:

            QMessageBox.critical(
                self,
                "Academic Research Error",
                str(exc),
            )

        finally:

            self.progress.setVisible(
                False
            )

            self.run_button.setEnabled(
                True
            )

            self.run_button.setText(
                "Run Academic Research"
            )

    # =========================================================
    # ML RESULT
    # =========================================================

    def _build_ml_result(
        self,
        dataframe,
        fingerprint,
    ):

        try:

            detected = (
                self.task_detector.detect(
                    fingerprint
                )
            )

        except Exception:

            detected = {
                "status": "ESTIMATION",
                "primary_task": None,
                "tasks": [],
                "task_count": 0,
            }

        result = dict(
            detected
        )

        result["dataset"] = {
            "rows": int(
                dataframe.shape[0]
            ),
            "columns": int(
                dataframe.shape[1]
            ),
            "numeric_features": int(
                dataframe.select_dtypes(
                    include="number"
                ).shape[1]
            ),
        }

        return result

    # =========================================================
    # UPDATE STATUS
    # =========================================================

    def _update_status(
        self,
        dataframe,
        analysis_result,
    ):

        filename = "Dataset"

        file_path = (
            self.main_window.current_file_path
        )

        if file_path:

            try:

                from pathlib import Path

                filename = Path(
                    file_path
                ).name

            except Exception:
                pass

        self.dataset_label.setText(
            f"Dataset: {filename} "
            f"({dataframe.shape[0]:,} rows × "
            f"{dataframe.shape[1]} columns)"
        )

        self.analysis_label.setText(
            "Analysis: Dataset fingerprint available"
        )

    # =========================================================
    # POPULATE RESULT
    # =========================================================

    def _populate_result(
        self,
        result,
        ml_result,
    ):

        summary = result.get(
            "summary",
            {}
        )

        papers = result.get(
            "papers",
            []
        )

        keywords_result = result.get(
            "keywords",
            {}
        )

        domain_result = result.get(
            "domain",
            {}
        )

        queries = result.get(
            "queries",
            []
        )

        # -----------------------------------------------------
        # SUMMARY
        # -----------------------------------------------------

        papers_found = summary.get(
            "papers_found",
            len(papers),
        )

        top_score = summary.get(
            "top_relevance_score"
        )

        gap_count = summary.get(
            "potential_gap_count",
            0,
        )

        domain = self._extract_domain(
            domain_result
        )

        self._set_metric(
            self.papers_card,
            str(papers_found),
        )

        self._set_metric(
            self.score_card,
            (
                f"{top_score:.1f}%"
                if isinstance(
                    top_score,
                    (int, float)
                )
                else "-"
            ),
        )

        self._set_metric(
            self.domain_card,
            domain or "-",
        )

        self._set_metric(
            self.gap_card,
            str(gap_count),
        )

        # -----------------------------------------------------
        # KEYWORDS
        # -----------------------------------------------------

        keywords = self._extract_keywords(
            keywords_result
        )

        if keywords:

            self.keyword_text.setText(
                "Keywords: "
                + ", ".join(
                    str(k)
                    for k in keywords
                )
            )

        else:

            self.keyword_text.setText(
                "Keywords: -"
            )

        # -----------------------------------------------------
        # DOMAIN
        # -----------------------------------------------------

        self.domain_text.setText(
            f"Research Domain: {domain or '-'}"
        )

        # -----------------------------------------------------
        # ML TASK
        # -----------------------------------------------------

        task = ml_result.get(
            "primary_task"
        )

        self.task_text.setText(
            f"ML Task: {task or '-'}"
        )

        # -----------------------------------------------------
        # LANDSCAPE SUMMARY
        # -----------------------------------------------------

        landscape = result.get(
            "landscape",
            {}
        )

        landscape_summary = (
            landscape.get(
                "summary",
                {}
            )
            if isinstance(
                landscape,
                dict
            )
            else {}
        )

        dominant_method = (
            landscape_summary.get(
                "dominant_method"
            )
        )

        latest_year = (
            landscape_summary.get(
                "latest_publication_year"
            )
        )

        self.method_text.setText(
            f"Dominant Method: "
            f"{dominant_method or '-'}"
        )

        self.latest_year_text.setText(
            f"Latest Publication: "
            f"{latest_year or '-'}"
        )

        # -----------------------------------------------------
        # TREND
        # -----------------------------------------------------

        trend = result.get(
            "trend",
            {}
        )

        direction = (
            trend.get(
                "recent_direction",
                "UNKNOWN",
            )
            if isinstance(
                trend,
                dict
            )
            else "UNKNOWN"
        )

        self.direction_text.setText(
            f"Research Direction: {direction}"
        )

        # -----------------------------------------------------
        # TOP PAPER
        # -----------------------------------------------------

        top_paper = (
            papers[0]
            if papers
            else None
        )

        if top_paper:

            title = top_paper.get(
                "title",
                "Untitled",
            )

            score = top_paper.get(
                "relevance_score"
            )

            confidence = top_paper.get(
                "dataset_usage_confidence",
                "UNKNOWN",
            )

            self.top_paper_title.setText(
                title
            )

            self.top_paper_info.setText(
                f"Relevance: "
                f"{self._format_score(score)}"
                f"    •    "
                f"Dataset Usage Confidence: "
                f"{confidence}"
            )

        else:

            self.top_paper_title.setText(
                "No academic papers found."
            )

            self.top_paper_info.setText(
                ""
            )

        # -----------------------------------------------------
        # QUERIES
        # -----------------------------------------------------

        self.query_list.clear()

        for query in queries:

            item = QListWidgetItem(
                str(query)
            )

            self.query_list.addItem(
                item
            )

        if not queries:

            self.query_list.addItem(
                "No search queries generated."
            )

        # -----------------------------------------------------
        # PAPERS
        # -----------------------------------------------------

        self._populate_papers(
            papers
        )

        # -----------------------------------------------------
        # LANDSCAPE
        # -----------------------------------------------------

        self.landscape_text.setPlainText(
            self._format_section(
                landscape
            )
        )

        # -----------------------------------------------------
        # TREND
        # -----------------------------------------------------

        self.trend_text.setPlainText(
            self._format_section(
                trend
            )
        )

        # -----------------------------------------------------
        # GAP
        # -----------------------------------------------------

        gaps = result.get(
            "gaps",
            {}
        )

        self.gap_text.setPlainText(
            self._format_gap(
                gaps
            )
        )

    # =========================================================
    # PAPER TABLE
    # =========================================================

    def _populate_papers(
        self,
        papers,
    ):

        self.paper_table.setRowCount(
            len(papers)
        )

        for row, paper in enumerate(
            papers
        ):

            title = paper.get(
                "title",
                "Untitled",
            )

            year = paper.get(
                "year"
            )

            relevance = paper.get(
                "relevance_score"
            )

            breakdown = paper.get(
                "score_breakdown",
                {}
            )

            dataset_score = (
                breakdown.get(
                    "dataset_name",
                    0,
                )
            )

            schema_score = (
                breakdown.get(
                    "schema",
                    0,
                )
            )

            confidence = paper.get(
                "dataset_usage_confidence",
                "UNKNOWN",
            )

            values = [
                str(row + 1),
                str(title),
                str(year or "-"),
                self._format_score(
                    relevance
                ),
                self._format_score(
                    dataset_score
                ),
                self._format_score(
                    schema_score
                ),
                str(confidence),
            ]

            for column, value in enumerate(
                values
            ):

                item = QTableWidgetItem(
                    value
                )

                item.setToolTip(
                    str(value)
                )

                self.paper_table.setItem(
                    row,
                    column,
                    item
                )

        self.paper_table.resizeRowsToContents()

    # =========================================================
    # PAPER DETAIL
    # =========================================================

    def _show_selected_paper(self):

        rows = (
            self.paper_table.selectionModel()
            .selectedRows()
        )

        if not rows:

            self.paper_detail.clear()

            return

        row = rows[0].row()

        papers = self.research_result.get(
            "papers",
            []
        )

        if row >= len(papers):

            return

        paper = papers[row]

        lines = []

        lines.append(
            f"TITLE\n{paper.get('title', '-')}"
        )

        lines.append(
            f"AUTHORS\n"
            f"{', '.join(paper.get('authors', [])) or '-'}"
        )

        lines.append(
            f"YEAR\n{paper.get('year') or '-'}"
        )

        lines.append(
            f"VENUE\n{paper.get('venue') or '-'}"
        )

        lines.append(
            f"DOI\n{paper.get('doi') or '-'}"
        )

        lines.append(
            f"RELEVANCE SCORE\n"
            f"{self._format_score(paper.get('relevance_score'))}"
        )

        lines.append(
            f"Dataset Usage Confidence\n"
            f"{paper.get('dataset_usage_confidence', '-')}"
        )

        evidence = paper.get(
            "dataset_usage_evidence",
            []
        )

        if evidence:

            lines.append(
                "Evidence\n"
                + "\n".join(
                    f"• {item}"
                    for item in evidence
                )
            )

        limitations = paper.get(
            "dataset_usage_limitations",
            []
        )

        if limitations:

            lines.append(
                "Limitations\n"
                + "\n".join(
                    f"• {item}"
                    for item in limitations
                )
            )

        abstract = paper.get(
            "abstract",
            ""
        )

        if abstract:

            lines.append(
                f"Abstract\n{abstract}"
            )

        self.paper_detail.setPlainText(
            "\n\n".join(lines)
        )

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _set_metric(
        card,
        value,
    ):

        if hasattr(
            card,
            "value_label"
        ):

            card.value_label.setText(
                str(value)
            )

    @staticmethod
    def _format_score(
        value
    ):

        if isinstance(
            value,
            (int, float)
        ):

            return f"{value:.1f}%"

        return "-"

    @staticmethod
    def _extract_domain(
        result
    ):

        if not isinstance(
            result,
            dict
        ):

            return ""

        return (
            result.get(
                "primary_domain"
            )
            or result.get(
                "domain"
            )
            or ""
        )

    @staticmethod
    def _extract_keywords(
        result
    ):

        if not isinstance(
            result,
            dict
        ):

            return []

        keywords = result.get(
            "keywords",
            []
        )

        if isinstance(
            keywords,
            dict
        ):

            keywords = keywords.get(
                "keywords",
                []
            )

        if isinstance(
            keywords,
            str
        ):

            return [keywords]

        if isinstance(
            keywords,
            list
        ):

            return keywords

        return []

    @staticmethod
    def _pretty_key(key):
        text = str(key).replace("_", " ").strip()
        return " ".join(word.capitalize() for word in text.split())

    @staticmethod
    def _format_value(value):
        if value is None or value == "":
            return "—"
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value).strip()

    @staticmethod
    def _format_section(data, indent=0):
        """Render research data with consistent, readable labels."""
        if not data:
            return "No data available."

        prefix = " " * indent
        lines = []

        if isinstance(data, dict):
            for key, value in data.items():
                label = ResearchPage._pretty_key(key)
                if isinstance(value, dict):
                    lines.append(f"{prefix}{label}")
                    lines.extend(ResearchPage._format_section(value, indent + 2).splitlines())
                elif isinstance(value, list):
                    lines.append(f"{prefix}{label}")
                    if value:
                        for item in value:
                            if isinstance(item, dict):
                                nested = ResearchPage._format_section(item, indent + 2)
                                lines.extend(f"{prefix}  • {line.strip()}" for line in nested.splitlines())
                            else:
                                lines.append(f"{prefix}  • {str(item).strip()}")
                    else:
                        lines.append(f"{prefix}  • —")
                else:
                    lines.append(f"{prefix}{label}: {ResearchPage._format_value(value)}")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    lines.extend(ResearchPage._format_section(item, indent).splitlines())
                else:
                    lines.append(f"{prefix}• {str(item).strip()}")
        else:
            lines.append(f"{prefix}{str(data).strip()}")

        return "\n".join(line.rstrip() for line in lines if line.strip())

    @staticmethod
    def _format_gap(gaps):
        if not gaps:
            return "No potential research gaps detected."

        if isinstance(gaps, dict):
            sections = []
            summary = gaps.get("summary")
            if summary:
                sections.append("SUMMARY\n" + ResearchPage._format_section(summary))

            gap_items = gaps.get("gaps") or gaps.get("potential_gaps") or gaps.get("items")
            if isinstance(gap_items, list):
                items = []
                for item in gap_items:
                    if isinstance(item, dict):
                        title = item.get("title") or item.get("gap") or item.get("description") or "Potential research direction"
                        items.append(f"• {str(title).strip()}")
                    else:
                        items.append(f"• {str(item).strip()}")
                sections.append("POTENTIAL RESEARCH DIRECTIONS\n" + "\n".join(items))
            elif not summary:
                sections.append(ResearchPage._format_section(gaps))

            return "\n\n".join(section for section in sections if section.strip())

        return ResearchPage._format_section(gaps)


class PapersToolPage(QWidget):
    """
    Dedicated Research Tool: Papers.

    Uses the already-generated Academic Research result instead of
    running a second academic search. This keeps the tool lightweight
    and synchronized with the Academic Research workspace.
    """

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.papers = []
        self.filtered_papers = []
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(30, 25, 30, 25)
        root.setSpacing(16)

        # -----------------------------------------------------
        # HEADER
        # -----------------------------------------------------

        header = QHBoxLayout()

        title_layout = QVBoxLayout()
        title_layout.setSpacing(4)

        title = QLabel("Papers")
        title.setObjectName("pageTitle")

        subtitle = QLabel(
            "Browse, filter, and inspect papers discovered by Academic Research."
        )
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        header.addLayout(title_layout)
        header.addStretch()

        self.count_label = QLabel("0 papers")
        self.count_label.setObjectName("topbarStatus")
        header.addWidget(self.count_label)

        root.addLayout(header)

        # -----------------------------------------------------
        # TOOLBAR
        # -----------------------------------------------------

        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText(
            "Search title, author, keyword, venue, DOI..."
        )
        self.search_input.setMinimumHeight(38)
        self.search_input.textChanged.connect(self._apply_filter)

        toolbar.addWidget(self.search_input, 1)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setObjectName("secondaryButton")
        self.refresh_button.setMinimumHeight(38)
        self.refresh_button.clicked.connect(self._refresh_from_research)

        toolbar.addWidget(self.refresh_button)

        root.addLayout(toolbar)

        # -----------------------------------------------------
        # PAPER TABLE
        # -----------------------------------------------------

        table_card = QFrame()
        table_card.setObjectName("researchCard")

        table_layout = QVBoxLayout(table_card)
        table_layout.setContentsMargins(12, 12, 12, 12)
        table_layout.setSpacing(8)

        self.paper_table = QTableWidget()
        self.paper_table.setColumnCount(7)
        self.paper_table.setHorizontalHeaderLabels(
            [
                "#",
                "Paper",
                "Year",
                "Relevance",
                "Usage",
                "Source",
                "Venue",
            ]
        )
        self.paper_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.paper_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.paper_table.setSelectionMode(QTableWidget.SingleSelection)
        self.paper_table.setAlternatingRowColors(True)
        self.paper_table.setSortingEnabled(True)

        table_header = self.paper_table.horizontalHeader()
        table_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table_header.setSectionResizeMode(1, QHeaderView.Stretch)
        table_header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        table_header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        table_header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        table_header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        table_header.setSectionResizeMode(6, QHeaderView.ResizeToContents)

        self.paper_table.itemSelectionChanged.connect(
            self._show_selected_paper
        )
        self.paper_table.cellDoubleClicked.connect(
            self._open_selected_source
        )

        table_layout.addWidget(self.paper_table, 1)
        root.addWidget(table_card, 1)

        # -----------------------------------------------------
        # DETAIL
        # -----------------------------------------------------

        detail_card = QFrame()
        detail_card.setObjectName("researchCard")

        detail_layout = QVBoxLayout(detail_card)
        detail_layout.setContentsMargins(16, 14, 16, 14)
        detail_layout.setSpacing(8)

        detail_title = QLabel("Paper Details")
        detail_title.setObjectName("sectionTitle")

        self.paper_detail = QTextEdit()
        self.paper_detail.setReadOnly(True)
        self.paper_detail.setMinimumHeight(165)
        self.paper_detail.setMaximumHeight(220)

        self.open_source_button = QPushButton("Open Paper Source")
        self.open_source_button.setObjectName("secondaryButton")
        self.open_source_button.setEnabled(False)
        self.open_source_button.clicked.connect(self._open_selected_source)

        detail_actions = QHBoxLayout()
        detail_actions.addStretch()
        detail_actions.addWidget(self.open_source_button)

        detail_layout.addWidget(detail_title)
        detail_layout.addWidget(self.paper_detail)
        detail_layout.addLayout(detail_actions)

        root.addWidget(detail_card)

        self.show_empty_state()

    # =====================================================
    # STATE
    # =====================================================

    def show_empty_state(self):
        self.papers = []
        self.filtered_papers = []
        self.paper_table.setSortingEnabled(False)
        self.paper_table.setRowCount(0)
        self.paper_table.setSortingEnabled(True)
        self.paper_detail.setPlainText(
            "No academic research results available.\n\n"
            "Run Academic Research first to populate this tool."
        )
        self.count_label.setText("0 papers")
        self.open_source_button.setEnabled(False)

    def update_from_result(self, result):
        if not isinstance(result, dict):
            self.show_empty_state()
            return

        papers = result.get("papers", [])
        if not isinstance(papers, list):
            papers = []

        self.papers = [
            dict(paper)
            for paper in papers
            if isinstance(paper, dict)
        ]

        self._apply_filter()

    def _refresh_from_research(self):
        result = getattr(
            self.main_window.research_page,
            "research_result",
            {},
        )
        self.update_from_result(result)

    # =====================================================
    # FILTER
    # =====================================================

    def _apply_filter(self):
        query = self.search_input.text().strip().lower()

        if not query:
            self.filtered_papers = list(self.papers)
        else:
            filtered = []

            for paper in self.papers:
                authors = paper.get("authors", [])
                if isinstance(authors, list):
                    authors_text = " ".join(str(a) for a in authors)
                else:
                    authors_text = str(authors)

                keywords = paper.get("keywords", [])
                if isinstance(keywords, list):
                    keywords_text = " ".join(
                        str(k) for k in keywords
                    )
                else:
                    keywords_text = str(keywords)

                searchable = " ".join(
                    [
                        str(paper.get("title", "")),
                        authors_text,
                        str(paper.get("venue", "")),
                        str(paper.get("doi", "")),
                        keywords_text,
                        str(paper.get("source", "")),
                        str(paper.get("year", "")),
                    ]
                ).lower()

                if query in searchable:
                    filtered.append(paper)

            self.filtered_papers = filtered

        self._populate_table()

    # =====================================================
    # TABLE
    # =====================================================

    def _populate_table(self):
        self.paper_table.setSortingEnabled(False)
        self.paper_table.setRowCount(
            len(self.filtered_papers)
        )

        for row, paper in enumerate(self.filtered_papers):
            relevance = paper.get("relevance_score")
            usage = paper.get(
                "dataset_usage_confidence",
                "UNKNOWN",
            )

            values = [
                str(row + 1),
                str(
                    paper.get(
                        "title",
                        "Untitled",
                    )
                ),
                str(
                    paper.get("year")
                    or "-"
                ),
                (
                    f"{relevance:.1f}%"
                    if isinstance(relevance, (int, float))
                    else "-"
                ),
                str(usage),
                str(
                    paper.get(
                        "source",
                        "-"
                    )
                    or "-"
                ),
                str(
                    paper.get(
                        "venue",
                        "-"
                    )
                    or "-"
                ),
            ]

            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)

                # Keep the underlying relevance numeric so sorting
                # remains useful when the table is manually sorted.
                if column == 3 and isinstance(
                    relevance,
                    (int, float),
                ):
                    item.setData(
                        Qt.UserRole,
                        float(relevance),
                    )

                self.paper_table.setItem(
                    row,
                    column,
                    item,
                )

        self.paper_table.setSortingEnabled(True)
        self.count_label.setText(
            f"{len(self.filtered_papers)} of {len(self.papers)} papers"
        )

        self.paper_detail.clear()
        self.open_source_button.setEnabled(False)

        if self.filtered_papers:
            self.paper_table.selectRow(0)
        else:
            self.paper_detail.setPlainText(
                "No papers match the current search."
            )

    # =====================================================
    # DETAIL
    # =====================================================

    def _show_selected_paper(self):
        rows = self.paper_table.selectionModel().selectedRows()

        if not rows:
            self.paper_detail.clear()
            self.open_source_button.setEnabled(False)
            return

        row = rows[0].row()

        if row < 0 or row >= len(self.filtered_papers):
            return

        paper = self.filtered_papers[row]

        authors = paper.get("authors", [])
        if isinstance(authors, list):
            authors_text = ", ".join(
                str(author)
                for author in authors
            )
        else:
            authors_text = str(authors)

        evidence = paper.get(
            "dataset_usage_evidence",
            [],
        )
        limitations = paper.get(
            "dataset_usage_limitations",
            [],
        )

        lines = [
            f"TITLE\n{paper.get('title') or '-'}",
            f"AUTHORS\n{authors_text or '-'}",
            f"YEAR\n{paper.get('year') or '-'}",
            f"VENUE\n{paper.get('venue') or '-'}",
            f"SOURCE\n{paper.get('source') or '-'}",
            f"DOI\n{paper.get('doi') or '-'}",
            (
                "RELEVANCE SCORE\n"
                + (
                    f"{paper.get('relevance_score'):.1f}%"
                    if isinstance(
                        paper.get("relevance_score"),
                        (int, float),
                    )
                    else "-"
                )
            ),
            (
                "DATASET USAGE CONFIDENCE\n"
                f"{paper.get('dataset_usage_confidence') or '-'}"
            ),
        ]

        if evidence:
            lines.append(
                "EVIDENCE\n"
                + "\n".join(
                    f"• {item}"
                    for item in evidence
                )
            )

        if limitations:
            lines.append(
                "LIMITATIONS\n"
                + "\n".join(
                    f"• {item}"
                    for item in limitations
                )
            )

        abstract = paper.get("abstract")
        if abstract:
            lines.append(
                f"ABSTRACT\n{abstract}"
            )

        url = paper.get("url")
        if url:
            lines.append(
                f"SOURCE URL\n{url}"
            )

        self.paper_detail.setPlainText(
            "\n\n".join(lines)
        )

        self.open_source_button.setEnabled(
            bool(url)
        )

    def _selected_paper(self):
        rows = self.paper_table.selectionModel().selectedRows()

        if not rows:
            return None

        row = rows[0].row()

        if row < 0 or row >= len(self.filtered_papers):
            return None

        return self.filtered_papers[row]

    def _open_selected_source(self):
        paper = self._selected_paper()

        if not paper:
            return

        url = paper.get("url")

        if not url:
            doi = paper.get("doi")
            if doi:
                url = f"https://doi.org/{doi}"

        if not url:
            QMessageBox.information(
                self,
                "Source Unavailable",
                "This paper does not provide a source URL or DOI.",
            )
            return

        try:
            QDesktopServices.openUrl(
                QUrl(str(url))
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Open Source Failed",
                f"Unable to open the paper source.\n\n{exc}",
            )


class ResearchLandscapeToolPage(QWidget):
    """Dedicated Research Tool: Research Landscape.

    Reads the already-generated Academic Research landscape result.
    No additional academic search is performed here.
    """

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(30, 25, 30, 25)
        root.setSpacing(16)

        header = QHBoxLayout()
        title_layout = QVBoxLayout()
        title = QLabel("Research Landscape")
        title.setObjectName("pageTitle")
        subtitle = QLabel(
            "Explore the methods, topics, and publication structure found in the analyzed literature."
        )
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        header.addLayout(title_layout, 1)

        self.refresh_button = QPushButton("Refresh Landscape")
        self.refresh_button.setObjectName("primaryButton")
        self.refresh_button.setMinimumHeight(42)
        self.refresh_button.setCursor(Qt.PointingHandCursor)
        self.refresh_button.clicked.connect(self.refresh)
        header.addWidget(self.refresh_button, 0, Qt.AlignTop)
        root.addLayout(header)

        self.status_label = QLabel("No academic research result available.")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        self.container = QWidget()
        self.content = QVBoxLayout(self.container)
        self.content.setContentsMargins(0, 0, 10, 20)
        self.content.setSpacing(14)
        scroll.setWidget(self.container)
        root.addWidget(scroll, 1)

        self.show_empty_state()

    def _clear(self):
        while self.content.count():
            item = self.content.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def show_empty_state(self):
        self._clear()
        card = QFrame()
        card.setObjectName("contentCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(28, 36, 28, 36)
        layout.setSpacing(10)

        title = QLabel("Research Landscape belum tersedia")
        title.setObjectName("sectionTitle")
        title.setAlignment(Qt.AlignCenter)
        desc = QLabel(
            "Run Academic Research terlebih dahulu.\n"
            "Setelah literature analysis selesai, hasil landscape akan muncul di sini."
        )
        desc.setObjectName("cardDescription")
        desc.setAlignment(Qt.AlignCenter)
        desc.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(desc)
        self.content.addWidget(card)
        self.content.addStretch()
        self.status_label.setText("No academic research result available.")

    def refresh(self):
        result = getattr(self.main_window.research_page, "research_result", {})
        if not result:
            self.show_empty_state()
            return
        self.update_from_result(result)

    @staticmethod
    def _safe_dict(value):
        return value if isinstance(value, dict) else {}

    @staticmethod
    def _pick(mapping, *keys, default=None):
        for key in keys:
            value = mapping.get(key)
            if value not in (None, "", [], {}):
                return value
        return default

    @staticmethod
    def _listify(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return list(value)
        if isinstance(value, dict):
            return list(value.items())
        return [value]

    @staticmethod
    def _pretty(value):
        if isinstance(value, bool):
            return "Yes" if value else "No"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    def _metric_card(self, title, value, subtitle=""):
        return InfoCard(title, self._pretty(value), subtitle)

    def _add_text_card(self, title, content):
        card = QFrame()
        card.setObjectName("contentCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(8)
        heading = QLabel(title)
        heading.setObjectName("sectionTitle")
        body = QLabel(content)
        body.setObjectName("cardDescription")
        body.setWordWrap(True)
        layout.addWidget(heading)
        layout.addWidget(body)
        self.content.addWidget(card)

    def _format_mapping(self, mapping):
        lines = []
        for key, value in mapping.items():
            if isinstance(value, (dict, list, tuple, set)):
                lines.append(f"{key}: {self._pretty(value)}")
            else:
                lines.append(f"{key}: {self._pretty(value)}")
        return "\n".join(lines)

    def _add_distribution_card(self, title, value):
        card = QFrame()
        card.setObjectName("contentCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(10)

        heading = QLabel(title)
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)

        rows = self._listify(value)
        if not rows:
            empty = QLabel("No distribution data available.")
            empty.setObjectName("cardDescription")
            layout.addWidget(empty)
        else:
            for item in rows[:12]:
                if isinstance(item, tuple) and len(item) == 2:
                    label, amount = item
                elif isinstance(item, dict):
                    label = self._pick(item, "method", "topic", "label", "name", "category", default="Item")
                    amount = self._pick(item, "percentage", "percent", "confidence", "score", "count", "value", default="-")
                else:
                    label = str(item)
                    amount = ""
                row = QHBoxLayout()
                name_label = QLabel(str(label))
                name_label.setObjectName("cardDescription")
                value_label = QLabel(self._pretty(amount))
                value_label.setObjectName("scoreBadge")
                row.addWidget(name_label, 1)
                row.addWidget(value_label, 0)
                layout.addLayout(row)

        self.content.addWidget(card)

    def update_from_result(self, result):
        self._clear()

        landscape = self._safe_dict(result.get("landscape"))
        summary = self._safe_dict(landscape.get("summary"))

        papers = result.get("papers", [])
        if not isinstance(papers, list):
            papers = []

        paper_count = self._pick(
            summary,
            "paper_count", "papers_analyzed", "total_papers", "papers_found",
            default=len(papers),
        )
        dominant_method = self._pick(
            summary,
            "dominant_method", "top_method", "most_common_method",
            default="-",
        )
        latest_year = self._pick(
            summary,
            "latest_publication_year", "latest_year", "max_year",
            default="-",
        )

        self.status_label.setText(
            f"Landscape loaded from Academic Research • {paper_count} papers analyzed"
        )

        cards = QHBoxLayout()
        cards.setSpacing(12)
        cards.addWidget(self._metric_card("Papers", paper_count, "Literature analyzed"))
        cards.addWidget(self._metric_card("Dominant Method", dominant_method, "Most represented method"))
        cards.addWidget(self._metric_card("Latest Publication", latest_year, "Newest publication year"))
        self.content.addLayout(cards)

        # Common analyzer field names + generic aliases.
        methods = self._pick(
            landscape,
            "method_distribution", "methods", "method_counts", "top_methods", "dominant_methods",
        )
        topics = self._pick(
            landscape,
            "topic_distribution", "topics", "topic_counts", "top_topics", "research_topics",
        )
        venues = self._pick(
            landscape,
            "venue_distribution", "venues", "venue_counts", "top_venues",
        )
        years = self._pick(
            landscape,
            "publication_years", "year_distribution", "years", "publication_trend",
        )

        if methods:
            self._add_distribution_card("Method Distribution", methods)
        if topics:
            self._add_distribution_card("Research Topics", topics)
        if venues:
            self._add_distribution_card("Publication Venues", venues)
        if years:
            self._add_distribution_card("Publication Years", years)

        # Always expose the full structured landscape so no backend field is hidden.
        if not any((methods, topics, venues, years)):
            self._add_text_card(
                "Landscape Analysis",
                self._format_mapping(landscape) or "No landscape details available.",
            )
        else:
            details = {k: v for k, v in landscape.items() if k not in {
                "summary", "method_distribution", "methods", "method_counts", "top_methods", "dominant_methods",
                "topic_distribution", "topics", "topic_counts", "top_topics", "research_topics",
                "venue_distribution", "venues", "venue_counts", "top_venues",
                "publication_years", "year_distribution", "years", "publication_trend",
            }}
            if details:
                self._add_text_card("Additional Landscape Details", self._format_mapping(details))

        self.content.addStretch()




class ResearchGapToolPage(QWidget):
    """Standalone potential research gap explorer."""
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.research_result: Dict[str, Any] = {}
        self.build_ui()

    def build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(30, 25, 30, 25)
        root.setSpacing(16)

        header = QHBoxLayout()
        box = QVBoxLayout()
        title = QLabel("Research Gap")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Explore potential research directions inferred from the analyzed literature.")
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)
        box.addWidget(title)
        box.addWidget(subtitle)
        header.addLayout(box, 1)
        refresh = QPushButton("Refresh")
        refresh.setObjectName("primaryButton")
        refresh.clicked.connect(self.refresh)
        header.addWidget(refresh)
        root.addLayout(header)

        self.status = QLabel("No research result available.")
        self.status.setObjectName("statusLabel")
        root.addWidget(self.status)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self.container = QWidget()
        self.content = QVBoxLayout(self.container)
        self.content.setContentsMargins(0, 4, 8, 16)
        self.content.setSpacing(14)
        scroll.setWidget(self.container)
        root.addWidget(scroll, 1)
        self.show_empty_state()

    def clear_content(self):
        while self.content.count():
            item = self.content.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
            elif item.layout():
                while item.layout().count():
                    child = item.layout().takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()

    def show_empty_state(self):
        self.research_result = {}
        self.status.setText("Run Academic Research first to unlock Research Gap.")
        self.clear_content()
        card = QFrame()
        card.setObjectName("contentCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(30, 40, 30, 40)
        title = QLabel("Research Gap belum tersedia")
        title.setObjectName("sectionTitle")
        title.setAlignment(Qt.AlignCenter)
        desc = QLabel("Run Academic Research to analyze potential research gaps from the discovered literature.")
        desc.setObjectName("cardDescription")
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignCenter)
        lay.addWidget(title)
        lay.addWidget(desc)
        self.content.addWidget(card)
        self.content.addStretch()

    def refresh(self):
        result = getattr(self.main_window.research_page, "research_result", {})
        if result:
            self.update_from_result(result)
        else:
            self.show_empty_state()

    def update_from_result(self, result):
        self.research_result = result or {}
        gaps = self.research_result.get("gaps", {})
        summary = gaps.get("summary", {}) if isinstance(gaps, dict) else {}
        items = []
        if isinstance(gaps, dict):
            items = gaps.get("gaps") or gaps.get("potential_gaps") or gaps.get("items") or []
        if not isinstance(items, list):
            items = []

        self.status.setText("Potential research gap analysis loaded from Academic Research.")
        self.clear_content()

        cards = QHBoxLayout()
        gap_count = summary.get("gap_count", len(items)) if isinstance(summary, dict) else len(items)
        conf = summary.get("overall_confidence", summary.get("confidence", "—")) if isinstance(summary, dict) else "—"
        cards.addWidget(InfoCard("Potential Gaps", str(gap_count), "Detected candidates"))
        cards.addWidget(InfoCard("Confidence", str(conf), "System estimation"))
        cards.addWidget(InfoCard("Papers", str(len(self.research_result.get("papers", []))), "Literature analyzed"))
        self.content.addLayout(cards)

        summary_card = QFrame()
        summary_card.setObjectName("contentCard")
        sl = QVBoxLayout(summary_card)
        st = QLabel("Gap Analysis Summary")
        st.setObjectName("sectionTitle")
        sl.addWidget(st)
        summary_text = ResearchPage._format_section(summary) if summary else "No summary available."
        sd = QLabel(summary_text)
        sd.setObjectName("cardDescription")
        sd.setWordWrap(True)
        sl.addWidget(sd)
        self.content.addWidget(summary_card)

        heading = QLabel("Potential Research Directions")
        heading.setObjectName("sectionTitle")
        self.content.addWidget(heading)

        if not items:
            empty = QLabel("No potential research gaps were detected.")
            empty.setObjectName("emptyState")
            self.content.addWidget(empty)
        else:
            for i, item in enumerate(items, 1):
                self.content.addWidget(self._gap_card(i, item))

        note = QFrame()
        note.setObjectName("contentCard")
        nl = QVBoxLayout(note)
        nt = QLabel("Interpretation")
        nt.setObjectName("cardLabel")
        nd = QLabel(
            "These are POTENTIAL_GAP candidates produced from the available literature and dataset context. "
            "They are not definitive scientific claims and should be validated through manual literature review."
        )
        nd.setObjectName("cardDescription")
        nd.setWordWrap(True)
        nl.addWidget(nt)
        nl.addWidget(nd)
        self.content.addWidget(note)
        self.content.addStretch()

    @staticmethod
    def _gap_card(index, item):
        card = QFrame()
        card.setObjectName("contentCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(20, 18, 20, 18)
        if isinstance(item, dict):
            title = item.get("title") or item.get("gap") or item.get("description") or "Potential research direction"
        else:
            title = str(item)
        lab = QLabel(f"{index:02d}  {title}")
        lab.setObjectName("sectionTitle")
        lab.setWordWrap(True)
        lay.addWidget(lab)
        if isinstance(item, dict):
            for key, value in item.items():
                if key in {"title", "gap", "description"} or value in (None, "", [], {}):
                    continue
                detail = QLabel(f"{ResearchPage._pretty_key(key)}: {ResearchPage._format_value(value)}")
                detail.setObjectName("cardDescription")
                detail.setWordWrap(True)
                lay.addWidget(detail)
        return card


class ResearchReportToolPage(QWidget):
    """Standalone report preview and HTML export."""
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.research_result: Dict[str, Any] = {}
        self.current_html = ""
        self.build_ui()

    def build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(30, 25, 30, 25)
        root.setSpacing(16)
        header = QHBoxLayout()
        box = QVBoxLayout()
        title = QLabel("Research Report")
        title.setObjectName("pageTitle")
        subtitle = QLabel("Generate a structured report from the latest Dataset Research analysis.")
        subtitle.setObjectName("pageSubtitle")
        subtitle.setWordWrap(True)
        box.addWidget(title)
        box.addWidget(subtitle)
        header.addLayout(box, 1)

        self.generate_button = QPushButton("Generate Report")
        self.generate_button.setObjectName("primaryButton")
        self.generate_button.clicked.connect(self.generate_preview)
        header.addWidget(self.generate_button)
        self.export_button = QPushButton("Export HTML")
        self.export_button.clicked.connect(self.export_html)
        self.export_button.setEnabled(False)
        header.addWidget(self.export_button)
        root.addLayout(header)

        self.status = QLabel("No research result available.")
        self.status.setObjectName("statusLabel")
        root.addWidget(self.status)
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        root.addWidget(self.preview, 1)
        self.show_empty_state()

    def show_empty_state(self):
        self.research_result = {}
        self.current_html = ""
        self.status.setText("Run Academic Research first to generate a report.")
        self.preview.setPlainText("Research Report belum tersedia.\n\nRun Academic Research terlebih dahulu.")
        self.export_button.setEnabled(False)

    def update_from_result(self, result):
        self.research_result = result or {}
        self.generate_preview()

    def refresh_from_main(self):
        self.research_result = getattr(self.main_window.research_page, "research_result", {}) or {}

    def generate_preview(self):
        if not self.research_result:
            self.refresh_from_main()
        if not self.research_result:
            self.show_empty_state()
            return
        self.preview.setPlainText(self._build_report_text(self.research_result))
        self.current_html = self._build_report_html(self.research_result)
        self.status.setText("Report generated from the latest Academic Research result.")
        self.export_button.setEnabled(True)

    def export_html(self):
        if not self.current_html:
            self.generate_preview()
        if not self.current_html:
            return
        filename = Path(self.main_window._current_filename()).stem or "dataset"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Research Report", f"{filename}_research_report.html", "HTML Files (*.html)"
        )
        if not path:
            return
        try:
            Path(path).write_text(self.current_html, encoding="utf-8")
            QMessageBox.information(self, "Report Exported", f"Research report saved to:\n{path}")
        except OSError as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    @staticmethod
    def _build_report_text(result):
        summary = result.get("summary", {})
        keywords = result.get("keywords", {})
        domain = result.get("domain", {})
        papers = result.get("papers", [])
        landscape = result.get("landscape", {})
        trend = result.get("trend", {})
        gaps = result.get("gaps", {})
        lines = [
            "DATASET RESEARCH REPORT",
            "=" * 80,
            "",
            "RESEARCH SUMMARY",
            ResearchPage._format_section(summary),
            "",
            "KEYWORDS",
            ResearchPage._format_section(keywords),
            "",
            "RESEARCH DOMAIN",
            ResearchPage._format_section(domain),
            "",
            f"ACADEMIC PAPERS ({len(papers)})",
        ]
        for i, paper in enumerate(papers[:10], 1):
            lines.append(
                f"{i}. {paper.get('title', 'Untitled')} | {paper.get('year') or '-'} | "
                f"Relevance {ResearchPage._format_value(paper.get('relevance_score'))}"
            )
        lines.extend([
            "",
            "RESEARCH LANDSCAPE",
            ResearchPage._format_section(landscape),
            "",
            "RESEARCH TREND",
            ResearchPage._format_section(trend),
            "",
            "POTENTIAL RESEARCH GAP",
            ResearchPage._format_gap(gaps),
            "",
            "TRANSPARENCY",
            ResearchPage._format_section(result.get("transparency", {})),
        ])
        return "\n".join(lines)

    @staticmethod
    def _build_report_html(result):
        import html
        text = ResearchReportToolPage._build_report_text(result)
        escaped = html.escape(text)
        return (
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>Dataset Research Report</title>"
            "<style>body{font-family:Segoe UI,Arial,sans-serif;background:#f5f8fc;"
            "color:#1f2937;margin:0;padding:36px}main{max-width:1000px;margin:auto;"
            "background:#fff;padding:36px;border-radius:16px;box-shadow:0 8px 30px rgba(15,23,42,.08)}"
            "h1{color:#2563eb}pre{white-space:pre-wrap;font-family:Consolas,monospace;"
            "line-height:1.55}</style></head><body><main><h1>Dataset Research Report</h1>"
            f"<pre>{escaped}</pre></main></body></html>"
        )

class MainWindow(QMainWindow):
    """Main application window for Dataset Research.

    The UI pages are intentionally kept in this single module so the
    desktop interface has one clear entry point while the analytical
    backend remains completely independent.
    """

    def __init__(self):
        super().__init__()

        # Load the stylesheet before building the interface so every
        # widget is styled as soon as it is created.
        self._load_stylesheet()

        self.setWindowTitle("Dataset Research")
        self.resize(1180, 720)
        self.setMinimumSize(1100, 680)

        self.current_dataset = None
        self.current_file_path = None
        self.analysis_result = {}

        self.loader = DatasetLoader()
        self.profiler = DatasetProfiler()
        self.statistics = DatasetStatistics()
        self.missing_analyzer = MissingValueAnalyzer()
        self.duplicate_analyzer = DuplicateAnalyzer()
        self.outlier_analyzer = OutlierAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.fingerprint_analyzer = DatasetFingerprint()

        self._build_window()

    # =========================================================
    # STYLESHEET
    # =========================================================

    def _load_stylesheet(self):
        """Load the UI stylesheet located beside this module."""
        qss_path = Path(__file__).with_name("app.qss")

        if not qss_path.exists():
            print(f"Warning: stylesheet not found: {qss_path}")
            return

        try:
            stylesheet = qss_path.read_text(encoding="utf-8")
            self.setStyleSheet(stylesheet)
        except OSError as error:
            print(f"Warning: failed to load stylesheet: {error}")

    # =========================================================
    # WINDOW
    # =========================================================

    def _build_window(self):
        root = QWidget()
        root.setObjectName("appRoot")
        self.setCentralWidget(root)

        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_sidebar())

        content = QWidget()
        content.setObjectName("contentContainer")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        self.topbar = self._build_topbar()
        content_layout.addWidget(self.topbar)

        self.page_stack = QStackedWidget()
        self.page_stack.setObjectName("pageStack")
        content_layout.addWidget(self.page_stack, 1)

        root_layout.addWidget(content, 1)

        self.dashboard = DashboardPage(self)
        self.upload_page = UploadPage(self)
        self.analysis_page = AnalysisPage(self)
        self.ml_page = MLIntelligencePage(self)
        self.research_page = ResearchPage(self)
        self.papers_tool_page = PapersToolPage(self)
        self.landscape_tool_page = ResearchLandscapeToolPage(self)
        self.gap_tool_page = ResearchGapToolPage(self)
        self.report_tool_page = ResearchReportToolPage(self)

        self.pages = {
            "dashboard": self.dashboard,
            "dataset": self.upload_page,
            "analysis": self.analysis_page,
            "ml": self.ml_page,
            "research": self.research_page,
            "papers": self.papers_tool_page,
            "landscape": self.landscape_tool_page,
            "gap": self.gap_tool_page,
            "report": self.report_tool_page,
        }

        for page in self.pages.values():
            self.page_stack.addWidget(page)

        self._page_titles = {
            "dashboard": "Dashboard",
            "dataset": "Dataset",
            "analysis": "Dataset Analysis",
            "ml": "ML Intelligence",
            "research": "Academic Research",
            "papers": "Papers",
            "landscape": "Research Landscape",
            "gap": "Research Gap",
            "report": "Research Report",
        }

        self.open_dashboard()

    # =========================================================
    # SIDEBAR
    # =========================================================

    def _build_sidebar(self):
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(250)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(16, 18, 16, 16)
        layout.setSpacing(6)

        brand = QHBoxLayout()
        brand.setSpacing(10)

        logo = QLabel()
        logo.setObjectName("brandLogo")
        logo.setFixedSize(44, 44)
        logo.setAlignment(Qt.AlignCenter)

        logo_path = _resource_path("assets/logo.jpg")
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            if not pixmap.isNull():
                logo.setPixmap(
                    pixmap.scaled(
                        44,
                        44,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                )
        else:
            logo.setText("DR")

        brand_text = QVBoxLayout()
        brand_text.setSpacing(0)

        title = QLabel("DATASET")
        title.setObjectName("brandTitle")

        accent = QLabel("RESEARCH")
        accent.setObjectName("brandAccent")

        subtitle = QLabel("Research Intelligence")
        subtitle.setObjectName("brandSubtitle")

        brand_text.addWidget(title)
        brand_text.addWidget(accent)
        brand_text.addWidget(subtitle)

        brand.addWidget(logo)
        brand.addLayout(brand_text, 1)
        layout.addLayout(brand)

        layout.addSpacing(20)

        workspace = QLabel("WORKSPACE")
        workspace.setObjectName("sectionLabel")
        layout.addWidget(workspace)

        self.nav_buttons = {}

        self._add_nav_button(layout, "dashboard", "⌂", "Dashboard")
        self._add_nav_button(layout, "dataset", "▣", "Dataset")
        self._add_nav_button(layout, "analysis", "◇", "Analysis")
        self._add_nav_button(layout, "ml", "✦", "ML Intelligence")
        self._add_nav_button(layout, "research", "◎", "Academic Research")

        layout.addSpacing(18)

        tools = QLabel("RESEARCH TOOLS")
        tools.setObjectName("sectionLabel")
        layout.addWidget(tools)

        for key, icon, text, enabled in [
            ("papers", "▤", "Papers", True),
            ("landscape", "▥", "Research Landscape", True),
            ("gap", "◇", "Research Gap", True),
            ("report", "▰", "Research Report", True),
        ]:
            self._add_nav_button(
                layout,
                key,
                icon,
                text,
                enabled=enabled,
                tool_item=True,
            )

        layout.addStretch()

        status_card = QFrame()
        status_card.setObjectName("workspaceStatus")
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(13, 11, 13, 11)
        status_layout.setSpacing(3)

        status_title = QLabel("LOCAL WORKSPACE")
        status_title.setObjectName("workspaceStatusTitle")
        status_value = QLabel("●  Ready")
        status_value.setObjectName("workspaceStatusValue")

        status_layout.addWidget(status_title)
        status_layout.addWidget(status_value)
        layout.addWidget(status_card)

        footer = QLabel("Dataset Research\n© 2024 Sains Data")
        footer.setObjectName("sidebarFooter")
        footer.setWordWrap(True)
        layout.addWidget(footer)

        return sidebar

    def _add_nav_button(self, layout, key, icon, text, enabled=True, tool_item=False):
        button = QPushButton()
        button.setObjectName("navButton")
        button.setEnabled(enabled)
        button.setProperty("toolItem", bool(tool_item))
        button.setCursor(
            Qt.PointingHandCursor if enabled else Qt.ArrowCursor
        )

        row = QHBoxLayout(button)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(11)

        icon_label = QLabel(icon)
        icon_label.setObjectName("navIcon")
        icon_label.setFixedWidth(22)
        icon_label.setAlignment(Qt.AlignCenter)

        text_label = QLabel(text)
        text_label.setObjectName("navText")

        row.addWidget(icon_label)
        row.addWidget(text_label)
        row.addStretch()

        layout.addWidget(button)
        self.nav_buttons[key] = button

        callbacks = {
            "dashboard": self.open_dashboard,
            "dataset": self.open_upload_page,
            "analysis": self.open_analysis_page,
            "ml": self.open_ml_page,
            "research": self.open_research_page,
            "papers": self.open_papers_tool,
            "landscape": self.open_landscape_tool,
            "gap": self.open_gap_tool,
            "report": self.open_report_tool,
        }

        if key in callbacks:
            button.clicked.connect(callbacks[key])

    # =========================================================
    # TOPBAR
    # =========================================================

    def _build_topbar(self):
        topbar = QFrame()
        topbar.setObjectName("topbar")
        topbar.setFixedHeight(72)

        layout = QHBoxLayout(topbar)
        layout.setContentsMargins(26, 0, 26, 0)

        self.page_title = QLabel("Dashboard")
        self.page_title.setObjectName("topbarTitle")

        layout.addWidget(self.page_title)
        layout.addStretch()

        status = QLabel("●  LOCAL WORKSPACE")
        status.setObjectName("topbarStatus")
        layout.addWidget(status)

        return topbar

    # =========================================================
    # NAVIGATION
    # =========================================================

    def _show_page(self, key):
        page = self.pages[key]
        self.page_stack.setCurrentWidget(page)
        self.page_title.setText(self._page_titles[key])
        self.set_active_button(key)

    def open_dashboard(self):
        self._show_page("dashboard")

    def open_upload_page(self):
        self._show_page("dataset")

    def open_analysis_page(self):
        if self.current_dataset is not None:
            self.analysis_page.update_dataset(
                self.current_dataset,
                self._current_filename(),
            )
        self._show_page("analysis")

    def open_ml_page(self):
        if self.current_dataset is None or not self.analysis_result:
            self.ml_page.show_empty_state()
        else:
            self.ml_page.update_dataset()
        self._show_page("ml")

    def open_research_page(self):
        if self.current_dataset is None or not self.analysis_result:
            self.research_page.show_empty_state()
        self._show_page("research")

    def open_papers_tool(self):
        if not self.analysis_result or not self.research_page.research_result:
            self.papers_tool_page.show_empty_state()
        else:
            self.papers_tool_page.update_from_result(
                self.research_page.research_result
            )
        self._show_page("papers")

    def open_landscape_tool(self):
        if not self.analysis_result or not self.research_page.research_result:
            self.landscape_tool_page.show_empty_state()
        else:
            self.landscape_tool_page.update_from_result(
                self.research_page.research_result
            )
        self._show_page("landscape")

    def open_gap_tool(self):
        if not self.analysis_result or not self.research_page.research_result:
            self.gap_tool_page.show_empty_state()
        else:
            self.gap_tool_page.update_from_result(self.research_page.research_result)
        self._show_page("gap")

    def open_report_tool(self):
        if not self.analysis_result or not self.research_page.research_result:
            self.report_tool_page.show_empty_state()
        else:
            self.report_tool_page.update_from_result(self.research_page.research_result)
        self._show_page("report")

    def _current_filename(self):
        if self.current_file_path:
            try:
                return Path(self.current_file_path).name
            except Exception:
                pass
        return "Dataset"

    def set_active_button(self, key):
        for name, button in self.nav_buttons.items():
            active = name == key
            button.setProperty("active", active)
            style = button.style()
            style.unpolish(button)
            style.polish(button)
            button.update()

    # =========================================================
    # DATASET STATE
    # =========================================================

    def update_dataset_state(self):
        self.analysis_result = {}

        filename = self._current_filename()

        self.dashboard.update_dataset(
            self.current_dataset,
            filename,
        )

        self.analysis_page.update_dataset(
            self.current_dataset,
            filename,
        )

        self.ml_page.show_empty_state()
        self.research_page.show_empty_state()
        self.papers_tool_page.show_empty_state()
        self.landscape_tool_page.show_empty_state()
        self.gap_tool_page.show_empty_state()
        self.report_tool_page.show_empty_state()

    # =========================================================
    # DATASET ANALYSIS
    # =========================================================

    def run_dataset_analysis(self):
        dataframe = self.current_dataset

        if dataframe is None:
            raise ValueError("No dataset is currently loaded.")

        result = {
            "profile": self.profiler.profile(dataframe),
            "statistics": self.statistics.analyze(dataframe),
            "missing_values": self.missing_analyzer.analyze(dataframe),
            "duplicates": self.duplicate_analyzer.analyze(dataframe),
            "outliers": self.outlier_analyzer.analyze(dataframe),
            "correlations": self.correlation_analyzer.analyze(dataframe),
            "fingerprint": self.fingerprint_analyzer.generate(dataframe),
        }

        self.analysis_result = result

        self.ml_page.update_dataset()
        self.research_page.show_empty_state()

        return result


__all__ = [
    "StatCard",
    "QuickActionCard",
    "PipelineStage",
    "DashboardPage",
    "UploadPage",
    "AnalysisPage",
    "InfoCard",
    "TaskCard",
    "MethodCard",
    "MethodDetailCard",
    "MLIntelligencePage",
    "ResearchPage",
    "PapersToolPage",
    "ResearchLandscapeToolPage",
    "ResearchGapToolPage",
    "ResearchReportToolPage",
    "MainWindow",
]
