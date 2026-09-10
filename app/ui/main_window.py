from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
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

class DashboardPage(QWidget):

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.build_ui()

    def build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        # =================================================
        # HERO
        # =================================================
        hero = QFrame()
        hero.setObjectName("heroCard")
        hero.setFixedHeight(178)

        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(24, 18, 24, 18)
        hero_layout.setSpacing(18)

        # Keep the hero left side intentionally minimal.
        # The previous heading/description could overflow on compact windows.
        hero_text = QVBoxLayout()
        hero_text.setContentsMargins(0, 0, 0, 0)
        hero_text.setSpacing(0)

        start_button = QPushButton("Upload Dataset")
        start_button.setObjectName("heroButton")
        start_button.setFixedSize(150, 38)
        start_button.setCursor(Qt.PointingHandCursor)
        start_button.clicked.connect(self.main_window.open_upload_page)

        # Center the only hero action vertically and horizontally.
        hero_text.addStretch(1)
        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.addStretch(1)
        button_row.addWidget(start_button)
        button_row.addStretch(1)
        hero_text.addLayout(button_row)
        hero_text.addStretch(1)

        hero_layout.addLayout(hero_text, 1)

        visual = QFrame()
        visual.setObjectName("heroVisual")
        visual.setMinimumWidth(245)
        visual.setMaximumWidth(280)

        visual_layout = QVBoxLayout(visual)
        visual_layout.setContentsMargins(16, 12, 16, 12)
        visual_layout.setSpacing(5)

        visual_title = QLabel("RESEARCH WORKFLOW")
        visual_title.setObjectName("heroVisualTitle")
        visual_layout.addWidget(visual_title)

        workflow_items = [
            ("01", "Dataset"),
            ("02", "Analysis"),
            ("03", "ML Intelligence"),
            ("04", "Academic Research"),
        ]

        for number, title in workflow_items:
            item = QHBoxLayout()
            item.setSpacing(9)

            number_label = QLabel(number)
            number_label.setObjectName("heroWorkflowNumber")
            number_label.setFixedWidth(22)

            title_label = QLabel(title)
            title_label.setObjectName("heroWorkflowTitle")

            item.addWidget(number_label)
            item.addWidget(title_label)
            item.addStretch()
            visual_layout.addLayout(item)

        visual_layout.addStretch()
        hero_layout.addWidget(visual)
        layout.addWidget(hero)

        # =================================================
        # QUICK ACTIONS
        # =================================================
        quick_header = QLabel("Quick Actions")
        quick_header.setObjectName("sectionTitle")
        layout.addWidget(quick_header)

        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(10)

        cards = [
            ("↑", "Upload Dataset", "Import a dataset and start.", self.main_window.open_upload_page),
            ("◇", "Analyze Dataset", "Explore data quality and patterns.", self.main_window.open_analysis_page),
            ("✦", "ML Intelligence", "Discover suitable ML methods.", self.main_window.open_ml_page),
            ("◎", "Academic Research", "Find related papers and gaps.", self.main_window.open_research_page),
        ]

        for icon, title, description, callback in cards:
            card = QuickActionCard(icon, title, description, callback)
            card.setMinimumWidth(0)
            card.setMinimumHeight(104)
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            actions_layout.addWidget(card, 1)

        layout.addLayout(actions_layout)

        # =================================================
        # CURRENT DATASET
        # =================================================
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
        dataset_card.setFixedHeight(72)

        dataset_layout = QHBoxLayout(dataset_card)
        dataset_layout.setContentsMargins(16, 10, 16, 10)
        dataset_layout.setSpacing(14)

        dataset_info = QVBoxLayout()
        dataset_info.setSpacing(2)

        self.dataset_name_label = QLabel("No dataset loaded")
        self.dataset_name_label.setObjectName("datasetName")

        self.dataset_description_label = QLabel(
            "Upload a dataset to begin your research workflow."
        )
        self.dataset_description_label.setObjectName("datasetDescription")
        self.dataset_description_label.setWordWrap(True)

        dataset_info.addWidget(self.dataset_name_label)
        dataset_info.addWidget(self.dataset_description_label)
        dataset_layout.addLayout(dataset_info, 1)

        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(8)

        self.rows_card = StatCard("ROWS", "—", "Records")
        self.columns_card = StatCard("COLUMNS", "—", "Features")
        self.rows_card.setFixedSize(92, 58)
        self.columns_card.setFixedSize(92, 58)

        stats_layout.addWidget(self.rows_card)
        stats_layout.addWidget(self.columns_card)
        dataset_layout.addLayout(stats_layout)
        layout.addWidget(dataset_card)

        # =================================================
        # RESEARCH PIPELINE
        # =================================================
        pipeline_title = QLabel("Research Pipeline")
        pipeline_title.setObjectName("sectionTitle")
        layout.addWidget(pipeline_title)

        pipeline_card = QFrame()
        pipeline_card.setObjectName("pipelineCard")
        pipeline_card.setFixedHeight(68)

        pipeline_layout = QHBoxLayout(pipeline_card)
        pipeline_layout.setContentsMargins(8, 7, 8, 7)
        pipeline_layout.setSpacing(3)

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

        if hasattr(self, "pipeline_stages"):
            if len(self.pipeline_stages) >= 1:
                self.pipeline_stages[0].set_status("READY", "success")
            if len(self.pipeline_stages) >= 2:
                self.pipeline_stages[1].set_status("NEXT", "active")
            for stage in self.pipeline_stages[2:4]:
                stage.set_status("WAITING", "waiting")
            for stage in self.pipeline_stages[4:]:
                stage.set_status("V2", "future")

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

        if hasattr(self, "pipeline_stages"):
            for index, stage in enumerate(self.pipeline_stages):
                if index == 0:
                    stage.set_status("UPLOAD", "active")
                elif index < 4:
                    stage.set_status("WAITING", "waiting")
                else:
                    stage.set_status("V2", "future")

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

        self.pages = {
            "dashboard": self.dashboard,
            "dataset": self.upload_page,
            "analysis": self.analysis_page,
            "ml": self.ml_page,
            "research": self.research_page,
        }

        for page in self.pages.values():
            self.page_stack.addWidget(page)

        self._page_titles = {
            "dashboard": "Dashboard",
            "dataset": "Dataset",
            "analysis": "Dataset Analysis",
            "ml": "ML Intelligence",
            "research": "Academic Research",
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

        logo_path = Path(__file__).resolve().parent / "assets" / "logo.jpg"
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

        for key, icon, text in [
            ("papers", "○", "Papers"),
            ("landscape", "○", "Research Landscape"),
            ("gap", "○", "Research Gap"),
            ("report", "○", "Research Report"),
        ]:
            self._add_nav_button(
                layout,
                key,
                icon,
                text,
                enabled=False,
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

    def _add_nav_button(self, layout, key, icon, text, enabled=True):
        button = QPushButton()
        button.setObjectName("navButton")
        button.setEnabled(enabled)
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
    "MainWindow",
]
