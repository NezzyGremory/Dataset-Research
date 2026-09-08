from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QFrame,
    QStackedWidget,
    QSizePolicy,
)

from app.analyzer.loader import DatasetLoader
from app.analyzer.profiler import DatasetProfiler
from app.analyzer.statistics import DatasetStatistics
from app.analyzer.missing_values import MissingValueAnalyzer
from app.analyzer.duplicates import DuplicateAnalyzer
from app.analyzer.outliers import OutlierAnalyzer
from app.analyzer.correlations import CorrelationAnalyzer
from app.analyzer.fingerprint import DatasetFingerprint

from app.ui.dashboard import DashboardPage
from app.ui.upload_page import UploadPage
from app.ui.analysis_page import AnalysisPage
from app.ui.ml_page import MLIntelligencePage
from app.ui.research_page import ResearchPage


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        # =================================================
        # WINDOW
        # =================================================

        self.setWindowTitle("Dataset Research")

        self.resize(
            1180,
            720
        )

        self.setMinimumSize(
            1100,
            700
        )

        # =================================================
        # APPLICATION STATE
        # =================================================

        self.current_dataset = None
        self.current_file_path = None
        self.analysis_result = {}

        # =================================================
        # ANALYZERS
        # =================================================

        self.loader = DatasetLoader()
        self.profiler = DatasetProfiler()
        self.statistics = DatasetStatistics()

        self.missing_analyzer = MissingValueAnalyzer()
        self.duplicate_analyzer = DuplicateAnalyzer()
        self.outlier_analyzer = OutlierAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()

        self.fingerprint_generator = DatasetFingerprint()

        # =================================================
        # CENTRAL WIDGET
        # =================================================

        central = QWidget()
        central.setObjectName(
            "appRoot"
        )

        self.setCentralWidget(
            central
        )

        main_layout = QHBoxLayout(
            central
        )

        main_layout.setContentsMargins(
            0,
            0,
            0,
            0
        )

        main_layout.setSpacing(
            0
        )

        # =================================================
        # SIDEBAR
        # =================================================

        self.sidebar = self.create_sidebar()

        main_layout.addWidget(
            self.sidebar
        )

        # =================================================
        # CONTENT CONTAINER
        # =================================================

        content_container = QFrame()
        content_container.setObjectName(
            "contentContainer"
        )

        content_layout = QVBoxLayout(
            content_container
        )

        content_layout.setContentsMargins(
            0,
            0,
            0,
            0
        )

        content_layout.setSpacing(
            0
        )

        # =================================================
        # TOP BAR
        # =================================================

        self.topbar = self.create_topbar()

        content_layout.addWidget(
            self.topbar
        )

        # =================================================
        # PAGE STACK
        # =================================================

        self.stack = QStackedWidget()
        self.stack.setObjectName(
            "pageStack"
        )

        content_layout.addWidget(
            self.stack,
            1
        )

        main_layout.addWidget(
            content_container,
            1
        )

        # =================================================
        # PAGES
        # =================================================

        self.dashboard = DashboardPage(
            self
        )

        self.upload_page = UploadPage(
            self
        )

        self.analysis_page = AnalysisPage(
            self
        )

        self.ml_page = MLIntelligencePage(
            self
        )

        self.research_page = ResearchPage(
            self
        )

        # =================================================
        # ADD PAGES
        # =================================================

        self.stack.addWidget(
            self.dashboard
        )

        self.stack.addWidget(
            self.upload_page
        )

        self.stack.addWidget(
            self.analysis_page
        )

        self.stack.addWidget(
            self.ml_page
        )

        self.stack.addWidget(
            self.research_page
        )

        # =================================================
        # INITIAL PAGE
        # =================================================

        self.open_dashboard()

    # =====================================================
    # SIDEBAR
    # =====================================================

    def create_sidebar(self):

        sidebar = QFrame()

        sidebar.setObjectName(
            "sidebar"
        )

        sidebar.setFixedWidth(
            250
        )

        layout = QVBoxLayout(
            sidebar
        )

        layout.setContentsMargins(
            18,
            22,
            18,
            18
        )

        layout.setSpacing(
            6
        )

        # =================================================
        # BRAND
        # =================================================

        brand_container = QFrame()

        brand_container.setObjectName(
            "brandContainer"
        )

        brand_layout = QHBoxLayout(
            brand_container
        )

        brand_layout.setContentsMargins(
            6,
            4,
            6,
            12
        )

        brand_layout.setSpacing(
            11
        )

        # -------------------------------------------------
        # LOGO
        # -------------------------------------------------

        logo_label = QLabel()

        logo_label.setObjectName(
            "appLogo"
        )

        logo_path = (
            Path(__file__).resolve().parent
            / "assets"
            / "logo.jpg"
        )

        pixmap = QPixmap(
            str(logo_path)
        )

        if not pixmap.isNull():

            logo_label.setPixmap(
                pixmap.scaled(
                    44,
                    44,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
            )

        else:

            # Fallback jika logo tidak ditemukan
            logo_label.setText(
                "DR"
            )

        logo_label.setFixedSize(
            44,
            44
        )

        logo_label.setAlignment(
            Qt.AlignCenter
        )

        brand_layout.addWidget(
            logo_label
        )

        # -------------------------------------------------
        # BRAND TEXT
        # -------------------------------------------------

        brand_text = QVBoxLayout()

        brand_text.setContentsMargins(
            0,
            0,
            0,
            0
        )

        brand_text.setSpacing(
            0
        )

        brand = QLabel(
            "DATASET"
        )

        brand.setObjectName(
            "brandTitle"
        )

        research = QLabel(
            "RESEARCH"
        )

        research.setObjectName(
            "brandAccent"
        )

        subtitle = QLabel(
            "Research Intelligence"
        )

        subtitle.setObjectName(
            "brandSubtitle"
        )

        brand_text.addWidget(
            brand
        )

        brand_text.addWidget(
            research
        )

        brand_text.addSpacing(
            3
        )

        brand_text.addWidget(
            subtitle
        )

        brand_layout.addLayout(
            brand_text
        )

        layout.addWidget(
            brand_container
        )

        layout.addSpacing(
            18
        )

        # =================================================
        # MAIN NAVIGATION
        # =================================================

        self.add_section_label(
            layout,
            "WORKSPACE"
        )

        # -------------------------------------------------
        # DASHBOARD
        # -------------------------------------------------

        self.dashboard_button = self.create_nav_button(
            "⌂",
            "Dashboard"
        )

        self.dashboard_button.clicked.connect(
            self.open_dashboard
        )

        layout.addWidget(
            self.dashboard_button
        )

        # -------------------------------------------------
        # DATASET
        # -------------------------------------------------

        self.dataset_button = self.create_nav_button(
            "▣",
            "Dataset"
        )

        self.dataset_button.clicked.connect(
            self.open_upload_page
        )

        layout.addWidget(
            self.dataset_button
        )

        # -------------------------------------------------
        # ANALYSIS
        # -------------------------------------------------

        self.analysis_button = self.create_nav_button(
            "◇",
            "Analysis"
        )

        self.analysis_button.clicked.connect(
            self.open_analysis_page
        )

        layout.addWidget(
            self.analysis_button
        )

        # -------------------------------------------------
        # ML INTELLIGENCE
        # -------------------------------------------------

        self.ml_button = self.create_nav_button(
            "✦",
            "ML Intelligence"
        )

        self.ml_button.clicked.connect(
            self.open_ml_page
        )

        layout.addWidget(
            self.ml_button
        )

        # -------------------------------------------------
        # ACADEMIC RESEARCH
        # -------------------------------------------------

        self.research_button = self.create_nav_button(
            "◎",
            "Academic Research"
        )

        self.research_button.clicked.connect(
            self.open_research_page
        )

        layout.addWidget(
            self.research_button
        )

        layout.addSpacing(
            18
        )

        # =================================================
        # RESEARCH TOOLS
        # =================================================

        self.add_section_label(
            layout,
            "RESEARCH TOOLS"
        )

        # -------------------------------------------------
        # PAPERS
        # -------------------------------------------------

        self.papers_button = self.create_nav_button(
            "○",
            "Papers",
            disabled=True
        )

        layout.addWidget(
            self.papers_button
        )

        # -------------------------------------------------
        # RESEARCH LANDSCAPE
        # -------------------------------------------------

        self.landscape_button = self.create_nav_button(
            "○",
            "Research Landscape",
            disabled=True
        )

        layout.addWidget(
            self.landscape_button
        )

        # -------------------------------------------------
        # RESEARCH GAP
        # -------------------------------------------------

        self.gap_button = self.create_nav_button(
            "○",
            "Research Gap",
            disabled=True
        )

        layout.addWidget(
            self.gap_button
        )

        # -------------------------------------------------
        # RESEARCH REPORT
        # -------------------------------------------------

        self.report_button = self.create_nav_button(
            "○",
            "Research Report",
            disabled=True
        )

        layout.addWidget(
            self.report_button
        )

        # =================================================
        # SPACER
        # =================================================

        layout.addStretch(
            1
        )

        # =================================================
        # WORKSPACE STATUS
        # =================================================

        status_card = QFrame()

        status_card.setObjectName(
            "workspaceStatus"
        )

        status_layout = QVBoxLayout(
            status_card
        )

        status_layout.setContentsMargins(
            12,
            10,
            12,
            10
        )

        status_layout.setSpacing(
            3
        )

        status_title = QLabel(
            "LOCAL WORKSPACE"
        )

        status_title.setObjectName(
            "workspaceStatusTitle"
        )

        status_value = QLabel(
            "●  Ready"
        )

        status_value.setObjectName(
            "workspaceStatusValue"
        )

        status_layout.addWidget(
            status_title
        )

        status_layout.addWidget(
            status_value
        )

        layout.addWidget(
            status_card
        )

        layout.addSpacing(
            10
        )

        # =================================================
        # FOOTER
        # =================================================

        footer = QLabel(
            "Dataset Research\n"
            "© 2024 Sains Data"
        )

        footer.setObjectName(
            "sidebarFooter"
        )

        footer.setAlignment(
            Qt.AlignLeft
            | Qt.AlignBottom
        )

        layout.addWidget(
            footer
        )

        return sidebar

    # =====================================================
    # SECTION LABEL
    # =====================================================

    @staticmethod
    def add_section_label(
        layout,
        text
    ):

        label = QLabel(
            text
        )

        label.setObjectName(
            "navSectionLabel"
        )

        label.setContentsMargins(
            8,
            0,
            0,
            5
        )

        layout.addWidget(
            label
        )

    # =====================================================
    # NAVIGATION BUTTON
    # =====================================================

    @staticmethod
    def create_nav_button(
        icon,
        text,
        disabled=False
    ):

        button = QPushButton()

        button.setObjectName(
            "navButton"
        )

        button.setMinimumHeight(
            44
        )

        button.setCursor(
            Qt.PointingHandCursor
        )

        button.setEnabled(
            not disabled
        )

        button.setProperty(
            "disabledFeature",
            disabled
        )

        # -------------------------------------------------
        # BUTTON CONTENT
        # -------------------------------------------------

        button_layout = QHBoxLayout(
            button
        )

        button_layout.setContentsMargins(
            10,
            0,
            10,
            0
        )

        button_layout.setSpacing(
            12
        )

        # -------------------------------------------------
        # ICON
        # -------------------------------------------------

        icon_label = QLabel(
            icon
        )

        icon_label.setObjectName(
            "navIcon"
        )

        icon_label.setFixedWidth(
            22
        )

        icon_label.setAlignment(
            Qt.AlignCenter
        )

        # -------------------------------------------------
        # TEXT
        # -------------------------------------------------

        text_label = QLabel(
            text
        )

        text_label.setObjectName(
            "navText"
        )

        text_label.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred
        )

        button_layout.addWidget(
            icon_label
        )

        button_layout.addWidget(
            text_label
        )

        return button

    # =====================================================
    # TOP BAR
    # =====================================================

    def create_topbar(self):

        topbar = QFrame()

        topbar.setObjectName(
            "topbar"
        )

        topbar.setFixedHeight(
            72
        )

        layout = QHBoxLayout(
            topbar
        )

        layout.setContentsMargins(
            28,
            0,
            28,
            0
        )

        # =================================================
        # PAGE TITLE
        # =================================================

        self.page_title = QLabel(
            "Dashboard"
        )

        self.page_title.setObjectName(
            "pageTitle"
        )

        layout.addWidget(
            self.page_title
        )

        layout.addStretch()

        # =================================================
        # WORKSPACE STATUS
        # =================================================

        status = QLabel(
            "●  LOCAL WORKSPACE"
        )

        status.setObjectName(
            "topbarStatus"
        )

        layout.addWidget(
            status
        )

        return topbar

    # =====================================================
    # UPDATE PAGE TITLE
    # =====================================================

    def set_page_title(
        self,
        title
    ):

        self.page_title.setText(
            title
        )

    # =====================================================
    # NAVIGATION
    # =====================================================

    def open_dashboard(self):

        self.stack.setCurrentWidget(
            self.dashboard
        )

        self.set_page_title(
            "Dashboard"
        )

        self.set_active_button(
            self.dashboard_button
        )

    # -----------------------------------------------------

    def open_upload_page(self):

        self.stack.setCurrentWidget(
            self.upload_page
        )

        self.set_page_title(
            "Dataset"
        )

        self.set_active_button(
            self.dataset_button
        )

    # -----------------------------------------------------

    def open_analysis_page(self):

        self.stack.setCurrentWidget(
            self.analysis_page
        )

        self.set_page_title(
            "Dataset Analysis"
        )

        self.set_active_button(
            self.analysis_button
        )

        if self.current_dataset is not None:

            filename = Path(
                self.current_file_path
            ).name

            self.analysis_page.update_dataset(
                self.current_dataset,
                filename
            )

    # -----------------------------------------------------

    def open_ml_page(self):

        self.stack.setCurrentWidget(
            self.ml_page
        )

        self.set_page_title(
            "ML Intelligence"
        )

        self.set_active_button(
            self.ml_button
        )

        if self.current_dataset is None:

            self.ml_page.show_empty_state()

            return

        if not self.analysis_result:

            self.ml_page.show_empty_state()

            return

        self.ml_page.update_dataset()

    # -----------------------------------------------------

    def open_research_page(self):

        self.stack.setCurrentWidget(
            self.research_page
        )

        self.set_page_title(
            "Academic Research"
        )

        self.set_active_button(
            self.research_button
        )

        if self.current_dataset is None:

            self.research_page.show_empty_state()

            return

        if not self.analysis_result:

            self.research_page.show_empty_state()

            return

        self.research_page._update_status(
            self.current_dataset,
            self.analysis_result
        )

    # =====================================================
    # ACTIVE NAVIGATION
    # =====================================================

    def set_active_button(
        self,
        active_button
    ):

        buttons = [
            self.dashboard_button,
            self.dataset_button,
            self.analysis_button,
            self.ml_button,
            self.research_button,
        ]

        for button in buttons:

            button.setProperty(
                "active",
                button is active_button
            )

            style = button.style()

            style.unpolish(
                button
            )

            style.polish(
                button
            )

            button.update()

    # =====================================================
    # DATASET STATE
    # =====================================================

    def update_dataset_state(self):

        if self.current_dataset is None:
            return

        filename = Path(
            self.current_file_path
        ).name

        self.dashboard.update_dataset(
            self.current_dataset,
            filename
        )

        self.analysis_page.update_dataset(
            self.current_dataset,
            filename
        )

        # -------------------------------------------------
        # RESET ANALYSIS RESULT
        # -------------------------------------------------

        self.analysis_result = {}

        # -------------------------------------------------
        # RESET ML PAGE
        # -------------------------------------------------

        if hasattr(
            self,
            "ml_page"
        ):

            self.ml_page.show_empty_state()

        # -------------------------------------------------
        # RESET RESEARCH PAGE
        # -------------------------------------------------

        if hasattr(
            self,
            "research_page"
        ):

            self.research_page.show_empty_state()

    # =====================================================
    # DATASET ANALYSIS
    # =====================================================

    def run_dataset_analysis(self):

        dataframe = self.current_dataset

        if dataframe is None:

            raise RuntimeError(
                "No dataset loaded."
            )

        # =================================================
        # PROFILE
        # =================================================

        profile = self.profiler.profile(
            dataframe
        )

        # =================================================
        # STATISTICS
        # =================================================

        statistics = self.statistics.analyze(
            dataframe
        )

        # =================================================
        # MISSING VALUES
        # =================================================

        missing_values = (
            self.missing_analyzer.analyze(
                dataframe
            )
        )

        # =================================================
        # DUPLICATES
        # =================================================

        duplicates = (
            self.duplicate_analyzer.analyze(
                dataframe
            )
        )

        # =================================================
        # OUTLIERS
        # =================================================

        outliers = (
            self.outlier_analyzer.analyze(
                dataframe
            )
        )

        # =================================================
        # CORRELATION
        # =================================================

        correlations = (
            self.correlation_analyzer.analyze(
                dataframe
            )
        )

        # =================================================
        # FINGERPRINT
        # =================================================

        fingerprint = (
            self.fingerprint_generator.generate(
                dataframe
            )
        )

        # =================================================
        # STORE RESULT
        # =================================================

        self.analysis_result = {

            "profile": profile,

            "statistics": statistics,

            "missing_values": missing_values,

            "duplicates": duplicates,

            "outliers": outliers,

            "correlations": correlations,

            "fingerprint": fingerprint,
        }

        # =================================================
        # UPDATE ML PAGE
        # =================================================

        if hasattr(
            self,
            "ml_page"
        ):

            self.ml_page.update_dataset()

        # =================================================
        # UPDATE RESEARCH PAGE
        # =================================================

        if hasattr(
            self,
            "research_page"
        ):

            self.research_page.show_empty_state()

        return self.analysis_result