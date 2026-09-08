from __future__ import annotations

from typing import Any, Dict

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QMessageBox,
    QProgressBar,
    QHeaderView,
)

from app.research.intelligence import ResearchIntelligenceEngine
from app.ml.task_detector import MLTaskDetector


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
            f"DATASET USAGE CONFIDENCE\n"
            f"{paper.get('dataset_usage_confidence', '-')}"
        )

        evidence = paper.get(
            "dataset_usage_evidence",
            []
        )

        if evidence:

            lines.append(
                "EVIDENCE\n"
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
                "LIMITATIONS\n"
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
                f"ABSTRACT\n{abstract}"
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
    def _format_section(
        data
    ):

        if not data:

            return "No data available."

        lines = []

        if isinstance(
            data,
            dict
        ):

            for key, value in data.items():

                formatted_key = (
                    str(key)
                    .replace("_", " ")
                    .title()
                )

                if isinstance(
                    value,
                    list
                ):

                    lines.append(
                        f"{formatted_key}:"
                    )

                    for item in value:

                        lines.append(
                            f"  • {item}"
                        )

                elif isinstance(
                    value,
                    dict
                ):

                    lines.append(
                        f"{formatted_key}:"
                    )

                    for sub_key, sub_value in (
                        value.items()
                    ):

                        lines.append(
                            f"  • "
                            f"{str(sub_key).replace('_', ' ').title()}: "
                            f"{sub_value}"
                        )

                else:

                    lines.append(
                        f"{formatted_key}: {value}"
                    )

        elif isinstance(
            data,
            list
        ):

            for item in data:

                lines.append(
                    f"• {item}"
                )

        else:

            lines.append(
                str(data)
            )

        return "\n".join(
            lines
        )

    @staticmethod
    def _format_gap(
        gaps
    ):

        if not gaps:

            return "No potential research gaps detected."

        if isinstance(
            gaps,
            dict
        ):

            lines = []

            summary = gaps.get(
                "summary"
            )

            if summary:

                lines.append(
                    "SUMMARY"
                )

                lines.append(
                    ResearchPage._format_section(
                        summary
                    )
                )

                lines.append("")

            gap_items = (
                gaps.get(
                    "gaps"
                )
                or gaps.get(
                    "potential_gaps"
                )
                or gaps.get(
                    "items"
                )
            )

            if isinstance(
                gap_items,
                list
            ):

                lines.append(
                    "POTENTIAL RESEARCH DIRECTIONS"
                )

                for item in gap_items:

                    if isinstance(
                        item,
                        dict
                    ):

                        title = (
                            item.get(
                                "title"
                            )
                            or item.get(
                                "gap"
                            )
                            or item.get(
                                "description"
                            )
                            or str(item)
                        )

                        lines.append(
                            f"• {title}"
                        )

                    else:

                        lines.append(
                            f"• {item}"
                        )

                return "\n".join(
                    lines
                )

            return ResearchPage._format_section(
                gaps
            )

        return str(gaps)