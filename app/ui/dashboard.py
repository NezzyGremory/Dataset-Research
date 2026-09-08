from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QPushButton,
)


# =========================================================
# STAT CARD
# =========================================================


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


# =========================================================
# QUICK ACTION CARD
# =========================================================


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


# =========================================================
# PIPELINE STAGE
# =========================================================


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


# =========================================================
# DASHBOARD PAGE
# =========================================================


class DashboardPage(QWidget):

    def __init__(
        self,
        main_window,
    ):
        super().__init__()

        self.main_window = main_window

        self.build_ui()

    # =====================================================
    # BUILD UI
    # =====================================================

    def build_ui(self):

        layout = QVBoxLayout(
            self
        )

        layout.setContentsMargins(
            28,
            22,
            28,
            22,
        )

        layout.setSpacing(
            16
        )

        # =================================================
        # HERO
        # =================================================

        hero = QFrame()

        hero.setObjectName(
            "heroCard"
        )

        hero.setMinimumHeight(
            190
        )

        hero_layout = QHBoxLayout(
            hero
        )

        hero_layout.setContentsMargins(
            24,
            20,
            24,
            20,
        )

        hero_layout.setSpacing(
            20
        )

        # -------------------------------------------------
        # HERO TEXT
        # -------------------------------------------------

        hero_text = QVBoxLayout()

        hero_text.setSpacing(
            5
        )

        eyebrow = QLabel(
            "DATA SCIENCE  •  RESEARCH"
        )

        eyebrow.setObjectName(
            "heroEyebrow"
        )

        hero_title = QLabel(
            "Turn your dataset into\n"
            "research insights."
        )

        hero_title.setObjectName(
            "heroTitle"
        )

        hero_title.setWordWrap(
            True
        )

        hero_description = QLabel(
            "Analyze data, discover suitable ML methods, "
            "and explore academic literature from one workspace."
        )

        hero_description.setObjectName(
            "heroDescription"
        )

        hero_description.setWordWrap(
            True
        )

        hero_text.addWidget(
            eyebrow
        )

        hero_text.addWidget(
            hero_title
        )

        hero_text.addWidget(
            hero_description
        )

        hero_text.addStretch()

        # -------------------------------------------------
        # HERO BUTTON
        # -------------------------------------------------

        start_button = QPushButton(
            "+  Upload Dataset"
        )

        start_button.setObjectName(
            "heroButton"
        )

        start_button.setFixedHeight(
            38
        )

        start_button.setMaximumWidth(
            170
        )

        start_button.setCursor(
            Qt.PointingHandCursor
        )

        start_button.clicked.connect(
            self.main_window.open_upload_page
        )

        hero_text.addWidget(
            start_button
        )

        hero_layout.addLayout(
            hero_text,
            1
        )

        # -------------------------------------------------
        # HERO WORKFLOW
        # -------------------------------------------------

        visual = QFrame()

        visual.setObjectName(
            "heroVisual"
        )

        visual.setFixedWidth(
            270
        )

        visual_layout = QVBoxLayout(
            visual
        )

        visual_layout.setContentsMargins(
            16,
            13,
            16,
            13,
        )

        visual_layout.setSpacing(
            4
        )

        visual_title = QLabel(
            "RESEARCH WORKFLOW"
        )

        visual_title.setObjectName(
            "heroVisualTitle"
        )

        visual_layout.addWidget(
            visual_title
        )

        workflow_items = [
            ("01", "Dataset"),
            ("02", "Analysis"),
            ("03", "ML Intelligence"),
            ("04", "Academic Research"),
        ]

        for number, title in workflow_items:

            item = QHBoxLayout()

            item.setSpacing(
                9
            )

            number_label = QLabel(
                number
            )

            number_label.setObjectName(
                "heroWorkflowNumber"
            )

            number_label.setFixedWidth(
                22
            )

            title_label = QLabel(
                title
            )

            title_label.setObjectName(
                "heroWorkflowTitle"
            )

            item.addWidget(
                number_label
            )

            item.addWidget(
                title_label
            )

            item.addStretch()

            visual_layout.addLayout(
                item
            )

        visual_layout.addStretch()

        hero_layout.addWidget(
            visual
        )

        layout.addWidget(
            hero
        )

        # =================================================
        # QUICK ACTIONS HEADER
        # =================================================

        quick_header = QHBoxLayout()

        quick_title = QLabel(
            "Quick Actions"
        )

        quick_title.setObjectName(
            "sectionTitle"
        )

        quick_description = QLabel(
            "Start your research workflow"
        )

        quick_description.setObjectName(
            "sectionDescription"
        )

        quick_header.addWidget(
            quick_title
        )

        quick_header.addSpacing(
            8
        )

        quick_header.addWidget(
            quick_description
        )

        quick_header.addStretch()

        layout.addLayout(
            quick_header
        )

        # =================================================
        # QUICK ACTIONS
        # =================================================

        actions_layout = QHBoxLayout()

        actions_layout.setSpacing(
            12
        )

        actions_layout.addWidget(
            QuickActionCard(
                "↑",
                "Upload Dataset",
                "Import a dataset and start a new project.",
                self.main_window.open_upload_page,
            )
        )

        actions_layout.addWidget(
            QuickActionCard(
                "◇",
                "Analyze Dataset",
                "Explore statistics, missing values and outliers.",
                self.main_window.open_analysis_page,
            )
        )

        actions_layout.addWidget(
            QuickActionCard(
                "✦",
                "ML Intelligence",
                "Discover suitable machine learning methods.",
                self.main_window.open_ml_page,
            )
        )

        actions_layout.addWidget(
            QuickActionCard(
                "◎",
                "Academic Research",
                "Find related papers and research opportunities.",
                self.main_window.open_research_page,
            )
        )

        layout.addLayout(
            actions_layout
        )

        # =================================================
        # CURRENT DATASET HEADER
        # =================================================

        dataset_header = QHBoxLayout()

        dataset_title = QLabel(
            "Current Dataset"
        )

        dataset_title.setObjectName(
            "sectionTitle"
        )

        self.dataset_status_badge = QLabel(
            "NO DATASET"
        )

        self.dataset_status_badge.setObjectName(
            "datasetStatusBadge"
        )

        dataset_header.addWidget(
            dataset_title
        )

        dataset_header.addStretch()

        dataset_header.addWidget(
            self.dataset_status_badge
        )

        layout.addLayout(
            dataset_header
        )

        # =================================================
        # CURRENT DATASET CARD
        # =================================================

        dataset_card = QFrame()

        dataset_card.setObjectName(
            "currentDatasetCard"
        )

        dataset_card.setMinimumHeight(
            92
        )

        dataset_layout = QHBoxLayout(
            dataset_card
        )

        dataset_layout.setContentsMargins(
            18,
            14,
            18,
            14,
        )

        dataset_layout.setSpacing(
            16
        )

        # -------------------------------------------------
        # DATASET INFO
        # -------------------------------------------------

        dataset_info = QVBoxLayout()

        dataset_info.setSpacing(
            3
        )

        self.dataset_name_label = QLabel(
            "No dataset loaded"
        )

        self.dataset_name_label.setObjectName(
            "datasetName"
        )

        self.dataset_description_label = QLabel(
            "Upload a dataset to begin your research workflow."
        )

        self.dataset_description_label.setObjectName(
            "datasetDescription"
        )

        self.dataset_description_label.setWordWrap(
            True
        )

        dataset_info.addWidget(
            self.dataset_name_label
        )

        dataset_info.addWidget(
            self.dataset_description_label
        )

        dataset_info.addStretch()

        dataset_layout.addLayout(
            dataset_info,
            1
        )

        # -------------------------------------------------
        # DATASET STATS
        # -------------------------------------------------

        stats_layout = QHBoxLayout()

        stats_layout.setSpacing(
            8
        )

        self.rows_card = StatCard(
            "ROWS",
            "—",
            "Records",
        )

        self.columns_card = StatCard(
            "COLUMNS",
            "—",
            "Features",
        )

        stats_layout.addWidget(
            self.rows_card
        )

        stats_layout.addWidget(
            self.columns_card
        )

        dataset_layout.addLayout(
            stats_layout
        )

        layout.addWidget(
            dataset_card
        )

        # =================================================
        # PIPELINE HEADER
        # =================================================

        pipeline_header = QHBoxLayout()

        pipeline_title = QLabel(
            "Research Pipeline"
        )

        pipeline_title.setObjectName(
            "sectionTitle"
        )

        pipeline_description = QLabel(
            "Track your research progress"
        )

        pipeline_description.setObjectName(
            "sectionDescription"
        )

        pipeline_header.addWidget(
            pipeline_title
        )

        pipeline_header.addSpacing(
            8
        )

        pipeline_header.addWidget(
            pipeline_description
        )

        pipeline_header.addStretch()

        layout.addLayout(
            pipeline_header
        )

        # =================================================
        # PIPELINE
        # =================================================

        pipeline_card = QFrame()

        pipeline_card.setObjectName(
            "pipelineCard"
        )

        pipeline_layout = QHBoxLayout(
            pipeline_card
        )

        pipeline_layout.setContentsMargins(
            10,
            10,
            10,
            10,
        )

        pipeline_layout.setSpacing(
            4
        )

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

        for index, (
            title,
            status,
        ) in enumerate(
            pipeline_data
        ):

            stage = PipelineStage(
                index + 1,
                title,
                status.upper(),
            )

            self.pipeline_stages.append(
                stage
            )

            pipeline_layout.addWidget(
                stage,
                1
            )

            if index < len(
                pipeline_data
            ) - 1:

                connector = QLabel(
                    "›"
                )

                connector.setObjectName(
                    "pipelineConnector"
                )

                connector.setAlignment(
                    Qt.AlignCenter
                )

                pipeline_layout.addWidget(
                    connector
                )

        layout.addWidget(
            pipeline_card
        )

    # =====================================================
    # UPDATE DATASET
    # =====================================================

    def update_dataset(
        self,
        dataframe,
        filename,
    ):

        if dataframe is None:
            return

        # -------------------------------------------------
        # NAME
        # -------------------------------------------------

        self.dataset_name_label.setText(
            filename
        )

        # -------------------------------------------------
        # DESCRIPTION
        # -------------------------------------------------

        self.dataset_description_label.setText(
            "Dataset loaded successfully. "
            "Run analysis to generate dataset intelligence."
        )

        # -------------------------------------------------
        # STATUS
        # -------------------------------------------------

        self.dataset_status_badge.setText(
            "● DATASET LOADED"
        )

        self.dataset_status_badge.setProperty(
            "state",
            "success"
        )

        style = self.dataset_status_badge.style()

        style.unpolish(
            self.dataset_status_badge
        )

        style.polish(
            self.dataset_status_badge
        )

        self.dataset_status_badge.update()

        # -------------------------------------------------
        # STATISTICS
        # -------------------------------------------------

        self.rows_card.set_value(
            f"{len(dataframe):,}"
        )

        self.columns_card.set_value(
            f"{len(dataframe.columns):,}"
        )

        # -------------------------------------------------
        # PIPELINE
        # -------------------------------------------------

        if hasattr(
            self,
            "pipeline_stages"
        ):

            if len(
                self.pipeline_stages
            ) >= 1:

                self.pipeline_stages[0].set_status(
                    "READY",
                    "success",
                )

            if len(
                self.pipeline_stages
            ) >= 2:

                self.pipeline_stages[1].set_status(
                    "NEXT",
                    "active",
                )

            for stage in self.pipeline_stages[2:4]:

                stage.set_status(
                    "WAITING",
                    "waiting",
                )

            for stage in self.pipeline_stages[4:]:

                stage.set_status(
                    "V2",
                    "future",
                )

    # =====================================================
    # RESET DATASET
    # =====================================================

    def reset_dataset(self):

        self.dataset_name_label.setText(
            "No dataset loaded"
        )

        self.dataset_description_label.setText(
            "Upload a dataset to begin your research workflow."
        )

        self.dataset_status_badge.setText(
            "NO DATASET"
        )

        self.dataset_status_badge.setProperty(
            "state",
            "waiting"
        )

        style = self.dataset_status_badge.style()

        style.unpolish(
            self.dataset_status_badge
        )

        style.polish(
            self.dataset_status_badge
        )

        self.dataset_status_badge.update()

        self.rows_card.set_value(
            "—"
        )

        self.columns_card.set_value(
            "—"
        )

        if hasattr(
            self,
            "pipeline_stages"
        ):

            for index, stage in enumerate(
                self.pipeline_stages
            ):

                if index == 0:

                    stage.set_status(
                        "UPLOAD",
                        "active",
                    )

                elif index < 4:

                    stage.set_status(
                        "WAITING",
                        "waiting",
                    )

                else:

                    stage.set_status(
                        "V2",
                        "future",
                    )