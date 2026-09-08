"""
Dataset Preview Component
Modern table view untuk menampilkan dataset preview dengan metadata
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QTableView,
    QAbstractItemView,
)
from PySide6.QtGui import QStandardItemModel, QStandardItem, QFont


class DatasetPreviewCard(QFrame):
    """
    Kartu preview dataset modern dengan tabel dan metadata
    
    Menampilkan:
    - Nama file
    - Jumlah baris dan kolom
    - Tabel data (10-20 baris pertama)
    - Footer dengan info pagination
    """

    def __init__(self, dataframe=None, filename=""):
        super().__init__()

        self.dataframe = dataframe
        self.filename = filename
        self.max_preview_rows = 15

        self.setObjectName("datasetPreviewCard")
        self.build_ui()

    def build_ui(self):
        """Build UI untuk preview card"""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # =================================================
        # HEADER
        # =================================================

        header = QFrame()
        header.setObjectName("datasetPreviewHeader")

        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(16, 14, 16, 14)
        header_layout.setSpacing(6)

        # Title (filename)
        title = QLabel(self.filename if self.filename else "Dataset Preview")
        title.setObjectName("datasetPreviewTitle")
        header_layout.addWidget(title)

        # Metadata (rows, columns)
        meta_text = ""
        if self.dataframe is not None:
            rows = len(self.dataframe)
            cols = len(self.dataframe.columns)
            meta_text = f"{rows:,} rows  •  {cols} columns"

        meta = QLabel(meta_text)
        meta.setObjectName("datasetPreviewMeta")
        header_layout.addWidget(meta)

        layout.addWidget(header)

        # =================================================
        # TABLE
        # =================================================

        self.table = QTableView()
        self.table.setObjectName("datasetPreviewTable")
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(300)
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.verticalHeader().setVisible(False)

        # Populate table
        if self.dataframe is not None:
            self.populate_table()

        layout.addWidget(self.table)

        # =================================================
        # FOOTER
        # =================================================

        footer = QFrame()
        footer.setObjectName("datasetPreviewFooter")

        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(16, 10, 16, 10)
        footer_layout.setSpacing(0)

        # Footer text
        footer_text = ""
        if self.dataframe is not None:
            total_rows = len(self.dataframe)
            preview_rows = min(self.max_preview_rows, total_rows)
            footer_text = f"Showing {preview_rows} of {total_rows:,} rows"

        footer_label = QLabel(footer_text)
        footer_label.setObjectName("datasetPreviewFooter")

        footer_layout.addWidget(footer_label)
        footer_layout.addStretch()

        layout.addWidget(footer)

    def populate_table(self):
        """Populate table dengan data dari dataframe"""

        if self.dataframe is None or self.dataframe.empty:
            return

        # Get preview data (first N rows)
        preview_df = self.dataframe.head(self.max_preview_rows)

        # Create model
        model = QStandardItemModel()

        # Set column headers
        headers = list(self.dataframe.columns)
        model.setHorizontalHeaderLabels(headers)

        # Add rows
        for idx, row in preview_df.iterrows():
            items = []
            for col in self.dataframe.columns:
                value = str(row[col])
                item = QStandardItem(value)
                # Make items non-editable
                item.setEditable(False)
                items.append(item)
            model.appendRow(items)

        # Set model ke table
        self.table.setModel(model)

        # Adjust column widths
        self.table.resizeColumnsToContents()

        # Set uniform row height
        self.table.verticalHeader().setDefaultSectionSize(32)

    def update_data(self, dataframe, filename=""):
        """Update preview dengan dataset baru"""

        self.dataframe = dataframe
        self.filename = filename

        # Clear existing layout except header
        # (Re-create UI untuk kesederhanaan)
        self.setLayout(None)
        self.build_ui()


class DatasetPreviewPage(QWidget):
    """
    Halaman lengkap untuk dataset preview
    Dapat diintegrasikan ke upload page atau analysis page
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.dataframe = None
        self.filename = ""

        self.build_ui()

    def build_ui(self):
        """Build halaman preview"""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(20)

        # Title
        title = QLabel("Dataset Preview")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        # Preview card
        self.preview_card = DatasetPreviewCard(
            self.dataframe,
            self.filename
        )
        layout.addWidget(self.preview_card)

        layout.addStretch()

    def update_preview(self, dataframe, filename=""):
        """Update preview dengan dataset baru"""

        self.dataframe = dataframe
        self.filename = filename

        self.preview_card.update_data(dataframe, filename)

    def show_preview(self, dataframe, filename):
        """Tampilkan preview untuk dataset tertentu"""

        self.update_preview(dataframe, filename)