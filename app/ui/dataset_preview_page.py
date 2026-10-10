"""Spreadsheet-like, read-only preview of the active pandas dataset."""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QFrame,
    QHeaderView,
    QLabel,
    QTableView,
    QVBoxLayout,
    QWidget,
)


class DataFrameTableModel(QAbstractTableModel):
    """Expose a DataFrame to Qt without copying every cell into widgets."""

    def __init__(self, dataframe: pd.DataFrame | None = None, parent=None):
        super().__init__(parent)
        self._dataframe = dataframe if dataframe is not None else pd.DataFrame()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._dataframe.index)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._dataframe.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        value = self._dataframe.iat[index.row(), index.column()]
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal and 0 <= section < len(self._dataframe.columns):
            return str(self._dataframe.columns[section])
        if orientation == Qt.Vertical and 0 <= section < len(self._dataframe.index):
            return str(section + 1)
        return None

    def set_dataframe(self, dataframe: pd.DataFrame | None):
        self.beginResetModel()
        self._dataframe = dataframe if dataframe is not None else pd.DataFrame()
        self.endResetModel()


class DatasetTableView(QTableView):
    """Read-only grid with spreadsheet-style selection and copy support."""

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Copy):
            indexes = self.selectedIndexes()
            if indexes:
                rows = range(min(i.row() for i in indexes), max(i.row() for i in indexes) + 1)
                columns = range(min(i.column() for i in indexes), max(i.column() for i in indexes) + 1)
                selected = {(i.row(), i.column()) for i in indexes}
                lines = []
                for row in rows:
                    values = [
                        str(self.model().data(self.model().index(row, column), Qt.DisplayRole) or "")
                        if (row, column) in selected else ""
                        for column in columns
                    ]
                    lines.append("\t".join(values))
                QApplication.clipboard().setText("\n".join(lines))
                return
        super().keyPressEvent(event)


class DatasetPreviewPage(QWidget):
    """Scrollable table showing every row and column in the active dataset."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dataset_name = ""
        self._dataframe: pd.DataFrame | None = None

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 22)
        root.setSpacing(14)

        header = QFrame()
        header.setObjectName("datasetPreviewHeader")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(18, 16, 18, 16)
        header_layout.setSpacing(5)

        title = QLabel("Dataset preview")
        title.setObjectName("pageTitle")
        header_layout.addWidget(title)

        self.summary = QLabel("Pilih atau unggah dataset untuk melihat isinya.")
        self.summary.setObjectName("pageSubtitle")
        self.summary.setWordWrap(True)
        header_layout.addWidget(self.summary)
        root.addWidget(header)

        self.table = DatasetTableView()
        self.table.setObjectName("datasetPreviewTable")
        self.model = DataFrameTableModel(parent=self.table)
        self.table.setModel(self.model)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setWordWrap(False)
        self.table.setCornerButtonEnabled(False)
        self.table.setSortingEnabled(False)
        self.table.verticalHeader().setDefaultSectionSize(28)
        self.table.verticalHeader().setMinimumWidth(54)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setDefaultSectionSize(150)
        self.table.horizontalHeader().setMinimumSectionSize(80)
        root.addWidget(self.table, 1)

        self.empty_state = QLabel("Belum ada dataset yang dimuat.")
        self.empty_state.setObjectName("emptyState")
        self.empty_state.setAlignment(Qt.AlignCenter)
        self.empty_state.setMinimumHeight(100)
        self.empty_state.hide()
        root.addWidget(self.empty_state, 1)
        self.update_dataset(None)

    def update_dataset(self, dataframe: pd.DataFrame | None, filename: str = ""):
        self._dataframe = dataframe
        self._dataset_name = filename or "Dataset"
        self.model.set_dataframe(dataframe)
        if dataframe is None:
            self.summary.setText("Pilih atau unggah dataset untuk melihat isinya.")
            self.table.hide()
            self.empty_state.show()
            return

        self.empty_state.hide()
        self.table.show()
        rows, columns = dataframe.shape
        self.summary.setText(
            f"{self._dataset_name}  ·  {rows:,} baris  ·  {columns:,} kolom"
            "  ·  Hanya lihat"
        )
        if columns <= 12 and rows <= 500:
            self.table.resizeColumnsToContents()
            for column in range(columns):
                self.table.setColumnWidth(column, min(self.table.columnWidth(column), 280))
