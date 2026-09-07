from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # =================================================
        # WINDOW CONFIGURATION
        # =================================================

        self.setWindowTitle("Dataset Research")
        self.resize(1100, 700)

        # Dataset yang sedang digunakan
        self.current_dataset = None

        # =================================================
        # DATASET LOADER
        # =================================================

        self.loader = DatasetLoader()

        # =================================================
        # ANALYZER
        # =================================================

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

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        # =================================================
        # TITLE
        # =================================================

        self.title_label = QLabel(
            "Dataset Research"
        )

        self.title_label.setStyleSheet(
            """
            QLabel {
                font-size: 28px;
                font-weight: bold;
                padding: 10px;
            }
            """
        )

        layout.addWidget(self.title_label)

        # =================================================
        # DESCRIPTION
        # =================================================

        self.description_label = QLabel(
            "AI-Assisted Academic Dataset Research"
        )

        self.description_label.setStyleSheet(
            """
            QLabel {
                font-size: 14px;
                padding: 5px 10px 15px 10px;
            }
            """
        )

        layout.addWidget(self.description_label)

        # =================================================
        # UPLOAD BUTTON
        # =================================================

        self.upload_button = QPushButton(
            "Upload Dataset"
        )

        self.upload_button.setMinimumHeight(40)

        self.upload_button.clicked.connect(
            self.upload_dataset
        )

        layout.addWidget(self.upload_button)

        # =================================================
        # ANALYZE BUTTON
        # =================================================

        self.analyze_button = QPushButton(
            "Analyze Dataset"
        )

        self.analyze_button.setMinimumHeight(40)

        # Belum aktif sebelum dataset diupload
        self.analyze_button.setEnabled(False)

        self.analyze_button.clicked.connect(
            self.analyze_dataset
        )

        layout.addWidget(self.analyze_button)

        # =================================================
        # DATASET INFO
        # =================================================

        self.dataset_info_label = QLabel(
            "No dataset loaded."
        )

        self.dataset_info_label.setStyleSheet(
            """
            QLabel {
                font-size: 14px;
                padding: 10px;
            }
            """
        )

        layout.addWidget(
            self.dataset_info_label
        )

        # =================================================
        # PREVIEW LABEL
        # =================================================

        self.preview_label = QLabel(
            "Dataset Preview"
        )

        self.preview_label.setStyleSheet(
            """
            QLabel {
                font-size: 18px;
                font-weight: bold;
                padding: 10px 0px;
            }
            """
        )

        layout.addWidget(
            self.preview_label
        )

        # =================================================
        # TABLE
        # =================================================

        self.table = QTableWidget()

        self.table.setEditTriggers(
            QTableWidget.NoEditTriggers
        )

        self.table.setSelectionBehavior(
            QTableWidget.SelectRows
        )

        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )

        self.table.horizontalHeader().setStretchLastSection(
            True
        )

        layout.addWidget(
            self.table
        )

    # =====================================================
    # UPLOAD DATASET
    # =====================================================

    def upload_dataset(self):
        """
        Membuka file dialog untuk memilih dataset CSV.
        """

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset",
            "",
            "CSV Files (*.csv)",
        )

        if not file_path:
            return

        try:
            # Load dataset menggunakan DatasetLoader
            dataframe = self.loader.load_csv(
                file_path
            )

            # Simpan dataset
            self.current_dataset = dataframe

            # Tampilkan informasi dataset
            self.dataset_info_label.setText(
                f"Dataset loaded successfully\n"
                f"Rows: {len(dataframe):,}\n"
                f"Columns: {len(dataframe.columns):,}\n"
                f"File: {file_path}"
            )

            # Aktifkan tombol analysis
            self.analyze_button.setEnabled(
                True
            )

            # Tampilkan preview
            self.show_preview(
                dataframe
            )

            QMessageBox.information(
                self,
                "Dataset Loaded",
                "Dataset berhasil dimuat.",
            )

        except Exception as error:

            QMessageBox.critical(
                self,
                "Upload Error",
                f"Gagal memuat dataset.\n\n"
                f"{error}",
            )

    # =====================================================
    # SHOW DATASET PREVIEW
    # =====================================================

    def show_preview(self, dataframe):
        """
        Menampilkan maksimal 20 baris pertama dataset.
        """

        preview = dataframe.head(20)

        # Bersihkan tabel
        self.table.clear()

        # Jumlah baris
        self.table.setRowCount(
            len(preview)
        )

        # Jumlah kolom
        self.table.setColumnCount(
            len(preview.columns)
        )

        # Header kolom
        self.table.setHorizontalHeaderLabels(
            [
                str(column)
                for column in preview.columns
            ]
        )

        # Isi tabel
        for row_index, (_, row) in enumerate(
            preview.iterrows()
        ):

            for column_index, value in enumerate(
                row
            ):

                if pd_is_missing(value):
                    text = ""
                else:
                    text = str(value)

                item = QTableWidgetItem(
                    text
                )

                self.table.setItem(
                    row_index,
                    column_index,
                    item,
                )

    # =====================================================
    # ANALYZE DATASET
    # =====================================================

    def analyze_dataset(self):
        """
        Menjalankan seluruh analyzer pada dataset.
        """

        if self.current_dataset is None:

            QMessageBox.warning(
                self,
                "No Dataset",
                "Silakan upload dataset terlebih dahulu.",
            )

            return

        try:

            dataframe = self.current_dataset

            # =================================================
            # 1. DATASET PROFILE
            # =================================================

            profile = self.profiler.profile(
                dataframe
            )

            # =================================================
            # 2. COLUMN STATISTICS
            # =================================================

            statistics = self.statistics.analyze(
                dataframe
            )

            # =================================================
            # 3. MISSING VALUES
            # =================================================

            missing_values = (
                self.missing_analyzer.analyze(
                    dataframe
                )
            )

            # =================================================
            # 4. DUPLICATES
            # =================================================

            duplicates = (
                self.duplicate_analyzer.analyze(
                    dataframe
                )
            )

            # =================================================
            # 5. OUTLIERS
            # =================================================

            outliers = (
                self.outlier_analyzer.analyze(
                    dataframe
                )
            )

            # =================================================
            # 6. CORRELATIONS
            # =================================================

            correlations = (
                self.correlation_analyzer.analyze(
                    dataframe
                )
            )

            # =================================================
            # 7. DATASET FINGERPRINT
            # =================================================

            fingerprint_result = (
                self.fingerprint_generator.generate(
                    dataframe
                )
            )

            # =================================================
            # PRINT RESULTS
            # =================================================

            print("\n")
            print("=" * 70)
            print("DATASET PROFILE")
            print("=" * 70)

            print(profile)

            # -------------------------------------------------

            print("\n")
            print("=" * 70)
            print("COLUMN STATISTICS")
            print("=" * 70)

            for item in statistics:
                print(item)

            # -------------------------------------------------

            print("\n")
            print("=" * 70)
            print("MISSING VALUES")
            print("=" * 70)

            for item in missing_values:
                print(item)

            # -------------------------------------------------

            print("\n")
            print("=" * 70)
            print("DUPLICATES")
            print("=" * 70)

            print(duplicates)

            # -------------------------------------------------

            print("\n")
            print("=" * 70)
            print("OUTLIERS")
            print("=" * 70)

            for item in outliers:
                print(item)

            # -------------------------------------------------

            print("\n")
            print("=" * 70)
            print("CORRELATION")
            print("=" * 70)

            print(correlations)

            # -------------------------------------------------

            print("\n")
            print("=" * 70)
            print("DATASET FINGERPRINT")
            print("=" * 70)

            print("\nRepresentation:")

            print(
                fingerprint_result[
                    "representation"
                ]
            )

            print(
                "\nSHA-256 Fingerprint:"
            )

            print(
                fingerprint_result[
                    "fingerprint"
                ]
            )

            # -------------------------------------------------

            print("\n")
            print("=" * 70)
            print("ANALYSIS COMPLETE")
            print("=" * 70)

            # =================================================
            # SUCCESS MESSAGE
            # =================================================

            QMessageBox.information(
                self,
                "Analysis Complete",
                "Dataset berhasil dianalisis.\n\n"
                "Hasil analysis sementara dapat "
                "dilihat di terminal.",
            )

        except Exception as error:

            QMessageBox.critical(
                self,
                "Analysis Error",
                f"Terjadi error saat melakukan analysis.\n\n"
                f"{error}",
            )


# =========================================================
# HELPER FUNCTION
# =========================================================

def pd_is_missing(value):
    """
    Mengecek apakah sebuah nilai merupakan missing value.

    Dibuat sebagai helper agar preview tabel tidak
    menampilkan 'nan' atau 'NaT'.
    """

    try:
        import pandas as pd

        return bool(
            pd.isna(value)
        )

    except Exception:
        return False