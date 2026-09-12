from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from app.core.constants import MAX_DATASET_SIZE_MB
from app.core.exceptions import DatasetLoadError


class DatasetLoader:
    """
    Load and validate CSV-like tabular datasets.

    The loader automatically detects common delimiters instead of assuming
    commas. This allows datasets exported by different tools to work without
    dataset-specific rules.
    """

    COMMON_DELIMITERS = [",", ";", "\t", "|", ":"]
    ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin1"]

    def load_csv(self, file_path: str) -> pd.DataFrame:
        path = Path(file_path)

        if not path.exists():
            raise DatasetLoadError("File dataset tidak ditemukan.")

        if path.suffix.lower() != ".csv":
            raise DatasetLoadError("Format dataset harus CSV.")

        file_size = path.stat().st_size
        file_size_mb = file_size / (1024 * 1024)

        if file_size_mb > MAX_DATASET_SIZE_MB:
            raise DatasetLoadError(
                f"Ukuran dataset terlalu besar. "
                f"Maksimal {MAX_DATASET_SIZE_MB} MB."
            )

        if file_size == 0:
            raise DatasetLoadError("File CSV kosong.")

        try:
            sample_text = self._read_sample(path)

            delimiter = self._detect_delimiter(sample_text)

            dataframe = self._read_dataframe(
                path=path,
                delimiter=delimiter,
            )

            # Safety fallback:
            # Jika pandas membaca hanya 1 kolom, tetapi sample jelas memiliki
            # delimiter lain, coba kandidat delimiter lain secara otomatis.
            if len(dataframe.columns) <= 1:
                alternative = self._best_alternative_delimiter(
                    sample_text,
                    current=delimiter,
                )

                if alternative is not None:
                    alternative_df = self._read_dataframe(
                        path=path,
                        delimiter=alternative,
                    )

                    if len(alternative_df.columns) > len(
                        dataframe.columns
                    ):
                        dataframe = alternative_df

        except UnicodeDecodeError as error:
            raise DatasetLoadError(
                "Encoding CSV tidak dapat dibaca."
            ) from error
        except pd.errors.ParserError as error:
            raise DatasetLoadError(
                "Format CSV tidak valid atau rusak."
            ) from error
        except DatasetLoadError:
            raise
        except Exception as error:
            raise DatasetLoadError(
                f"Gagal membaca dataset: {error}"
            ) from error

        if dataframe.empty:
            raise DatasetLoadError("Dataset tidak memiliki data.")

        if len(dataframe.columns) == 0:
            raise DatasetLoadError("Dataset tidak memiliki kolom.")

        # Clean up accidental whitespace/BOM in column names while preserving
        # the actual data values.
        dataframe.columns = [
            str(column).replace("\ufeff", "").strip()
            for column in dataframe.columns
        ]

        return dataframe

    # =============================================================
    # SAMPLE / ENCODING
    # =============================================================

    def _read_sample(self, path: Path, sample_size: int = 64 * 1024) -> str:
        """
        Read a small text sample using common CSV encodings.
        """

        raw = path.read_bytes()[:sample_size]

        last_error: Exception | None = None

        for encoding in self.ENCODINGS:
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError as error:
                last_error = error

        if last_error is not None:
            raise last_error

        raise DatasetLoadError(
            "Encoding CSV tidak dapat dibaca."
        )

    def _read_dataframe(
        self,
        path: Path,
        delimiter: str | None,
    ) -> pd.DataFrame:
        """
        Read the dataframe using a detected delimiter.

        The encoding loop makes the loader tolerant of common exported CSV
        encodings without changing the dataset content.
        """

        last_decode_error: Exception | None = None

        for encoding in self.ENCODINGS:
            try:
                return pd.read_csv(
                    path,
                    sep=delimiter,
                    engine="python",
                    encoding=encoding,
                    skipinitialspace=True,
                    on_bad_lines="error",
                )
            except UnicodeDecodeError as error:
                last_decode_error = error
                continue

        if last_decode_error is not None:
            raise last_decode_error

        raise DatasetLoadError(
            "Encoding CSV tidak dapat dibaca."
        )

    # =============================================================
    # DELIMITER DETECTION
    # =============================================================

    def _detect_delimiter(self, sample_text: str) -> str:
        """
        Detect a delimiter using csv.Sniffer first, then fall back to
        frequency/consistency scoring over common delimiters.
        """

        # Remove completely empty lines before sniffing.
        sample = "\n".join(
            line
            for line in sample_text.splitlines()
            if line.strip()
        )

        if not sample:
            raise DatasetLoadError("CSV tidak memiliki isi yang dapat dibaca.")

        # First choice: Python's CSV dialect detector.
        try:
            dialect = csv.Sniffer().sniff(
                sample[:65536],
                delimiters="".join(self.COMMON_DELIMITERS),
            )

            detected = dialect.delimiter

            if detected in self.COMMON_DELIMITERS:
                return detected
        except (csv.Error, TypeError, ValueError):
            pass

        # Fallback: score delimiter consistency across non-empty lines.
        lines = sample.splitlines()
        candidate_scores: list[tuple[float, str]] = []

        for delimiter in self.COMMON_DELIMITERS:
            counts = [
                line.count(delimiter)
                for line in lines[:30]
            ]

            positive = [
                count
                for count in counts
                if count > 0
            ]

            if not positive:
                continue

            # A real delimiter usually occurs repeatedly with a reasonably
            # stable number of fields across rows.
            average = sum(positive) / len(positive)
            consistency = 1.0 - (
                max(positive) - min(positive)
            ) / max(max(positive), 1)

            coverage = len(positive) / max(len(counts), 1)

            score = (
                average * 2.0
                + consistency * 3.0
                + coverage * 4.0
            )

            candidate_scores.append(
                (score, delimiter)
            )

        if candidate_scores:
            candidate_scores.sort(
                key=lambda item: item[0],
                reverse=True,
            )
            return candidate_scores[0][1]

        # A true single-column CSV is valid, so comma is the safest default.
        return ","

    def _best_alternative_delimiter(
        self,
        sample_text: str,
        current: str,
    ) -> str | None:
        """
        When the first pass produces one column, find another delimiter that
        clearly creates more fields.
        """

        lines = [
            line
            for line in sample_text.splitlines()
            if line.strip()
        ]

        if not lines:
            return None

        candidates: list[tuple[int, float, str]] = []

        for delimiter in self.COMMON_DELIMITERS:
            if delimiter == current:
                continue

            counts = [
                line.count(delimiter)
                for line in lines[:30]
            ]

            positive = [
                count
                for count in counts
                if count > 0
            ]

            if not positive:
                continue

            average = sum(positive) / len(positive)
            coverage = len(positive) / max(len(counts), 1)

            candidates.append(
                (
                    int(max(positive)),
                    average * coverage,
                    delimiter,
                )
            )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )

        return candidates[0][2]
