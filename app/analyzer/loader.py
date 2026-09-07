from pathlib import Path

import pandas as pd

from app.core.constants import MAX_DATASET_SIZE_MB
from app.core.exceptions import DatasetLoadError


class DatasetLoader:
    """Load and validate CSV datasets."""

    def load_csv(self, file_path: str) -> pd.DataFrame:
        path = Path(file_path)

        if not path.exists():
            raise DatasetLoadError("File dataset tidak ditemukan.")

        if path.suffix.lower() != ".csv":
            raise DatasetLoadError("Format dataset harus CSV.")

        file_size_mb = path.stat().st_size / (1024 * 1024)

        if file_size_mb > MAX_DATASET_SIZE_MB:
            raise DatasetLoadError(
                f"Ukuran dataset terlalu besar. "
                f"Maksimal {MAX_DATASET_SIZE_MB} MB."
            )

        if path.stat().st_size == 0:
            raise DatasetLoadError("File CSV kosong.")

        try:
            dataframe = pd.read_csv(path)
        except UnicodeDecodeError as error:
            raise DatasetLoadError(
                "Encoding CSV tidak dapat dibaca."
            ) from error
        except pd.errors.ParserError as error:
            raise DatasetLoadError(
                "Format CSV tidak valid atau rusak."
            ) from error
        except Exception as error:
            raise DatasetLoadError(
                f"Gagal membaca dataset: {error}"
            ) from error

        if dataframe.empty:
            raise DatasetLoadError("Dataset tidak memiliki data.")

        if len(dataframe.columns) == 0:
            raise DatasetLoadError("Dataset tidak memiliki kolom.")

        return dataframe