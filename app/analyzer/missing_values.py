import pandas as pd


MISSING_TEXT_VALUES = {
    "", "na", "n/a", "n.a.", "nan", "null", "none", "nil",
    "undefined", "missing", "missing value", "-", "?", "#na", "#n/a",
}
INVISIBLE_WHITESPACE = "\u200b\u200c\u200d\ufeff"
INVISIBLE_WHITESPACE_TRANSLATION = str.maketrans("", "", INVISIBLE_WHITESPACE)


def _normalize_missing_cell(value):
    if not isinstance(value, str):
        return value
    text = value.translate(INVISIBLE_WHITESPACE_TRANSLATION).strip()
    return pd.NA if text.casefold() in MISSING_TEXT_VALUES else text


def normalize_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where blank strings and common null markers are missing."""
    normalized = dataframe.copy()
    for column in normalized.columns:
        series = normalized[column]
        is_categorical = isinstance(series.dtype, pd.CategoricalDtype)
        if not (
            is_categorical
            or pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
        ):
            continue
        # Excel/CSV cells may contain ordinary or invisible whitespace. Keep
        # non-text values intact in mixed-type object columns.
        source = series.astype(object) if is_categorical else series
        normalized[column] = source.map(_normalize_missing_cell)
    return normalized


def drop_missing_rows(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop every row containing an actual or recognized textual missing value."""
    if dataframe.empty or dataframe.shape[1] == 0:
        raise ValueError("Dataset tidak memiliki baris atau kolom untuk dibersihkan.")

    normalized = normalize_missing_values(dataframe)
    missing_before = int(normalized.isna().sum().sum())
    rows_before = len(normalized)
    rows_with_missing = int(normalized.isna().any(axis=1).sum())
    cleaned = normalized.dropna(axis=0, how="any").copy()
    missing_after = int(cleaned.isna().sum().sum())

    if missing_after:
        raise ValueError(
            f"Drop missing belum tuntas: masih ada {missing_after:,} nilai kosong."
        )
    if cleaned.empty:
        raise ValueError(
            "Tidak ada baris yang seluruh kolomnya terisi. Dataset tidak diubah."
        )

    return cleaned, {
        "rows_before": rows_before,
        "rows_after": len(cleaned),
        "rows_dropped": rows_before - len(cleaned),
        "rows_with_missing": rows_with_missing,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "how": "any",
    }


def impute_missing_values(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Impute usable columns and remove columns with no observed values."""
    if dataframe.empty or dataframe.shape[1] == 0:
        raise ValueError("Dataset tidak memiliki baris atau kolom untuk dibersihkan.")

    cleaned = normalize_missing_values(dataframe)
    missing_before = int(cleaned.isna().sum().sum())
    empty_columns = [column for column in cleaned.columns if cleaned[column].isna().all()]
    if len(empty_columns) == len(cleaned.columns):
        raise ValueError(
            "Semua kolom berisi nilai kosong. Tidak ada nilai yang bisa digunakan "
            "untuk menghitung imputasi."
        )

    if empty_columns:
        cleaned = cleaned.drop(columns=empty_columns)

    imputed_columns = []
    for column in cleaned.columns:
        series = cleaned[column]
        if not series.isna().any():
            continue

        if (
            pd.api.types.is_numeric_dtype(series.dtype)
            and not pd.api.types.is_bool_dtype(series.dtype)
        ):
            fill_value = series.median(skipna=True)
            if pd.isna(fill_value):
                raise ValueError(
                    f"Nilai pengganti untuk kolom numerik '{column}' tidak dapat dihitung."
                )

            if pd.api.types.is_integer_dtype(series.dtype) and float(fill_value) % 1:
                cleaned[column] = series.astype("Float64").fillna(fill_value)
            else:
                cleaned[column] = series.fillna(fill_value)
            method = "median"
        else:
            modes = series.mode(dropna=True)
            if modes.empty:
                raise ValueError(
                    f"Nilai pengganti untuk kolom '{column}' tidak dapat dihitung."
                )
            fill_value = modes.iloc[0]
            if pd.api.types.is_bool_dtype(series.dtype):
                fill_value = bool(fill_value)
            cleaned[column] = series.fillna(fill_value)
            method = "mode"

        imputed_columns.append(
            {"column": str(column), "method": method, "fill_value": str(fill_value)}
        )

    remaining_missing = int(cleaned.isna().sum().sum())
    if remaining_missing:
        raise ValueError(
            f"Pembersihan belum tuntas: masih ada {remaining_missing:,} nilai kosong."
        )

    return cleaned, {
        "missing_before": missing_before,
        "missing_after": remaining_missing,
        "imputed_columns": imputed_columns,
        "dropped_empty_columns": [str(column) for column in empty_columns],
    }


class MissingValueAnalyzer:
    """Analyze missing values in a dataset."""

    def analyze(self, dataframe: pd.DataFrame) -> list[dict]:
        normalized = normalize_missing_values(dataframe)
        total_rows = len(normalized)
        results = []

        for column in normalized.columns:
            missing_count = int(normalized[column].isna().sum())

            results.append(
                {
                    "column": str(column),
                    "missing_count": missing_count,
                    "missing_percentage": (
                        missing_count / total_rows * 100
                        if total_rows > 0
                        else 0
                    ),
                }
            )

        return results
