import pandas as pd


class DatasetStatistics:
    """Calculate statistics for dataset columns."""

    def analyze(self, dataframe: pd.DataFrame) -> list[dict]:
        results = []

        total_rows = len(dataframe)

        for column in dataframe.columns:
            series = dataframe[column]

            missing_count = int(series.isna().sum())
            unique_count = int(series.nunique(dropna=True))

            result = {
                "column": str(column),
                "dtype": str(series.dtype),
                "missing_count": missing_count,
                "missing_percentage": (
                    missing_count / total_rows * 100
                    if total_rows > 0
                    else 0
                ),
                "unique_count": unique_count,
                "unique_percentage": (
                    unique_count / total_rows * 100
                    if total_rows > 0
                    else 0
                ),
            }

            if pd.api.types.is_numeric_dtype(series):
                result.update(
                    {
                        "mean": float(series.mean())
                        if not series.dropna().empty
                        else None,
                        "median": float(series.median())
                        if not series.dropna().empty
                        else None,
                        "std": float(series.std())
                        if not series.dropna().empty
                        else None,
                        "min": float(series.min())
                        if not series.dropna().empty
                        else None,
                        "max": float(series.max())
                        if not series.dropna().empty
                        else None,
                        "q1": float(series.quantile(0.25))
                        if not series.dropna().empty
                        else None,
                        "q3": float(series.quantile(0.75))
                        if not series.dropna().empty
                        else None,
                        "skewness": float(series.skew())
                        if not series.dropna().empty
                        else None,
                    }
                )

            elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
                value_counts = series.value_counts(dropna=True)

                result.update(
                    {
                        "category_count": unique_count,
                        "top_values": [
                            {
                                "value": str(value),
                                "frequency": int(frequency),
                            }
                            for value, frequency in value_counts.head(5).items()
                        ],
                    }
                )

            results.append(result)

        return results