import pandas as pd


class DatasetProfiler:
    """Generate general information about a dataset."""

    def profile(self, dataframe: pd.DataFrame) -> dict:
        numeric_columns = dataframe.select_dtypes(
            include=["number"]
        ).columns.tolist()

        categorical_columns = dataframe.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        datetime_columns = dataframe.select_dtypes(
            include=["datetime"]
        ).columns.tolist()

        return {
            "rows": len(dataframe),
            "columns": len(dataframe.columns),
            "memory_usage": int(
                dataframe.memory_usage(deep=True).sum()
            ),
            "duplicate_rows": int(
                dataframe.duplicated().sum()
            ),
            "missing_values": int(
                dataframe.isna().sum().sum()
            ),
            "numeric_columns": len(numeric_columns),
            "categorical_columns": len(categorical_columns),
            "datetime_columns": len(datetime_columns),
        }