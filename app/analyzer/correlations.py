import pandas as pd


class CorrelationAnalyzer:
    """Calculate Pearson correlation between numeric columns."""

    def analyze(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        numeric_data = dataframe.select_dtypes(
            include=["number"]
        )
        boolean_columns = [
            column for column in numeric_data.columns
            if pd.api.types.is_bool_dtype(numeric_data[column].dtype)
        ]
        if boolean_columns:
            numeric_data = numeric_data.drop(columns=boolean_columns)

        if numeric_data.shape[1] < 2:
            return pd.DataFrame()

        return numeric_data.corr(method="pearson")
