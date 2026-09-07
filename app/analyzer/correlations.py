import pandas as pd


class CorrelationAnalyzer:
    """Calculate Pearson correlation between numeric columns."""

    def analyze(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        numeric_data = dataframe.select_dtypes(
            include=["number"]
        )

        if numeric_data.shape[1] < 2:
            return pd.DataFrame()

        return numeric_data.corr(method="pearson")