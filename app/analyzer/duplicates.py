import pandas as pd


class DuplicateAnalyzer:
    """Analyze duplicate rows."""

    def analyze(self, dataframe: pd.DataFrame) -> dict:
        duplicate_count = int(dataframe.duplicated().sum())

        total_rows = len(dataframe)

        percentage = (
            duplicate_count / total_rows * 100
            if total_rows > 0
            else 0
        )

        return {
            "duplicate_count": duplicate_count,
            "duplicate_percentage": percentage,
        }