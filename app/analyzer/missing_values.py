import pandas as pd


class MissingValueAnalyzer:
    """Analyze missing values in a dataset."""

    def analyze(self, dataframe: pd.DataFrame) -> list[dict]:
        total_rows = len(dataframe)
        results = []

        for column in dataframe.columns:
            missing_count = int(dataframe[column].isna().sum())

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