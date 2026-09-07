import pandas as pd


class OutlierAnalyzer:
    """Detect numerical outliers using the IQR method."""

    def analyze(self, dataframe: pd.DataFrame) -> list[dict]:
        results = []

        numeric_columns = dataframe.select_dtypes(
            include=["number"]
        ).columns

        for column in numeric_columns:
            series = dataframe[column].dropna()

            if series.empty:
                continue

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)

            iqr = q3 - q1

            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)

            outliers = series[
                (series < lower_bound)
                | (series > upper_bound)
            ]

            outlier_count = len(outliers)

            results.append(
                {
                    "column": str(column),
                    "outlier_count": int(outlier_count),
                    "outlier_percentage": (
                        outlier_count / len(series) * 100
                    ),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound),
                }
            )

        return results