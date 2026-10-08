"""Evaluasi cepat analyzer dengan kerusakan sintetis yang diketahui."""

from __future__ import annotations

import hashlib
import pandas as pd

from app.analyzer.correlations import CorrelationAnalyzer
from app.analyzer.duplicates import DuplicateAnalyzer
from app.analyzer.missing_values import MissingValueAnalyzer
from app.analyzer.outliers import OutlierAnalyzer
from app.analyzer.profiler import DatasetProfiler


def run_quality_quick() -> dict:
    """Ukur hitungan masalah sintetis dan verifikasi analyzer tidak mengubah input."""
    frame = pd.DataFrame({
        "base": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "linear": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
        "with_missing": [1, 2, None, 4, 5, 6, 7, 8, 9, 10],
        "outlier": [10, 10, 10, 10, 10, 10, 10, 10, 10, 100],
    })
    frame.loc[1] = frame.loc[0]
    before = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
    profile = DatasetProfiler().profile(frame)
    missing = MissingValueAnalyzer().analyze(frame)
    duplicates = DuplicateAnalyzer().analyze(frame)
    outliers = OutlierAnalyzer().analyze(frame)
    correlations = CorrelationAnalyzer().analyze(frame)
    after = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()
    expected_missing = int(frame["with_missing"].isna().sum())
    actual_missing = next(item["missing_count"] for item in missing if item["column"] == "with_missing")
    expected_duplicates = int(frame.duplicated().sum())
    expected_outliers = int(((frame["outlier"] < frame["outlier"].quantile(.25) - 1.5 * (frame["outlier"].quantile(.75) - frame["outlier"].quantile(.25))) | (frame["outlier"] > frame["outlier"].quantile(.75) + 1.5 * (frame["outlier"].quantile(.75) - frame["outlier"].quantile(.25)))).sum())
    outlier_result = next(item for item in outliers if item["column"] == "outlier")
    return {
        "fixture": "known_injected_missing_duplicate_outlier_linear_correlation",
        "rows": len(frame),
        "metrics": {
            "missing_expected": expected_missing,
            "missing_detected": actual_missing,
            "missing_absolute_error": abs(expected_missing - actual_missing),
            "duplicates_expected": expected_duplicates,
            "duplicates_detected": duplicates["duplicate_count"],
            "outliers_expected_by_IQR": expected_outliers,
            "outliers_detected": outlier_result["outlier_count"],
            "pearson_linear": float(correlations.loc["base", "linear"]),
            "profile_rows": profile["rows"],
            "profile_columns": profile["columns"],
        },
        "baseline": {
            "no_missing_detector_error": expected_missing,
            "no_duplicate_detector_error": expected_duplicates,
            "no_outlier_detector_error": expected_outliers,
        },
        "input_hash_before": before,
        "input_hash_after": after,
        "input_unchanged": before == after,
    }


def assert_quality_quick(result: dict) -> None:
    """Pastikan benchmark sintetis sesuai gold yang ditanam di fixture."""
    metrics = result["metrics"]
    checks = {
        "missing count": metrics["missing_expected"] == metrics["missing_detected"],
        "duplicate count": metrics["duplicates_expected"] == metrics["duplicates_detected"],
        "outlier count": metrics["outliers_expected_by_IQR"] == metrics["outliers_detected"],
        "input immutability": result["input_unchanged"],
        "known linear correlation": abs(metrics["pearson_linear"] - 1.0) < 1e-12,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError("Evaluasi sintetis gagal: " + ", ".join(failed))

