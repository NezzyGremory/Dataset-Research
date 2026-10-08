"""Uji ketahanan input memakai data temporer; tidak menyentuh data proyek."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from app.analyzer.loader import DatasetLoader
from app.core.exceptions import DatasetLoadError
from app.ml.task_detector import MLTaskDetector
from app.ml.target_detector import MLTargetDetector


def run_robustness(quick: bool = False) -> dict:
    """Catat graceful failure dan crash pada input CSV adversarial kecil."""
    cases = {
        "one_column": ("label\na\nb\nc\n", "utf-8"),
        "semicolon_delimiter": ("age;label\n18;yes\n19;no\n", "utf-8"),
        "cp1252_text": ("city;label\nMünchen;yes\nZürich;no\n", "cp1252"),
        "empty": ("", "utf-8"),
        "header_only": ("a,b\n", "utf-8"),
        "id_and_target": ("record_id,Unnamed: 0,target\n1,0,a\n2,1,b\n3,2,a\n", "utf-8"),
        "anonymous_columns": ("a,b,c\n1,4,yes\n2,5,no\n3,6,yes\n", "utf-8"),
        "shuffled_columns": ("target,feature_b,feature_a\nyes,4,1\nno,5,2\nyes,6,3\n", "utf-8"),
        "introduced_missing": ("feature_a,feature_b,target\n1,,yes\n,5,no\n3,6,yes\n", "utf-8"),
        "target_removed": ("feature_a,feature_b\n1,4\n2,5\n3,6\n", "utf-8"),
        "reduced_sample": ("feature_a,feature_b,target\n1,4,yes\n", "utf-8"),
        "added_id": ("row_id,feature_a,feature_b,target\n1,1,4,yes\n2,2,5,no\n3,3,6,yes\n", "utf-8"),
    }
    if quick:
        cases = {key: cases[key] for key in ("one_column", "semicolon_delimiter", "empty")}
    results = []
    loader = DatasetLoader()
    for name, (content, encoding) in cases.items():
        try:
            with tempfile.TemporaryDirectory(prefix="dataset_research_eval_") as temp:
                path = Path(temp) / f"{name}.csv"
                path.write_bytes(content.encode(encoding))
                before = path.read_bytes()
                dataframe = loader.load_csv(str(path))
                task_result = MLTaskDetector().detect({}, dataframe=dataframe)
                target_candidates = MLTargetDetector().detect(dataframe=dataframe)
                results.append({"case": name, "loader": "PASS", "rows": len(dataframe),
                                "columns": len(dataframe.columns), "task_status": task_result.get("status"),
                                "primary_task": (task_result.get("primary_task") or {}).get("task"),
                                "target_candidate_count": len(target_candidates),
                                "graceful": True, "source_file_unchanged": path.read_bytes() == before})
        except (DatasetLoadError, ValueError) as error:
            results.append({"case": name, "loader": "GRACEFUL_ERROR", "error_type": type(error).__name__,
                            "message": str(error), "graceful": True})
        except Exception as error:
            results.append({"case": name, "loader": "CRASH", "error_type": type(error).__name__,
                            "message": str(error), "graceful": False})
    return {"cases": results, "graceful_count": sum(row["graceful"] for row in results),
            "crash_count": sum(not row["graceful"] for row in results)}
