from __future__ import annotations

from typing import Any, Dict, List

from app.ml.method_info import get_method_info


class MethodRecommender:
    """
    Recommends ML methods from the detected task and dataset characteristics.

    The recommender is defensive by design:
    - primary_task may be None
    - task list may be empty
    - dataset metadata may be missing
    - knowledge-base entries may be incomplete

    It never assumes that an ML task was successfully detected.
    """

    def __init__(self) -> None:
        self.base_scores = {
            "binary_classification": {
                "random_forest_classifier": 92,
                "logistic_regression": 85,
                "decision_tree_classifier": 82,
                "svm_classifier": 81,
                "knn_classifier": 75,
            },
            "multiclass_classification": {
                "random_forest_classifier": 92,
                "decision_tree_classifier": 82,
                "svm_classifier": 81,
                "knn_classifier": 75,
                "logistic_regression": 78,
            },
            "regression": {
                "random_forest_regressor": 92,
                "gradient_boosting_regressor": 90,
                "linear_regression": 84,
                "decision_tree_regressor": 80,
            },
            "clustering": {
                "kmeans": 90,
                "dbscan": 82,
                "agglomerative_clustering": 78,
            },
            "anomaly_detection": {
                "isolation_forest": 92,
                "local_outlier_factor": 82,
                "one_class_svm": 79,
            },
            "time_series_forecasting": {},
        }

    def recommend(self, ml_result: Dict[str, Any] | None) -> Dict[str, Any]:
        """
        Generate method recommendations without ever crashing on None.

        Important:
        `primary_task` is allowed to be None when the detector cannot infer
        a supervised task with enough confidence.
        """

        if not isinstance(ml_result, dict):
            ml_result = {}

        primary_task = ml_result.get("primary_task")

        # ---------------------------------------------------------
        # RECOVER PRIMARY TASK FROM TASK LIST
        # ---------------------------------------------------------
        if not isinstance(primary_task, dict):
            primary_task = self._fallback_primary_task(ml_result)

        # ---------------------------------------------------------
        # NORMALIZE PRIMARY TASK
        # ---------------------------------------------------------
        if isinstance(primary_task, dict):
            task = str(primary_task.get("task") or "").strip()
            label = str(
                primary_task.get("label")
                or task
                or "Unknown Task"
            )
            target = primary_task.get("target")
        else:
            primary_task = None
            task = ""
            label = "Unknown Task"
            target = None

        # ---------------------------------------------------------
        # DATASET METADATA
        # ---------------------------------------------------------
        dataset = ml_result.get("dataset")
        if not isinstance(dataset, dict):
            dataset = {}

        rows = self._safe_number(
            dataset.get("rows"),
            ml_result.get("rows", 0),
        )

        columns = self._safe_number(
            dataset.get("columns"),
            ml_result.get("columns", 0),
        )

        numeric_features = self._safe_number(
            dataset.get("numeric_features"),
            ml_result.get("numeric_features", 0),
        )

        # ---------------------------------------------------------
        # UNKNOWN / MISSING TASK
        # ---------------------------------------------------------
        if not task:
            return {
                "status": "NO_CONFIDENT_TASK",
                "primary_task": None,
                "recommendations": [],
                "recommendation_count": 0,
                "message": (
                    "ML task belum dapat ditentukan dengan confidence "
                    "yang cukup dari struktur dataset. Sistem tidak "
                    "memaksakan pemilihan metode."
                ),
            }

        # ---------------------------------------------------------
        # METHOD KNOWLEDGE BASE
        # ---------------------------------------------------------
        method_scores = self.base_scores.get(task, {})
        recommendations: List[Dict[str, Any]] = []

        for method_id, base_score in method_scores.items():
            try:
                info = get_method_info(method_id)
            except Exception:
                info = None

            if not isinstance(info, dict):
                continue

            score = float(base_score)
            adjustments: Dict[str, float] = {}
            reasons: List[str] = []

            # -----------------------------------------------------
            # DATASET SIZE
            # -----------------------------------------------------
            if rows > 5000 and info.get("large_data"):
                adjustments["large_dataset"] = 3
                score += 3
                reasons.append(
                    "Metode sesuai untuk dataset berukuran besar."
                )

            elif 0 < rows < 1000 and info.get("small_data"):
                adjustments["small_dataset"] = 3
                score += 3
                reasons.append(
                    "Metode sesuai untuk dataset kecil hingga menengah."
                )

            # -----------------------------------------------------
            # NUMBER OF FEATURES
            # -----------------------------------------------------
            if numeric_features >= 10 and info.get("high_dimensional"):
                adjustments["many_numeric_features"] = 3
                score += 3
                reasons.append(
                    "Dataset memiliki banyak fitur numerik."
                )

            # -----------------------------------------------------
            # SCALING
            # -----------------------------------------------------
            if info.get("scaling_required"):
                adjustments["scaling_required"] = -1
                score -= 1
                reasons.append(
                    "Feature scaling diperlukan sebelum training."
                )

            # -----------------------------------------------------
            # INTERPRETABILITY
            # -----------------------------------------------------
            if info.get("interpretability") == "high":
                reasons.append(
                    "Mudah diinterpretasikan."
                )

            # -----------------------------------------------------
            # TASK / TARGET
            # -----------------------------------------------------
            reasons.insert(
                0,
                f"Metode sesuai dengan task {label}.",
            )

            if target:
                reasons.insert(
                    1,
                    f"Target yang terdeteksi adalah '{target}'.",
                )

            if rows:
                reasons.append(
                    f"Dataset memiliki sekitar {int(rows)} baris."
                )

            if columns:
                reasons.append(
                    f"Dataset memiliki {int(columns)} kolom."
                )

            score = max(0.0, min(100.0, score))

            recommendations.append(
                {
                    "method_id": method_id,
                    "method": info.get("name", method_id),
                    "category": info.get("category", task),
                    "score": round(score, 2),
                    "status": "RECOMMENDATION",
                    "description": info.get("description", ""),
                    "strengths": info.get("strengths", []),
                    "limitations": info.get("limitations", []),
                    "preprocessing": info.get("preprocessing", []),
                    "interpretability": info.get(
                        "interpretability",
                        "unknown",
                    ),
                    "scaling_required": bool(
                        info.get("scaling_required", False)
                    ),
                    "nonlinear": bool(
                        info.get("nonlinear", False)
                    ),
                    "small_data": bool(
                        info.get("small_data", False)
                    ),
                    "large_data": bool(
                        info.get("large_data", False)
                    ),
                    "reasons": reasons,
                    "adjustments": adjustments,
                    "knowledge_base": info,
                }
            )

        recommendations.sort(
            key=lambda item: item.get("score", 0),
            reverse=True,
        )

        if recommendations:
            message = (
                "Metode merupakan rekomendasi berdasarkan jenis task, "
                "karakteristik dataset, dan Method Knowledge Base. "
                "Evaluasi empiris tetap diperlukan untuk menentukan "
                "metode dengan performa terbaik."
            )
            status = "RECOMMENDATION"
        else:
            message = (
                f"Task '{task}' berhasil terdeteksi, tetapi belum ada "
                "metode yang tersedia di Method Knowledge Base."
            )
            status = "NO_METHODS"

        return {
            "status": status,
            "primary_task": primary_task,
            "recommendations": recommendations,
            "recommendation_count": len(recommendations),
            "message": message,
        }

    def _fallback_primary_task(
        self,
        ml_result: Dict[str, Any],
    ) -> Dict[str, Any] | None:
        """
        Recover the best task from ml_result["tasks"].

        Priority:
        1. supervised task with a target
        2. time-series task
        3. highest-scoring remaining task
        """

        tasks = ml_result.get("tasks")

        if not isinstance(tasks, list):
            return None

        valid = [
            task
            for task in tasks
            if isinstance(task, dict)
            and task.get("task")
        ]

        if not valid:
            return None

        supervised = [
            task
            for task in valid
            if task.get("target") is not None
            and task.get("task") in {
                "binary_classification",
                "multiclass_classification",
                "regression",
            }
        ]

        time_series = [
            task
            for task in valid
            if task.get("task") == "time_series_forecasting"
        ]

        pool = supervised or time_series or valid

        return max(
            pool,
            key=lambda item: self._safe_number(
                item.get("score"),
                item.get("confidence", 0),
            ),
        )

    @staticmethod
    def _safe_number(
        value: Any,
        fallback: Any = 0,
    ) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            try:
                return float(fallback)
            except (TypeError, ValueError):
                return 0.0
