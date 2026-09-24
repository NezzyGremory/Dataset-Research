from __future__ import annotations

from typing import Any, Dict, List

from app.ml.method_info import get_method_info


class MethodRecommender:
    """Recommend ML methods from detected task and dataset characteristics."""

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
        }

    def recommend(self, ml_result: Dict[str, Any] | None) -> Dict[str, Any]:
        """Generate safe recommendations even when no primary task is detected."""
        if not isinstance(ml_result, dict):
            ml_result = {}

        primary_task = ml_result.get("primary_task")
        if not isinstance(primary_task, dict):
            primary_task = self._fallback_primary_task(ml_result)

        task = str(primary_task.get("task") or "unknown_task")
        label = primary_task.get("label") or task
        target = primary_task.get("target")

        dataset = ml_result.get("dataset")
        if not isinstance(dataset, dict):
            dataset = {}

        rows = self._safe_number(dataset.get("rows"), ml_result.get("rows", 0))
        columns = self._safe_number(dataset.get("columns"), ml_result.get("columns", 0))
        numeric_features = self._safe_number(
            dataset.get("numeric_features"),
            ml_result.get("numeric_features", 0),
        )

        method_scores = self.base_scores.get(task, {})
        recommendations: List[Dict[str, Any]] = []

        for method_id, base_score in method_scores.items():
            info = get_method_info(method_id)
            if not isinstance(info, dict):
                continue

            score = float(base_score)
            adjustments: Dict[str, float] = {}
            reasons: List[str] = []

            if rows > 5000 and info.get("large_data"):
                adjustments["large_dataset"] = 3
                score += 3
                reasons.append("Metode sesuai untuk dataset berukuran besar.")
            elif 0 < rows < 1000 and info.get("small_data"):
                adjustments["small_dataset"] = 3
                score += 3
                reasons.append("Metode sesuai untuk dataset kecil hingga menengah.")

            if numeric_features >= 10 and info.get("high_dimensional"):
                adjustments["many_numeric_features"] = 3
                score += 3
                reasons.append("Dataset memiliki banyak fitur numerik.")

            if info.get("scaling_required"):
                adjustments["scaling_required"] = -1
                score -= 1
                reasons.append("Feature scaling diperlukan sebelum training.")

            if info.get("interpretability") == "high":
                reasons.append("Mudah diinterpretasikan.")

            reasons.insert(0, f"Metode sesuai dengan task {label}.")
            if target:
                reasons.insert(1, f"Target yang terdeteksi adalah '{target}'.")
            if rows:
                reasons.append(f"Dataset memiliki sekitar {int(rows)} baris.")
            if columns:
                reasons.append(f"Dataset memiliki {int(columns)} kolom.")

            score = max(0.0, min(100.0, score))

            recommendations.append({
                "method_id": method_id,
                "method": info.get("name", method_id),
                "category": info.get("category", task),
                "score": round(score, 2),
                "status": "RECOMMENDATION",
                "description": info.get("description", ""),
                "strengths": info.get("strengths", []),
                "limitations": info.get("limitations", []),
                "preprocessing": info.get("preprocessing", []),
                "interpretability": info.get("interpretability", "unknown"),
                "scaling_required": bool(info.get("scaling_required", False)),
                "nonlinear": bool(info.get("nonlinear", False)),
                "small_data": bool(info.get("small_data", False)),
                "large_data": bool(info.get("large_data", False)),
                "reasons": reasons,
                "adjustments": adjustments,
                "knowledge_base": info,
            })

        recommendations.sort(key=lambda item: item.get("score", 0), reverse=True)

        if not recommendations:
            message = (
                "Belum ada metode yang dapat direkomendasikan secara yakin. "
                "Sistem tidak menemukan task supervised yang cukup kuat atau "
                "knowledge base belum memiliki metode untuk task tersebut."
            )
        else:
            message = (
                "Metode merupakan rekomendasi berdasarkan jenis task, "
                "karakteristik dataset, dan Method Knowledge Base. "
                "Evaluasi empiris tetap diperlukan untuk menentukan "
                "metode dengan performa terbaik."
            )

        return {
            "status": "RECOMMENDATION",
            "primary_task": primary_task if primary_task else None,
            "recommendations": recommendations,
            "recommendation_count": len(recommendations),
            "message": message,
        }

    def _fallback_primary_task(self, ml_result: Dict[str, Any]) -> Dict[str, Any] | None:
        """Recover a usable primary task from the task list if needed."""
        tasks = ml_result.get("tasks")
        if not isinstance(tasks, list):
            return None

        valid = [task for task in tasks if isinstance(task, dict) and task.get("task")]
        if not valid:
            return None

        supervised = [
            task for task in valid
            if task.get("target") is not None
            and task.get("task") in {
                "binary_classification", "multiclass_classification", "regression"
            }
        ]
        pool = supervised or [
            task for task in valid if task.get("task") == "time_series_forecasting"
        ] or valid

        return max(pool, key=lambda item: self._safe_number(item.get("score"), 0))

    @staticmethod
    def _safe_number(value: Any, fallback: Any = 0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            try:
                return float(fallback)
            except (TypeError, ValueError):
                return 0.0
