from __future__ import annotations

from typing import Any, Dict, List

from app.ml.method_info import get_method_info


class MethodRecommender:
    """
    Merekomendasikan metode machine learning berdasarkan:
    - ML task
    - karakteristik dataset
    - target
    - jumlah baris/kolom
    - jumlah fitur numerik
    - metadata dari Method Knowledge Base
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
        }

    def recommend(self, ml_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate method recommendations.

        Parameters
        ----------
        ml_result:
            Hasil dari MLTaskDetector.detect()

        Returns
        -------
        dict
            Hasil rekomendasi lengkap.
        """

        primary_task = ml_result.get("primary_task", {})

        task = primary_task.get("task", "unknown_task")
        label = primary_task.get("label", task)

        target = primary_task.get("target")

        dataset = ml_result.get("dataset", {})

        rows = dataset.get("rows", 0)
        columns = dataset.get("columns", 0)
        numeric_features = dataset.get("numeric_features", 0)

        # Support format alternatif
        if not rows:
            rows = ml_result.get("rows", 0)

        if not columns:
            columns = ml_result.get("columns", 0)

        if not numeric_features:
            numeric_features = ml_result.get("numeric_features", 0)

        method_scores = self.base_scores.get(task, {})

        recommendations: List[Dict[str, Any]] = []

        for method_id, base_score in method_scores.items():

            info = get_method_info(method_id)

            if info is None:
                continue

            score = float(base_score)
            adjustments: Dict[str, float] = {}
            reasons: List[str] = []

            # -------------------------------------------------
            # DATASET SIZE
            # -------------------------------------------------

            if rows > 5000:
                if info.get("large_data"):
                    adjustments["large_dataset"] = 3
                    score += 3
                    reasons.append(
                        "Metode sesuai untuk dataset berukuran besar."
                    )

            elif rows < 1000:
                if info.get("small_data"):
                    adjustments["small_dataset"] = 3
                    score += 3
                    reasons.append(
                        "Metode sesuai untuk dataset berukuran kecil hingga menengah."
                    )

            # -------------------------------------------------
            # NUMBER OF FEATURES
            # -------------------------------------------------

            if numeric_features >= 10:
                if info.get("high_dimensional"):
                    adjustments["many_numeric_features"] = 3
                    score += 3
                    reasons.append(
                        "Dataset memiliki banyak fitur numerik."
                    )

            # -------------------------------------------------
            # SCALING
            # -------------------------------------------------

            if info.get("scaling_required"):
                if rows < 1000:
                    adjustments["scaling_required"] = -1
                    score -= 1
                    reasons.append(
                        "Feature scaling diperlukan sebelum training."
                    )

            # -------------------------------------------------
            # INTERPRETABILITY
            # -------------------------------------------------

            if info.get("interpretability") == "high":
                reasons.append(
                    "Mudah diinterpretasikan."
                )

            # -------------------------------------------------
            # TARGET
            # -------------------------------------------------

            if target:
                reasons.insert(
                    0,
                    f"Target yang terdeteksi adalah '{target}'."
                )

            # -------------------------------------------------
            # TASK
            # -------------------------------------------------

            reasons.insert(
                0,
                f"Metode sesuai dengan task {label}."
            )

            if rows:
                reasons.append(
                    f"Dataset memiliki sekitar {rows} baris."
                )

            if columns:
                reasons.append(
                    f"Dataset memiliki {columns} kolom."
                )

            # -------------------------------------------------
            # SCORE LIMIT
            # -------------------------------------------------

            score = max(0.0, min(100.0, score))

            recommendations.append(
                {
                    "method_id": method_id,
                    "method": info["name"],
                    "category": info["category"],
                    "score": round(score, 2),
                    "status": "RECOMMENDATION",

                    # Knowledge Base
                    "description": info["description"],
                    "strengths": info["strengths"],
                    "limitations": info["limitations"],
                    "preprocessing": info["preprocessing"],
                    "interpretability": info["interpretability"],
                    "scaling_required": info["scaling_required"],
                    "nonlinear": info["nonlinear"],
                    "small_data": info["small_data"],
                    "large_data": info["large_data"],

                    # Explanation
                    "reasons": reasons,
                    "adjustments": adjustments,

                    # Full KB object
                    "knowledge_base": info,
                }
            )

        # Ranking
        recommendations.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return {
            "status": "RECOMMENDATION",

            "primary_task": primary_task,

            "recommendations": recommendations,

            "recommendation_count": len(recommendations),

            "message": (
                "Metode merupakan rekomendasi berdasarkan jenis task, "
                "karakteristik dataset, dan Method Knowledge Base. "
                "Evaluasi empiris tetap diperlukan untuk menentukan "
                "metode dengan performa terbaik."
            ),
        }