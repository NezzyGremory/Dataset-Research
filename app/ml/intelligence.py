from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from app.ml.evaluation import MLEvaluator
from app.ml.method_recommender import MethodRecommender
from app.ml.task_detector import MLTaskDetector


class MLIntelligenceEngine:
    """
    Orchestrator utama untuk ML Intelligence.

    Alur:
        Fingerprint
            ↓
        Task Detection
            ↓
        Method Recommendation
            ↓
        Knowledge Base
            ↓
        Empirical Evaluation
            ↓
        ML Intelligence Result
    """

    def __init__(
        self,
        evaluator: Optional[MLEvaluator] = None,
        recommender: Optional[MethodRecommender] = None,
        task_detector: Optional[MLTaskDetector] = None,
    ) -> None:

        self.task_detector = task_detector or MLTaskDetector()
        self.recommender = recommender or MethodRecommender()
        self.evaluator = evaluator or MLEvaluator()

    def analyze(
        self,
        dataframe: pd.DataFrame,
        fingerprint: Dict[str, Any],
        evaluate_models: bool = True,
    ) -> Dict[str, Any]:
        """
        Menjalankan seluruh proses ML Intelligence.

        Parameters
        ----------
        dataframe:
            Dataset asli dalam bentuk DataFrame.

        fingerprint:
            Fingerprint dataset dari DatasetFingerprint.

        evaluate_models:
            Jika False, hanya melakukan detection + recommendation.
        """

        # =====================================================
        # 1. TASK DETECTION
        # =====================================================

        try:
            task_result = self.task_detector.detect(fingerprint)
        except Exception as exc:
            return self._error(
                f"Gagal melakukan ML task detection: {exc}"
            )

        primary_task = task_result.get("primary_task", {})

        task = primary_task.get(
            "task",
            "unknown_task",
        )

        target = primary_task.get("target")

        # =====================================================
        # 2. DATASET CHARACTERISTICS
        # =====================================================

        dataset_info = self._extract_dataset_info(
            dataframe,
            fingerprint,
        )

        # =====================================================
        # 3. METHOD RECOMMENDATION
        # =====================================================

        recommender_input = {
            "primary_task": primary_task,
            "dataset": dataset_info,
        }

        try:
            recommendation_result = self.recommender.recommend(
                recommender_input
            )
        except Exception as exc:
            return self._error(
                f"Gagal melakukan method recommendation: {exc}"
            )

        # =====================================================
        # 4. EMPIRICAL EVALUATION
        # =====================================================

        evaluation_result = {
            "status": "SKIPPED",
            "results": [],
            "result_count": 0,
            "message": "Evaluation tidak dijalankan.",
        }

        if evaluate_models and target and task in {
            "binary_classification",
            "multiclass_classification",
            "regression",
        }:

            recommended_methods = [
                item["method_id"]
                for item in recommendation_result.get(
                    "recommendations",
                    [],
                )
            ]

            try:
                evaluation_result = self.evaluator.evaluate(
                    dataframe=dataframe,
                    target=target,
                    task=task,
                    methods=recommended_methods,
                )

            except Exception as exc:
                evaluation_result = {
                    "status": "ERROR",
                    "results": [],
                    "result_count": 0,
                    "message": (
                        f"Evaluation gagal: {exc}"
                    ),
                }

        # =====================================================
        # 5. BEST EMPIRICAL METHOD
        # =====================================================

        best_method = self._get_best_method(
            evaluation_result
        )

        # =====================================================
        # 6. FINAL RESULT
        # =====================================================

        return {
            "status": "ML_INTELLIGENCE",

            "task_detection": task_result,

            "primary_task": primary_task,

            "dataset": dataset_info,

            "recommendation": recommendation_result,

            "evaluation": evaluation_result,

            "best_method": best_method,

            "summary": self._build_summary(
                primary_task,
                recommendation_result,
                evaluation_result,
                best_method,
            ),

            "transparency": {
                "task_detection": "ESTIMATION",
                "method_recommendation": "RECOMMENDATION",
                "method_knowledge": "FACT",
                "model_evaluation": "EMPIRICAL",
                "best_method": (
                    "EMPIRICAL"
                    if best_method
                    else "NOT_AVAILABLE"
                ),
            },
        }

    # =========================================================
    # DATASET INFO
    # =========================================================

    def _extract_dataset_info(
        self,
        dataframe: pd.DataFrame,
        fingerprint: Dict[str, Any],
    ) -> Dict[str, Any]:

        numeric_features = len(
            dataframe.select_dtypes(
                include=["number"]
            ).columns
        )

        categorical_features = len(
            dataframe.select_dtypes(
                include=["object", "category", "bool"]
            ).columns
        )

        datetime_features = len(
            dataframe.select_dtypes(
                include=["datetime"]
            ).columns
        )

        return {
            "rows": len(dataframe),
            "columns": len(dataframe.columns),
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "datetime_features": datetime_features,
            "memory_usage": int(
                dataframe.memory_usage(
                    deep=True
                ).sum()
            ),
        }

    # =========================================================
    # BEST METHOD
    # =========================================================

    def _get_best_method(
        self,
        evaluation_result: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:

        results = evaluation_result.get(
            "results",
            [],
        )

        if not results:
            return None

        best = results[0]

        return {
            "method_id": best.get("method_id"),
            "metrics": best.get("metrics", {}),
            "status": "EMPIRICAL_BEST",
        }

    # =========================================================
    # SUMMARY
    # =========================================================

    def _build_summary(
        self,
        primary_task: Dict[str, Any],
        recommendation_result: Dict[str, Any],
        evaluation_result: Dict[str, Any],
        best_method: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:

        task = primary_task.get(
            "label",
            "Unknown Task",
        )

        recommendations = recommendation_result.get(
            "recommendations",
            [],
        )

        summary = {
            "task": task,
            "recommended_method": (
                recommendations[0]["method_id"]
                if recommendations
                else None
            ),
            "recommended_method_score": (
                recommendations[0]["score"]
                if recommendations
                else None
            ),
            "evaluated_methods": evaluation_result.get(
                "result_count",
                0,
            ),
            "best_empirical_method": (
                best_method["method_id"]
                if best_method
                else None
            ),
        }

        return summary

    # =====================================================
    # ERROR
    # =====================================================

    def _error(
        self,
        message: str,
    ) -> Dict[str, Any]:

        return {
            "status": "ERROR",
            "message": message,
            "task_detection": {},
            "primary_task": {},
            "dataset": {},
            "recommendation": {},
            "evaluation": {},
            "best_method": None,
            "summary": {},
            "transparency": {},
        }