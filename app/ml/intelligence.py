from __future__ import annotations

from typing import Any, Dict, Optional
import pandas as pd

from app.ml.evaluation import MLEvaluator
from app.ml.method_recommender import MethodRecommender
from app.ml.task_detector import MLTaskDetector


class MLIntelligenceEngine:
    """Dataset-driven ML intelligence.

    Static method knowledge is retained for explanations, while empirical
    training/validation determines the final model ordering.
    """

    def __init__(
        self,
        evaluator: Optional[MLEvaluator] = None,
        recommender: Optional[MethodRecommender] = None,
        task_detector: Optional[MLTaskDetector] = None,
    ):
        self.task_detector = task_detector or MLTaskDetector()
        self.recommender = recommender or MethodRecommender()
        self.evaluator = evaluator or MLEvaluator()

    def analyze(self, dataframe: pd.DataFrame, fingerprint: Dict[str, Any], evaluate_models: bool = True):
        try:
            task_result = self.task_detector.detect(fingerprint)
        except Exception as exc:
            return self._error(f"Gagal melakukan ML task detection: {exc}")

        primary_task = task_result.get("primary_task", {}) or {}
        task = primary_task.get("task", "unknown_task")
        target = primary_task.get("target")

        dataset_info = self._extract_dataset_info(dataframe, fingerprint)

        # Keep the old recommender as a knowledge/candidate source.
        try:
            static_recommendation = self.recommender.recommend({
                "primary_task": primary_task,
                "dataset": dataset_info,
            })
        except Exception:
            static_recommendation = {
                "status": "RECOMMENDATION",
                "recommendations": [],
                "recommendation_count": 0,
            }

        evaluation = {
            "status": "SKIPPED",
            "results": [],
            "result_count": 0,
            "message": "Evaluation tidak dijalankan.",
        }

        supported = {"binary_classification", "multiclass_classification", "regression"}
        if evaluate_models and target and task in supported:
            # IMPORTANT: methods=None means evaluate the complete candidate set.
            try:
                evaluation = self.evaluator.evaluate(
                    dataframe=dataframe,
                    target=target,
                    task=task,
                    methods=None,
                )
            except Exception as exc:
                evaluation = {
                    "status": "ERROR",
                    "results": [],
                    "result_count": 0,
                    "message": f"Evaluation gagal: {exc}",
                }

        empirical = self._build_empirical_recommendation(
            evaluation=evaluation,
            static_recommendation=static_recommendation,
        )

        best = empirical[0] if empirical else None

        return {
            "status": "ML_INTELLIGENCE",
            "task_detection": task_result,
            "primary_task": primary_task,
            "dataset": dataset_info,

            # Compatibility: UI can keep reading recommendation.
            "recommendation": {
                "status": "EMPIRICAL_RECOMMENDATION",
                "recommendations": empirical,
                "recommendation_count": len(empirical),
                "message": (
                    "Urutan metode ditentukan dari performa training/validation "
                    "pada dataset ini. Knowledge base hanya digunakan untuk "
                    "penjelasan dan konteks metode."
                ),
                "selection_basis": evaluation.get(
                    "validation", "empirical_evaluation"
                ),
            },

            # Preserve the old heuristic result for transparency/debugging.
            "static_recommendation": static_recommendation,
            "evaluation": evaluation,
            "best_method": (
                {
                    "method_id": best["method_id"],
                    "method": best.get("method"),
                    "metrics": best.get("metrics", {}),
                    "status": "EMPIRICAL_BEST",
                    "selection_basis": "empirical_evaluation",
                }
                if best else None
            ),

            "summary": self._build_summary(primary_task, empirical, evaluation, best),

            "transparency": {
                "task_detection": "ESTIMATION",
                "static_method_knowledge": "RECOMMENDATION",
                "model_training": "EMPIRICAL",
                "model_validation": "EMPIRICAL",
                "final_ranking": "EMPIRICAL",
                "best_method": "EMPIRICAL" if best else "NOT_AVAILABLE",
            },
        }

    def _build_empirical_recommendation(self, evaluation, static_recommendation):
        results = evaluation.get("results", []) or []
        static = {
            x.get("method_id"): x
            for x in static_recommendation.get("recommendations", [])
            if isinstance(x, dict)
        }

        output = []
        for rank, result in enumerate(results, start=1):
            method_id = result.get("method_id")
            info = static.get(method_id, {})
            metrics = result.get("metrics", {})

            if "f1_score" in metrics:
                primary_metric = "f1_score"
                empirical_score = float(metrics["f1_score"])
            else:
                primary_metric = "r2_score"
                # R2 is deliberately kept as the actual metric; do not turn
                # it into a fake "suitability percentage".
                empirical_score = float(metrics.get("r2_score", 0.0))

            item = {
                "rank": rank,
                "method_id": method_id,
                "method": info.get("method", self._pretty_method(method_id)),
                "category": info.get("category", self._category(method_id)),
                "score": round(empirical_score, 2),
                "score_type": primary_metric,
                "status": "EMPIRICAL",
                "metrics": metrics,
                "validation": result.get("validation"),
                "folds": result.get("folds"),
                "stability": result.get("stability"),
                "description": info.get("description", ""),
                "strengths": info.get("strengths", []),
                "limitations": info.get("limitations", []),
                "reasons": [
                    f"Dilatih dan dievaluasi pada dataset ini.",
                    f"Metric utama: {primary_metric}.",
                    f"Validasi: {result.get('validation', 'empirical')}.",
                ],
            }
            output.append(item)

        return output

    @staticmethod
    def _pretty_method(method_id):
        return str(method_id or "Unknown").replace("_", " ").title()

    @staticmethod
    def _category(method_id):
        if "regressor" in str(method_id):
            return "Regression"
        return "Classification"

    @staticmethod
    def _extract_dataset_info(dataframe, fingerprint):
        return {
            "rows": len(dataframe),
            "columns": len(dataframe.columns),
            "numeric_features": len(dataframe.select_dtypes(include=["number"]).columns),
            "categorical_features": len(dataframe.select_dtypes(include=["object", "category", "bool"]).columns),
            "datetime_features": len(dataframe.select_dtypes(include=["datetime"]).columns),
            "memory_usage": int(dataframe.memory_usage(deep=True).sum()),
        }

    @staticmethod
    def _build_summary(primary_task, empirical, evaluation, best):
        return {
            "task": primary_task.get("label", "Unknown Task"),
            "recommended_method": empirical[0]["method_id"] if empirical else None,
            "recommended_method_score": empirical[0]["score"] if empirical else None,
            "evaluated_methods": evaluation.get("result_count", 0),
            "best_empirical_method": best["method_id"] if best else None,
            "selection_basis": "empirical_evaluation" if empirical else "not_available",
        }

    @staticmethod
    def _error(message):
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
