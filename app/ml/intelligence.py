from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from app.ml.evaluation import MLEvaluator
from app.ml.method_recommender import MethodRecommender
from app.ml.task_detector import MLTaskDetector


class MLIntelligenceEngine:
    """Orchestrate task detection, candidate selection, and empirical model selection.

    The existing MethodRecommender is intentionally retained as a knowledge
    and candidate provider. It no longer determines the final ranking.
    Final ordering comes from real model evaluation on the supplied dataset.
    """

    SUPPORTED_TASKS = {
        "binary_classification",
        "multiclass_classification",
        "regression",
    }

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
        """Run the complete ML Intelligence pipeline."""

        if dataframe is None or dataframe.empty:
            return self._error("Dataset kosong.")

        # ---------------------------------------------------------
        # 1. TASK DETECTION
        # ---------------------------------------------------------
        try:
            task_result = self.task_detector.detect(
                fingerprint,
                dataframe=dataframe,
            )
        except TypeError:
            # Backward compatibility with older detector signatures.
            try:
                task_result = self.task_detector.detect(fingerprint)
            except Exception as exc:
                return self._error(
                    f"Gagal melakukan ML task detection: {exc}"
                )
        except Exception as exc:
            return self._error(
                f"Gagal melakukan ML task detection: {exc}"
            )

        if not isinstance(task_result, dict):
            return self._error(
                "ML task detector mengembalikan hasil yang tidak valid."
            )

        primary_task = task_result.get("primary_task") or {}
        if not isinstance(primary_task, dict):
            primary_task = {}

        # Validate the detector's target against the actual dataframe. This
        # prevents a weak categorical candidate from replacing a strong
        # explicit target column such as `Target` in regression datasets.
        primary_task = self._resolve_primary_task(
            dataframe=dataframe,
            primary_task=primary_task,
            task_result=task_result,
        )
        task_result = self._synchronize_task_result(
            task_result,
            primary_task,
        )

        task = str(primary_task.get("task") or "unknown_task")
        target = primary_task.get("target")

        dataset_info = self._extract_dataset_info(dataframe)

        # ---------------------------------------------------------
        # 2. METHOD KNOWLEDGE / CANDIDATE SELECTION
        # ---------------------------------------------------------
        static_recommendation = self._get_static_recommendation(
            primary_task,
            dataset_info,
        )

        candidate_methods = [
            item.get("method_id")
            for item in static_recommendation.get("recommendations", [])
            if isinstance(item, dict) and item.get("method_id")
        ]

        # The old recommender is allowed to define which known candidates are
        # worth testing, but its numeric score is NOT used for final ranking.
        if not candidate_methods:
            candidate_methods = self._default_candidates(task)

        # ---------------------------------------------------------
        # 3. EMPIRICAL EVALUATION
        # ---------------------------------------------------------
        evaluation = {
            "status": "SKIPPED",
            "results": [],
            "result_count": 0,
            "message": (
                "Evaluation tidak dijalankan karena task/target belum "
                "mendukung supervised model training."
            ),
        }

        if evaluate_models and target and task in self.SUPPORTED_TASKS:
            try:
                evaluation = self.evaluator.evaluate(
                    dataframe=dataframe,
                    target=str(target),
                    task=task,
                    methods=candidate_methods,
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

        # ---------------------------------------------------------
        # 4. COMPATIBLE RESULT SHAPE FOR THE EXISTING UI
        # ---------------------------------------------------------
        recommendation = {
            "status": (
                "EMPIRICAL_RECOMMENDATION"
                if empirical
                else "NO_EMPIRICAL_RESULT"
            ),
            "recommendations": empirical,
            "recommendation_count": len(empirical),
            "message": (
                "Ranking metode ditentukan dari hasil training dan validation "
                "pada dataset ini. Method Knowledge Base hanya digunakan "
                "sebagai sumber kandidat dan informasi metode."
            ),
            "selection_basis": evaluation.get(
                "ranking_metric",
                "empirical_evaluation",
            ),
        }

        return {
            "status": "ML_INTELLIGENCE",
            "task_detection": task_result,
            "primary_task": primary_task,
            "tasks": task_result.get("tasks", []),
            "task_count": task_result.get("task_count", 0),
            "target_candidates": task_result.get("target_candidates", []),
            "message": task_result.get("message", ""),
            "dataset": dataset_info,
            "recommendation": recommendation,
            "static_recommendation": static_recommendation,
            "evaluation": evaluation,
            "best_method": (
                {
                    "method_id": best.get("method_id"),
                    "method": best.get("method"),
                    "metrics": best.get("metrics", {}),
                    "status": "EMPIRICAL_BEST",
                    "selection_basis": "empirical_evaluation",
                }
                if best
                else None
            ),
            "summary": self._build_summary(
                primary_task,
                empirical,
                evaluation,
                best,
            ),
            "transparency": {
                "task_detection": "ESTIMATION",
                "candidate_selection": "KNOWLEDGE_BASE",
                "model_training": "EMPIRICAL",
                "model_validation": "EMPIRICAL",
                "final_ranking": "EMPIRICAL",
                "best_method": "EMPIRICAL" if best else "NOT_AVAILABLE",
            },
        }

    # =========================================================
    # TARGET / TASK VALIDATION
    # =========================================================

    def _resolve_primary_task(
        self,
        dataframe: pd.DataFrame,
        primary_task: Dict[str, Any],
        task_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        columns = list(dataframe.columns)
        if not columns:
            return primary_task

        target_hints = {
            "target": 100,
            "label": 80,
            "class": 80,
            "outcome": 75,
            "result": 65,
            "response": 65,
            "output": 65,
            "prediction": 60,
            "dependent": 60,
            "status": 55,
            "grade": 50,
            "score": 45,
            "value": 40,
            "state": 40,
            "group": 35,
            "segment": 35,
        }
        id_hints = {
            "id",
            "uuid",
            "identifier",
            "index",
            "recordid",
            "passengerid",
        }

        detector_target = str(primary_task.get("target") or "")
        candidate_names = set()
        for item in task_result.get("target_candidates", []) or []:
            if isinstance(item, dict):
                name = item.get("name") or item.get("column") or item.get("target")
                if name is not None:
                    candidate_names.add(str(name))
            elif item is not None:
                candidate_names.add(str(item))

        scored = []
        for position, column in enumerate(columns):
            name = str(column)
            normalized = (
                name.strip().lower()
                .replace("_", "")
                .replace("-", "")
                .replace(" ", "")
            )
            series = dataframe[column]
            unique = int(series.nunique(dropna=True))
            dtype_numeric = bool(pd.api.types.is_numeric_dtype(series))

            score = 0.0
            for hint, weight in target_hints.items():
                if normalized == hint:
                    score += weight
                    break

            if detector_target and name == detector_target:
                score += 35

            if name in candidate_names:
                score += 10

            if normalized in id_hints:
                score -= 100

            # Slight position bonus: many tabular datasets place the target
            # near the end, but position never overrides stronger name/type
            # evidence.
            if position == len(columns) - 1:
                score += 5

            if unique >= 2:
                if dtype_numeric:
                    if unique <= 2:
                        score += 8
                    elif unique <= 20:
                        score += 4
                else:
                    if unique <= 20:
                        score += 8

            if score > 0:
                scored.append((score, column, unique, dtype_numeric))

        if not scored:
            return primary_task

        scored.sort(key=lambda item: item[0], reverse=True)
        _, target_column, unique, numeric = scored[0]

        # Prefer a strong explicit target hint, but do not hijack datasets
        # whose best evidence is weaker than the detector's existing result.
        normalized_target = (
            str(target_column)
            .strip()
            .lower()
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
        )
        explicit_hint = normalized_target in target_hints
        detector_name = str(primary_task.get("target") or "")

        if detector_name and not explicit_hint and detector_name != target_column:
            # Keep the detector's result when there is no strong explicit name
            # evidence for another column.
            target_column = detector_name
            series = dataframe[target_column]
            unique = int(series.nunique(dropna=True))
            numeric = bool(pd.api.types.is_numeric_dtype(series))
            normalized_target = (
                target_column.strip().lower()
                .replace("_", "")
                .replace("-", "")
                .replace(" ", "")
            )

        if numeric:
            if unique <= 2:
                task = "binary_classification"
                label = "Binary Classification"
            elif unique <= 10:
                task = "multiclass_classification"
                label = "Multiclass Classification"
            else:
                task = "regression"
                label = "Regression"
        else:
            task = (
                "binary_classification"
                if unique <= 2
                else "multiclass_classification"
            )
            label = (
                "Binary Classification"
                if unique <= 2
                else "Multiclass Classification"
            )

        reasons = list(primary_task.get("reasons", []) or [])
        reasons.insert(
            0,
            f"Target divalidasi dari struktur dataframe: '{target_column}'.",
        )

        return {
            **primary_task,
            "task": task,
            "label": label,
            "target": str(target_column),
            "score": max(float(primary_task.get("score", 0) or 0), 75.0),
            "confidence": max(float(primary_task.get("confidence", 0) or 0), 75.0),
            "status": "ESTIMATION",
            "reasons": reasons[:5],
        }

    @staticmethod
    def _synchronize_task_result(
        task_result: Dict[str, Any],
        primary_task: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = dict(task_result)
        tasks = [
            item
            for item in (result.get("tasks", []) or [])
            if isinstance(item, dict)
        ]

        supervised_tasks = {
            "binary_classification",
            "multiclass_classification",
            "regression",
        }
        tasks = [
            item
            for item in tasks
            if item.get("task") not in supervised_tasks
        ]
        tasks.insert(0, primary_task)
        result["primary_task"] = primary_task
        result["tasks"] = tasks
        result["task_count"] = len(tasks)
        result["target_candidates"] = result.get("target_candidates", []) or []
        return result

    # =========================================================
    # STATIC KNOWLEDGE
    # =========================================================

    def _get_static_recommendation(
        self,
        primary_task: Dict[str, Any],
        dataset_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            result = self.recommender.recommend(
                {
                    "primary_task": primary_task,
                    "dataset": dataset_info,
                }
            )
            if isinstance(result, dict):
                return result
        except Exception:
            pass

        return {
            "status": "RECOMMENDATION",
            "recommendations": [],
            "recommendation_count": 0,
            "message": "Knowledge base tidak tersedia; evaluator menggunakan candidate set default.",
        }

    def _default_candidates(self, task: str) -> list[str]:
        if task in {"binary_classification", "multiclass_classification"}:
            return list(self.evaluator.CLASSIFICATION_METHODS)
        if task == "regression":
            return list(self.evaluator.REGRESSION_METHODS)
        return []

    # =========================================================
    # EMPIRICAL RESULT BUILDING
    # =========================================================

    def _build_empirical_recommendation(
        self,
        evaluation: Dict[str, Any],
        static_recommendation: Dict[str, Any],
    ) -> list[Dict[str, Any]]:
        results = evaluation.get("results", []) or []
        static_items = static_recommendation.get("recommendations", []) or []
        static_map = {
            item.get("method_id"): item
            for item in static_items
            if isinstance(item, dict) and item.get("method_id")
        }

        output: list[Dict[str, Any]] = []

        for rank, result in enumerate(results, start=1):
            if not isinstance(result, dict):
                continue

            method_id = result.get("method_id")
            if not method_id:
                continue

            info = static_map.get(method_id, {})
            metrics = result.get("metrics", {}) or {}

            if "f1_score" in metrics:
                score_type = "f1_score"
                score = float(metrics.get("f1_score", 0.0))
                score_label = f"F1 {score:.2f}%"
            elif "r2_score" in metrics:
                score_type = "r2_score"
                score = float(metrics.get("r2_score", 0.0))
                score_label = f"R² {score:.4f}"
            else:
                score_type = evaluation.get(
                    "ranking_metric",
                    "empirical",
                )
                score = 0.0
                score_label = "Evaluated"

            validation = result.get(
                "validation",
                evaluation.get("validation", "empirical"),
            )

            reasons = [
                "Model benar-benar dilatih pada dataset ini.",
                f"Metric utama: {self._format_metric_name(score_type)}.",
                f"Validasi: {validation}.",
            ]

            if result.get("stability") is not None:
                reasons.append(
                    f"Variasi metric antar fold: {result.get('stability')}."
                )

            item = {
                "rank": rank,
                "method_id": method_id,
                "method": info.get(
                    "method",
                    self._pretty_method(method_id),
                ),
                "category": info.get(
                    "category",
                    self._category(method_id),
                ),
                "score": round(score, 4),
                "score_type": score_type,
                "score_label": score_label,
                "status": "EMPIRICAL",
                "metrics": metrics,
                "validation": validation,
                "folds": result.get(
                    "folds",
                    evaluation.get("folds"),
                ),
                "stability": result.get("stability"),
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
                "knowledge_base_score": info.get("score"),
                "empirical": True,
            }
            output.append(item)

        return output

    # =========================================================
    # HELPERS
    # =========================================================

    @staticmethod
    def _extract_dataset_info(
        dataframe: pd.DataFrame,
    ) -> Dict[str, Any]:
        return {
            "rows": int(len(dataframe)),
            "columns": int(len(dataframe.columns)),
            "numeric_features": int(
                dataframe.select_dtypes(include=["number"]).shape[1]
            ),
            "categorical_features": int(
                dataframe.select_dtypes(
                    include=["object", "category", "bool"]
                ).shape[1]
            ),
            "datetime_features": int(
                dataframe.select_dtypes(include=["datetime"]).shape[1]
            ),
            "missing_cells": int(dataframe.isna().sum().sum()),
            "memory_usage": int(
                dataframe.memory_usage(deep=True).sum()
            ),
        }

    @staticmethod
    def _build_summary(
        primary_task: Dict[str, Any],
        empirical: list[Dict[str, Any]],
        evaluation: Dict[str, Any],
        best: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        return {
            "task": primary_task.get("label", "Unknown Task"),
            "recommended_method": (
                empirical[0]["method_id"]
                if empirical
                else None
            ),
            "recommended_method_score": (
                empirical[0]["score"]
                if empirical
                else None
            ),
            "recommended_method_score_type": (
                empirical[0].get("score_type")
                if empirical
                else None
            ),
            "evaluated_methods": evaluation.get("result_count", 0),
            "best_empirical_method": (
                best.get("method_id")
                if best
                else None
            ),
            "selection_basis": (
                "empirical_evaluation"
                if empirical
                else "not_available"
            ),
        }

    @staticmethod
    def _pretty_method(method_id: str) -> str:
        return str(method_id or "Unknown").replace(
            "_",
            " ",
        ).title()

    @staticmethod
    def _category(method_id: str) -> str:
        return (
            "Regression"
            if "regressor" in str(method_id)
            else "Classification"
        )

    @staticmethod
    def _format_metric_name(metric: str) -> str:
        return {
            "f1_score": "F1 Score",
            "r2_score": "R²",
        }.get(metric, str(metric).replace("_", " ").title())

    @staticmethod
    def _error(message: str) -> Dict[str, Any]:
        return {
            "status": "ERROR",
            "message": message,
            "task_detection": {},
            "primary_task": {},
            "tasks": [],
            "task_count": 0,
            "dataset": {},
            "recommendation": {
                "status": "ERROR",
                "recommendations": [],
                "recommendation_count": 0,
            },
            "static_recommendation": {},
            "evaluation": {
                "status": "ERROR",
                "results": [],
                "result_count": 0,
            },
            "best_method": None,
            "summary": {},
            "transparency": {},
        }
