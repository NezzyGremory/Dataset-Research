"""Evaluasi rekomendasi metode terhadap cross-validation berpasangan pada DEV."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor,
                              ExtraTreesClassifier, ExtraTreesRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from app.analyzer.fingerprint import DatasetFingerprint
from app.ml.method_info import get_method_info
from app.ml.method_recommender import MethodRecommender
from evaluation.download_datasets import DATA_ROOT, sha256_file
from evaluation.metrics import (bootstrap_ci, mcnemar_test,
                                random_ranking_baseline, wilcoxon_signed_rank)
from evaluation.protocol import EvaluationInputError, load_manifest, select_split

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261008


def _estimator(method_id: str, task: str):
    """Bangun implementasi estimator untuk metode knowledge-base yang tersedia."""
    classifiers = {
        "random_forest_classifier": lambda: RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
        "extra_trees_classifier": lambda: ExtraTreesClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
        "gradient_boosting_classifier": lambda: GradientBoostingClassifier(random_state=SEED),
        "logistic_regression": lambda: LogisticRegression(max_iter=1000, random_state=SEED),
        "adaboost_classifier": lambda: AdaBoostClassifier(random_state=SEED),
        "decision_tree_classifier": lambda: DecisionTreeClassifier(random_state=SEED),
        "svm_classifier": lambda: SVC(random_state=SEED),
        "knn_classifier": KNeighborsClassifier,
    }
    regressors = {
        "random_forest_regressor": lambda: RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
        "extra_trees_regressor": lambda: ExtraTreesRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
        "gradient_boosting_regressor": lambda: GradientBoostingRegressor(random_state=SEED),
        "linear_regression": LinearRegression,
        "adaboost_regressor": lambda: AdaBoostRegressor(random_state=SEED),
        "decision_tree_regressor": lambda: DecisionTreeRegressor(random_state=SEED),
        "ridge_regression": Ridge,
        "lasso_regression": Lasso,
    }
    choices = classifiers if task == "classification" else regressors
    factory = choices.get(method_id)
    return factory() if factory else None


def _pipeline(estimator, frame: pd.DataFrame, scaling: bool) -> Pipeline:
    numeric = frame.select_dtypes(include=["number"]).columns.tolist()
    categorical = [column for column in frame.columns if column not in numeric]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scaling:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers = [("numeric", Pipeline(numeric_steps), numeric)]
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    preprocessing = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("preprocess", preprocessing), ("model", estimator)])


def run_method_evaluation(manifest_path: Path, quick: bool = False) -> dict:
    """Bandingkan top-3 aplikasi dengan performa 5-fold dan baseline tetap/acak."""
    rows = select_split(load_manifest(manifest_path), "dev")
    if not rows:
        raise EvaluationInputError("TIDAK DIJALANKAN: tidak ada manifest DEV.")
    if quick:
        rows = rows[:1]
    fingerprints = DatasetFingerprint()
    recommender = MethodRecommender()
    per_dataset, failures = [], []
    repeats = 3 if quick else 5
    for record in rows:
        dataset_id = record["dataset_id"]
        path = DATA_ROOT / "datasets" / f"{dataset_id.replace(':', '_')}.csv"
        metadata_path = DATA_ROOT / "manifests" / "downloaded_dev.json"
        if not path.exists() or not metadata_path.exists():
            raise EvaluationInputError(f"TIDAK DIJALANKAN: dataset/cache DEV belum tersedia: {dataset_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cached = next((item for item in metadata["datasets"] if item["dataset_id"] == dataset_id), None)
        digest = sha256_file(path)
        if not cached or digest != cached["sha256_before"]:
            raise EvaluationInputError(f"Hash dataset tidak sesuai cache: {dataset_id}")
        frame = pd.read_csv(path)
        if record["target_name"] not in frame.columns:
            failures.append({"dataset_id": dataset_id, "error": "target tidak ada pada salinan"})
            continue
        y = frame[record["target_name"]]
        X = frame.drop(columns=[record["target_name"]])
        task = record["task_label"]
        if task == "classification":
            task_name = "binary_classification" if y.nunique(dropna=True) == 2 else "multiclass_classification"
            splitter = StratifiedKFold(n_splits=repeats, shuffle=True, random_state=SEED)
            scoring = "accuracy"
        elif task == "regression":
            task_name = "regression"
            splitter = KFold(n_splits=repeats, shuffle=True, random_state=SEED)
            scoring = "r2"
        else:
            continue
        fingerprint = fingerprints.generate_representation(frame)
        recommendation = recommender.recommend({"primary_task": {"task": task_name,
                                        "label": task_name.replace("_", " "),
                                        "target": record["target_name"]}}, fingerprint)
        ranked = [item["method_id"] for item in recommendation["recommendations"]]
        top_three = ranked[:3]
        fixed_method = "random_forest_classifier" if task == "classification" else "random_forest_regressor"
        candidate_method_ids = ranked if not quick else list(dict.fromkeys(top_three + [fixed_method]))
        empirical, errors = {}, []
        for method_id in candidate_method_ids:
            estimator = _estimator(method_id, task)
            if estimator is None:
                errors.append({"method_id": method_id, "reason": "dependency/estimator belum tersedia"})
                continue
            info = get_method_info(method_id) or {}
            model = _pipeline(estimator, X, bool(info.get("scaling_required")))
            try:
                scores = cross_val_score(model, X, y, cv=splitter, scoring=scoring,
                                         error_score="raise", n_jobs=1)
                empirical[method_id] = float(np.mean(scores))
            except Exception as error:
                errors.append({"method_id": method_id, "reason": f"{type(error).__name__}: {error}"})
        if not empirical:
            failures.append({"dataset_id": dataset_id, "error": "tidak ada estimator yang berhasil", "methods": errors})
            continue
        empirical_ranked = sorted(empirical, key=empirical.get, reverse=True)
        best = empirical_ranked[0]
        supported_top_three = [method for method in ranked if method in empirical][:3]
        random_methods = list(empirical)
        random_top = [random_methods[index] for index in random_ranking_baseline(len(random_methods), seed=SEED)[:3]]
        fixed_top = [fixed_method]
        first_supported = next((method for method in ranked if method in empirical), best)
        app_rank = empirical_ranked.index(first_supported) + 1
        per_dataset.append({
            "dataset_id": dataset_id, "task": task, "recommended_top3": top_three,
            "recommended_top3_supported": supported_top_three,
            "empirical_scores": empirical, "empirical_best": best,
            "top3_hit": best in supported_top_three, "empirical_rank_of_top_recommendation": app_rank,
            "empirical_rank_of_fixed_baseline": empirical_ranked.index(fixed_method) + 1 if fixed_method in empirical_ranked else len(empirical_ranked) + 1,
            "score_gap_from_best": empirical[best] - max((empirical[m] for m in supported_top_three), default=empirical[best]),
            "fixed_baseline_top3_hit": best in fixed_top,
            "random_baseline_top3_hit": best in random_top,
            "cv_folds": repeats, "scoring": scoring, "failed_estimators": errors,
            "sha256_before": digest, "sha256_after": sha256_file(path),
        })
    if not per_dataset:
        raise EvaluationInputError("TIDAK DIJALANKAN: tidak ada dataset DEV yang dapat dievaluasi oleh estimator.")
    app_hits = [int(row["top3_hit"]) for row in per_dataset]
    fixed_hits = [int(row["fixed_baseline_top3_hit"]) for row in per_dataset]
    random_hits = [int(row["random_baseline_top3_hit"]) for row in per_dataset]
    return {
        "split": "dev", "quick": quick, "dataset_count": len(per_dataset),
        "cv_folds": repeats, "scoring_note": "Skor dibandingkan hanya dalam dataset/task yang sama; accuracy untuk klasifikasi dan R2 untuk regresi.",
        "top3_hit_rate": bootstrap_ci(app_hits, seed=SEED),
        "mean_empirical_rank_of_first_supported_recommendation": float(np.mean([x["empirical_rank_of_top_recommendation"] for x in per_dataset])),
        "mean_gap_to_best_within_recommended_top3": float(np.mean([x["score_gap_from_best"] for x in per_dataset])),
        "fixed_random_forest_top3_hit_rate": float(np.mean(fixed_hits)),
        "random_top3_hit_rate": float(np.mean(random_hits)),
        "paired_mcnemar_vs_fixed": mcnemar_test([1] * len(app_hits), app_hits, fixed_hits),
        "paired_mcnemar_vs_random": mcnemar_test([1] * len(app_hits), app_hits, random_hits),
        "paired_wilcoxon_rank_vs_fixed": wilcoxon_signed_rank(
            [x["empirical_rank_of_top_recommendation"] for x in per_dataset],
            [x["empirical_rank_of_fixed_baseline"] for x in per_dataset]),
        "per_dataset": per_dataset, "failures": failures,
    }
