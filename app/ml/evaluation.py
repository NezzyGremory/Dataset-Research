from __future__ import annotations

from typing import Any, Dict, Optional
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class MLEvaluator:
    """Train and compare real ML models on the supplied dataset.

    The evaluator is intentionally empirical: the final ranking comes from
    validation metrics produced by models trained on the user's dataset.
    """

    CLASSIFICATION_METHODS = [
        "random_forest_classifier",
        "gradient_boosting_classifier",
        "logistic_regression",
        "decision_tree_classifier",
        "svm_classifier",
        "knn_classifier",
    ]

    REGRESSION_METHODS = [
        "random_forest_regressor",
        "gradient_boosting_regressor",
        "linear_regression",
        "ridge_regression",
        "lasso_regression",
        "decision_tree_regressor",
    ]

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        cv: int = 3,
        max_cv_rows: int = 20000,
        n_jobs: int = 1,
    ) -> None:
        self.test_size = test_size
        self.random_state = random_state
        self.cv = cv
        self.max_cv_rows = max_cv_rows
        self.n_jobs = n_jobs

    # =========================================================
    # PUBLIC API
    # =========================================================

    def evaluate(
        self,
        dataframe: pd.DataFrame,
        target: str,
        task: str,
        methods: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        if dataframe is None or dataframe.empty:
            return self._error("Dataset kosong.")

        if target not in dataframe.columns:
            return self._error(
                f"Target '{target}' tidak ditemukan dalam dataset."
            )

        supported = {
            "binary_classification",
            "multiclass_classification",
            "regression",
        }
        if task not in supported:
            return self._error(
                f"Task '{task}' belum didukung evaluator."
            )

        data = dataframe.copy()
        data = data.dropna(subset=[target])

        if len(data) < 10:
            return self._error(
                "Dataset terlalu kecil untuk evaluation yang stabil."
            )

        X = self._prepare_features(data.drop(columns=[target]))
        y = data[target]

        if X.shape[1] == 0:
            return self._error(
                "Tidak terdapat feature yang dapat digunakan setelah preprocessing."
            )

        if task == "regression":
            if not pd.api.types.is_numeric_dtype(y):
                return self._error(
                    "Target regression harus berupa nilai numerik."
                )

            requested = methods or self.REGRESSION_METHODS
            return self._evaluate_regression(X, y, requested)

        class_count = int(y.nunique(dropna=True))
        if class_count < 2:
            return self._error(
                "Target classification harus memiliki minimal 2 kelas."
            )

        if task == "binary_classification" and class_count != 2:
            return self._error(
                "Binary classification membutuhkan tepat 2 kelas."
            )

        if task == "multiclass_classification" and class_count < 3:
            return self._error(
                "Multiclass classification membutuhkan minimal 3 kelas."
            )

        requested = methods or self.CLASSIFICATION_METHODS
        return self._evaluate_classification(X, y, task, requested)

    # =========================================================
    # CLASSIFICATION
    # =========================================================

    def _evaluate_classification(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task: str,
        methods: list[str],
    ) -> Dict[str, Any]:
        data_note = ""

        if len(X) > self.max_cv_rows:
            X, y = self._stratified_sample(
                X,
                y,
                self.max_cv_rows,
            )
            data_note = (
                f" Evaluasi menggunakan sampel {len(X):,} baris "
                "karena dataset sangat besar."
            )

        min_class = int(y.value_counts(dropna=False).min())
        folds = min(self.cv, min_class)
        use_cv = folds >= 2

        splitter = (
            StratifiedKFold(
                n_splits=folds,
                shuffle=True,
                random_state=self.random_state,
            )
            if use_cv
            else None
        )

        scoring = {
            "accuracy": "accuracy",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "f1": "f1_weighted",
        }

        results: list[Dict[str, Any]] = []

        for method_id in methods:
            model = self._classification_model(method_id)
            if model is None:
                continue

            try:
                pipe = self._pipeline(X, model, method_id)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    if use_cv:
                        scores = cross_validate(
                            pipe,
                            X,
                            y,
                            cv=splitter,
                            scoring=scoring,
                            n_jobs=self.n_jobs,
                            error_score="raise",
                        )

                        metrics = {
                            "accuracy": self._mean_pct(
                                scores["test_accuracy"]
                            ),
                            "precision": self._mean_pct(
                                scores["test_precision"]
                            ),
                            "recall": self._mean_pct(
                                scores["test_recall"]
                            ),
                            "f1_score": self._mean_pct(
                                scores["test_f1"]
                            ),
                        }

                        stability_values = (
                            scores["test_f1"] * 100.0
                        )
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X,
                            y,
                            test_size=self.test_size,
                            random_state=self.random_state,
                            stratify=y if min_class >= 2 else None,
                        )
                        pipe.fit(X_train, y_train)
                        predictions = pipe.predict(X_test)
                        metrics = self._classification_metrics(
                            y_test,
                            predictions,
                        )
                        stability_values = np.array(
                            [metrics["f1_score"]],
                            dtype=float,
                        )

                results.append(
                    {
                        "method_id": method_id,
                        "status": "EVALUATED",
                        "metrics": metrics,
                        "validation": (
                            "stratified_k_fold" if use_cv else "holdout"
                        ),
                        "folds": folds if use_cv else 1,
                        "stability": round(
                            float(np.std(stability_values)),
                            2,
                        ),
                    }
                )

            except Exception as exc:
                results.append(
                    {
                        "method_id": method_id,
                        "status": "FAILED",
                        "metrics": {},
                        "error": str(exc),
                    }
                )

        evaluated = [
            result
            for result in results
            if result.get("status") == "EVALUATED"
        ]
        evaluated.sort(
            key=lambda result: (
                result["metrics"]["f1_score"],
                result["metrics"]["accuracy"],
            ),
            reverse=True,
        )

        failed = [
            {
                "method_id": result.get("method_id"),
                "error": result.get("error", "Unknown error"),
            }
            for result in results
            if result.get("status") == "FAILED"
        ]

        return {
            "status": "EVALUATION" if evaluated else "FAILED",
            "task": task,
            "target": str(y.name),
            "dataset": {
                "rows": len(X),
                "features": X.shape[1],
            },
            "validation": (
                "stratified_k_fold" if use_cv else "holdout"
            ),
            "folds": folds if use_cv else 1,
            "metrics_used": [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
            ],
            "ranking_metric": "f1_score",
            "results": evaluated,
            "failed_methods": failed,
            "result_count": len(evaluated),
            "message": (
                "Model benar-benar dilatih dan dibandingkan menggunakan "
                f"{'Stratified K-Fold Cross Validation' if use_cv else 'holdout evaluation'}."
                f"{data_note}"
            ),
        }

    # =========================================================
    # REGRESSION
    # =========================================================

    def _evaluate_regression(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        methods: list[str],
    ) -> Dict[str, Any]:
        data_note = ""

        if len(X) > self.max_cv_rows:
            X = X.sample(
                self.max_cv_rows,
                random_state=self.random_state,
            )
            y = y.loc[X.index]
            data_note = (
                f" Evaluasi menggunakan sampel {len(X):,} baris "
                "karena dataset sangat besar."
            )

        folds = min(self.cv, len(X))
        use_cv = folds >= 2

        splitter = (
            KFold(
                n_splits=folds,
                shuffle=True,
                random_state=self.random_state,
            )
            if use_cv
            else None
        )

        # RMSE is derived from MSE for compatibility across sklearn versions.
        scoring = {
            "mae": "neg_mean_absolute_error",
            "mse": "neg_mean_squared_error",
            "r2": "r2",
        }

        results: list[Dict[str, Any]] = []

        for method_id in methods:
            model = self._regression_model(method_id)
            if model is None:
                continue

            try:
                pipe = self._pipeline(X, model, method_id)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    if use_cv:
                        scores = cross_validate(
                            pipe,
                            X,
                            y,
                            cv=splitter,
                            scoring=scoring,
                            n_jobs=self.n_jobs,
                            error_score="raise",
                        )

                        mae_values = -scores["test_mae"]
                        mse_values = -scores["test_mse"]
                        r2_values = scores["test_r2"]

                        metrics = {
                            "mae": round(
                                float(np.mean(mae_values)),
                                4,
                            ),
                            "rmse": round(
                                float(np.sqrt(np.mean(mse_values))),
                                4,
                            ),
                            "r2_score": round(
                                float(np.mean(r2_values)),
                                4,
                            ),
                        }
                        stability_values = r2_values
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X,
                            y,
                            test_size=self.test_size,
                            random_state=self.random_state,
                        )
                        pipe.fit(X_train, y_train)
                        predictions = pipe.predict(X_test)
                        mse = mean_squared_error(
                            y_test,
                            predictions,
                        )
                        metrics = {
                            "mae": round(
                                float(
                                    mean_absolute_error(
                                        y_test,
                                        predictions,
                                    )
                                ),
                                4,
                            ),
                            "rmse": round(
                                float(np.sqrt(mse)),
                                4,
                            ),
                            "r2_score": round(
                                float(
                                    r2_score(
                                        y_test,
                                        predictions,
                                    )
                                ),
                                4,
                            ),
                        }
                        stability_values = np.array(
                            [metrics["r2_score"]],
                            dtype=float,
                        )

                results.append(
                    {
                        "method_id": method_id,
                        "status": "EVALUATED",
                        "metrics": metrics,
                        "validation": "k_fold" if use_cv else "holdout",
                        "folds": folds if use_cv else 1,
                        "stability": round(
                            float(np.std(stability_values)),
                            4,
                        ),
                    }
                )

            except Exception as exc:
                results.append(
                    {
                        "method_id": method_id,
                        "status": "FAILED",
                        "metrics": {},
                        "error": str(exc),
                    }
                )

        evaluated = [
            result
            for result in results
            if result.get("status") == "EVALUATED"
        ]
        evaluated.sort(
            key=lambda result: (
                result["metrics"]["r2_score"],
                -result["metrics"]["rmse"],
            ),
            reverse=True,
        )

        failed = [
            {
                "method_id": result.get("method_id"),
                "error": result.get("error", "Unknown error"),
            }
            for result in results
            if result.get("status") == "FAILED"
        ]

        return {
            "status": "EVALUATION" if evaluated else "FAILED",
            "task": "regression",
            "target": str(y.name),
            "dataset": {
                "rows": len(X),
                "features": X.shape[1],
            },
            "validation": "k_fold" if use_cv else "holdout",
            "folds": folds if use_cv else 1,
            "metrics_used": ["mae", "rmse", "r2_score"],
            "ranking_metric": "r2_score",
            "results": evaluated,
            "failed_methods": failed,
            "result_count": len(evaluated),
            "message": (
                "Model benar-benar dilatih dan dibandingkan menggunakan "
                f"{'K-Fold Cross Validation' if use_cv else 'holdout evaluation'}."
                f"{data_note}"
            ),
        }

    # =========================================================
    # PREPROCESSING
    # =========================================================

    def _pipeline(
        self,
        X: pd.DataFrame,
        model,
        method_id: str,
    ) -> Pipeline:
        numeric = X.select_dtypes(include=["number"]).columns.tolist()
        categorical = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        transformers = []

        if numeric:
            transformers.append(
                (
                    "numeric",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(strategy="median"),
                            ),
                            (
                                "scaler",
                                StandardScaler(),
                            ),
                        ]
                    ),
                    numeric,
                )
            )

        if categorical:
            # Gradient Boosting works with dense matrices. Ordinal encoding
            # keeps categorical-heavy datasets from exploding in memory.
            if str(method_id).startswith("gradient_boosting"):
                encoder = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                )
            else:
                encoder = OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                )

            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="most_frequent"
                                ),
                            ),
                            ("encoder", encoder),
                        ]
                    ),
                    categorical,
                )
            )

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

        return Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

    def _prepare_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Expand real or strongly date-like columns.
        for col in list(X.columns):
            if self._looks_datetime(X[col]):
                parsed = pd.to_datetime(X[col], errors="coerce")
                if parsed.notna().mean() >= 0.8:
                    X[f"{col}_year"] = parsed.dt.year
                    X[f"{col}_month"] = parsed.dt.month
                    X[f"{col}_day"] = parsed.dt.day
                    X = X.drop(columns=[col])

        drop_columns = []
        for col in X.columns:
            series = X[col]
            normalized = (
                str(col)
                .strip()
                .lower()
                .replace("_", "")
                .replace("-", "")
                .replace(" ", "")
            )

            # Obvious identifiers should not become learned predictors.
            if normalized in {
                "id",
                "identifier",
                "uuid",
                "index",
                "recordid",
                "passengerid",
            }:
                drop_columns.append(col)
                continue

            if series.nunique(dropna=True) <= 1:
                drop_columns.append(col)
                continue

            # Very long, high-cardinality text is treated as non-tabular text
            # for this version rather than exploding one-hot encoding.
            if (
                pd.api.types.is_object_dtype(series)
                or pd.api.types.is_string_dtype(series)
            ):
                sample = series.dropna().astype(str).head(500)
                if not sample.empty:
                    avg_len = float(sample.str.len().mean())
                    unique_ratio = float(
                        sample.nunique() / max(len(sample), 1)
                    )
                    if avg_len > 50 and unique_ratio > 0.5:
                        drop_columns.append(col)

        return X.drop(
            columns=drop_columns,
            errors="ignore",
        )

    @staticmethod
    def _looks_datetime(series: pd.Series) -> bool:
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        if not (
            pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            return False
        sample = series.dropna().head(50)
        if sample.empty:
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce")
        return float(parsed.notna().mean()) >= 0.8

    # =========================================================
    # SAMPLING
    # =========================================================

    def _stratified_sample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n: int,
    ) -> tuple[pd.DataFrame, pd.Series]:
        if len(X) <= n:
            return X, y

        # Allocate the sample proportionally by class, keeping at least one
        # example from every class.
        selected_parts = []
        for _, indices in y.groupby(y).groups.items():
            count = max(
                1,
                int(round(n * len(indices) / len(y))),
            )
            count = min(count, len(indices))
            selected_parts.append(
                pd.Series(indices).sample(
                    count,
                    random_state=self.random_state,
                ).to_numpy()
            )

        selected = np.concatenate(selected_parts)
        if len(selected) > n:
            selected = pd.Series(selected).sample(
                n,
                random_state=self.random_state,
            ).to_numpy()

        return X.loc[selected], y.loc[selected]

    # =========================================================
    # MODEL FACTORIES
    # =========================================================

    def _classification_model(self, method_id: str):
        return {
            "random_forest_classifier": RandomForestClassifier(
                n_estimators=150,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "gradient_boosting_classifier": GradientBoostingClassifier(
                random_state=self.random_state,
            ),
            "logistic_regression": LogisticRegression(
                max_iter=1500,
                random_state=self.random_state,
            ),
            "decision_tree_classifier": DecisionTreeClassifier(
                random_state=self.random_state,
            ),
            "svm_classifier": SVC(
                random_state=self.random_state,
            ),
            "knn_classifier": KNeighborsClassifier(),
        }.get(method_id)

    def _regression_model(self, method_id: str):
        return {
            "random_forest_regressor": RandomForestRegressor(
                n_estimators=150,
                random_state=self.random_state,
                n_jobs=-1,
            ),
            "gradient_boosting_regressor": GradientBoostingRegressor(
                random_state=self.random_state,
            ),
            "linear_regression": LinearRegression(),
            "ridge_regression": Ridge(),
            "lasso_regression": Lasso(),
            "decision_tree_regressor": DecisionTreeRegressor(
                random_state=self.random_state,
            ),
        }.get(method_id)

    # =========================================================
    # METRICS / ERRORS
    # =========================================================

    @staticmethod
    def _mean_pct(values) -> float:
        return round(float(np.mean(values) * 100.0), 2)

    @staticmethod
    def _classification_metrics(
        y_true,
        predictions,
    ) -> Dict[str, float]:
        return {
            "accuracy": round(
                float(accuracy_score(y_true, predictions) * 100.0),
                2,
            ),
            "precision": round(
                float(
                    precision_score(
                        y_true,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    )
                    * 100.0
                ),
                2,
            ),
            "recall": round(
                float(
                    recall_score(
                        y_true,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    )
                    * 100.0
                ),
                2,
            ),
            "f1_score": round(
                float(
                    f1_score(
                        y_true,
                        predictions,
                        average="weighted",
                        zero_division=0,
                    )
                    * 100.0
                ),
                2,
            ),
        }

    @staticmethod
    def _error(message: str) -> Dict[str, Any]:
        return {
            "status": "ERROR",
            "results": [],
            "result_count": 0,
            "message": message,
        }
