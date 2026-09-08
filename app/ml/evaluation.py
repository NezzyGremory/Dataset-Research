from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


class MLEvaluator:
    """
    Mengevaluasi performa model Machine Learning secara empiris.

    Tahap awal:
    - Binary Classification
    - Multiclass Classification
    - Regression

    Catatan:
    Hasil evaluation adalah performa empiris pada dataset,
    bukan jaminan bahwa model akan memiliki performa yang sama
    pada data baru.
    """

    def __init__(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> None:
        self.test_size = test_size
        self.random_state = random_state

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
        """
        Mengevaluasi beberapa metode pada dataset.

        Parameters
        ----------
        dataframe:
            Dataset pandas.

        target:
            Nama kolom target.

        task:
            binary_classification,
            multiclass_classification,
            regression.

        methods:
            Daftar method_id yang ingin dievaluasi.
            Jika None, gunakan semua method yang tersedia.
        """

        if dataframe is None or dataframe.empty:
            return self._error("Dataset kosong.")

        if target not in dataframe.columns:
            return self._error(
                f"Target '{target}' tidak ditemukan dalam dataset."
            )

        if task not in {
            "binary_classification",
            "multiclass_classification",
            "regression",
        }:
            return self._error(
                f"Task '{task}' belum didukung evaluator."
            )

        data = dataframe.copy()

        # Hapus baris target kosong.
        data = data.dropna(subset=[target])

        if len(data) < 10:
            return self._error(
                "Dataset terlalu kecil untuk evaluation."
            )

        X = data.drop(columns=[target])
        y = data[target]

        # Jangan masukkan kolom ID yang jelas-jelas bukan feature.
        X = self._remove_id_columns(X)

        if X.shape[1] == 0:
            return self._error(
                "Tidak terdapat feature yang dapat digunakan."
            )

        X = self._prepare_features(X)

        if task == "regression":
            return self._evaluate_regression(
                X,
                y,
                methods,
            )

        return self._evaluate_classification(
            X,
            y,
            task,
            methods,
        )

    # =========================================================
    # CLASSIFICATION
    # =========================================================

    def _evaluate_classification(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        task: str,
        methods: Optional[list[str]],
    ) -> Dict[str, Any]:

        if y.nunique() < 2:
            return self._error(
                "Target classification harus memiliki minimal 2 kelas."
            )

        if task == "binary_classification" and y.nunique() != 2:
            return self._error(
                "Binary classification membutuhkan tepat 2 kelas."
            )

        if (
            task == "multiclass_classification"
            and y.nunique() < 3
        ):
            return self._error(
                "Multiclass classification membutuhkan minimal 3 kelas."
            )

        default_methods = [
            "random_forest_classifier",
            "logistic_regression",
            "decision_tree_classifier",
            "svm_classifier",
            "knn_classifier",
        ]

        methods = methods or default_methods

        stratify = y if y.value_counts().min() >= 2 else None

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=stratify,
            )
        except ValueError as exc:
            return self._error(
                f"Gagal membagi dataset: {exc}"
            )

        preprocessor = self._build_preprocessor(X_train)

        results = []

        for method_id in methods:
            model = self._classification_model(method_id)

            if model is None:
                continue

            try:
                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        ("model", model),
                    ]
                )

                pipeline.fit(X_train, y_train)

                predictions = pipeline.predict(X_test)

                result = {
                    "method_id": method_id,
                    "status": "EVALUATED",
                    "metrics": {
                        "accuracy": round(
                            accuracy_score(
                                y_test,
                                predictions,
                            ) * 100,
                            2,
                        ),
                        "precision": round(
                            precision_score(
                                y_test,
                                predictions,
                                average="weighted",
                                zero_division=0,
                            ) * 100,
                            2,
                        ),
                        "recall": round(
                            recall_score(
                                y_test,
                                predictions,
                                average="weighted",
                                zero_division=0,
                            ) * 100,
                            2,
                        ),
                        "f1_score": round(
                            f1_score(
                                y_test,
                                predictions,
                                average="weighted",
                                zero_division=0,
                            ) * 100,
                            2,
                        ),
                    },
                }

                results.append(result)

            except Exception as exc:
                results.append(
                    {
                        "method_id": method_id,
                        "status": "FAILED",
                        "metrics": {},
                        "error": str(exc),
                    }
                )

        results = [
            result
            for result in results
            if result["status"] == "EVALUATED"
        ]

        results.sort(
            key=lambda item: item["metrics"]["f1_score"],
            reverse=True,
        )

        return {
            "status": "EVALUATION",
            "task": task,
            "target": str(y.name),
            "dataset": {
                "rows": len(X),
                "features": X.shape[1],
            },
            "metrics_used": [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
            ],
            "results": results,
            "result_count": len(results),
            "message": (
                "Hasil berdasarkan evaluasi empiris menggunakan "
                "train-test split. Performa dapat berubah pada "
                "data baru."
            ),
        }

    # =========================================================
    # REGRESSION
    # =========================================================

    def _evaluate_regression(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        methods: Optional[list[str]],
    ) -> Dict[str, Any]:

        if not pd.api.types.is_numeric_dtype(y):
            return self._error(
                "Target regression harus berupa nilai numerik."
            )

        default_methods = [
            "random_forest_regressor",
            "gradient_boosting_regressor",
            "linear_regression",
            "decision_tree_regressor",
        ]

        methods = methods or default_methods

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
            )
        except ValueError as exc:
            return self._error(
                f"Gagal membagi dataset: {exc}"
            )

        preprocessor = self._build_preprocessor(X_train)

        results = []

        for method_id in methods:
            model = self._regression_model(method_id)

            if model is None:
                continue

            try:
                pipeline = Pipeline(
                    steps=[
                        ("preprocessor", preprocessor),
                        ("model", model),
                    ]
                )

                pipeline.fit(X_train, y_train)

                predictions = pipeline.predict(X_test)

                mse = mean_squared_error(
                    y_test,
                    predictions,
                )

                rmse = mse ** 0.5

                result = {
                    "method_id": method_id,
                    "status": "EVALUATED",
                    "metrics": {
                        "mae": round(
                            mean_absolute_error(
                                y_test,
                                predictions,
                            ),
                            4,
                        ),
                        "rmse": round(rmse, 4),
                        "r2_score": round(
                            r2_score(
                                y_test,
                                predictions,
                            ),
                            4,
                        ),
                    },
                }

                results.append(result)

            except Exception as exc:
                results.append(
                    {
                        "method_id": method_id,
                        "status": "FAILED",
                        "metrics": {},
                        "error": str(exc),
                    }
                )

        results = [
            result
            for result in results
            if result["status"] == "EVALUATED"
        ]

        results.sort(
            key=lambda item: item["metrics"]["r2_score"],
            reverse=True,
        )

        return {
            "status": "EVALUATION",
            "task": "regression",
            "target": str(y.name),
            "dataset": {
                "rows": len(X),
                "features": X.shape[1],
            },
            "metrics_used": [
                "mae",
                "rmse",
                "r2_score",
            ],
            "results": results,
            "result_count": len(results),
            "message": (
                "Hasil berdasarkan evaluasi empiris menggunakan "
                "train-test split. Performa dapat berubah pada "
                "data baru."
            ),
        }

    # =========================================================
    # MODEL FACTORIES
    # =========================================================

    def _classification_model(self, method_id: str):
        models = {
            "random_forest_classifier": RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
            ),
            "logistic_regression": LogisticRegression(
                max_iter=1000,
            ),
            "decision_tree_classifier": DecisionTreeClassifier(
                random_state=self.random_state,
            ),
            "svm_classifier": SVC(
                random_state=self.random_state,
            ),
            "knn_classifier": KNeighborsClassifier(),
        }

        return models.get(method_id)

    def _regression_model(self, method_id: str):
        models = {
            "random_forest_regressor": RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
            ),
            "gradient_boosting_regressor": GradientBoostingRegressor(
                random_state=self.random_state,
            ),
            "linear_regression": LinearRegression(),
            "decision_tree_regressor": DecisionTreeRegressor(
                random_state=self.random_state,
            ),
        }

        return models.get(method_id)

    # =========================================================
    # PREPROCESSING
    # =========================================================

    def _build_preprocessor(
        self,
        X: pd.DataFrame,
    ) -> ColumnTransformer:

        numeric_columns = X.select_dtypes(
            include=["number"]
        ).columns.tolist()

        categorical_columns = X.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    ),
                ),
            ]
        )

        transformers = []

        if numeric_columns:
            transformers.append(
                (
                    "numeric",
                    numeric_pipeline,
                    numeric_columns,
                )
            )

        if categorical_columns:
            transformers.append(
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_columns,
                )
            )

        return ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

    def _prepare_features(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:

        X = X.copy()

        # Datetime diubah menjadi komponen numerik sederhana.
        for column in X.columns:

            if pd.api.types.is_datetime64_any_dtype(
                X[column]
            ):
                X[f"{column}_year"] = X[column].dt.year
                X[f"{column}_month"] = X[column].dt.month
                X[f"{column}_day"] = X[column].dt.day

                X = X.drop(columns=[column])

        return X

    def _remove_id_columns(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:

        X = X.copy()

        id_keywords = {
            "id",
            "identifier",
            "uuid",
            "passengerid",
        }

        remove_columns = []

        for column in X.columns:
            normalized = (
                str(column)
                .strip()
                .lower()
                .replace("_", "")
                .replace("-", "")
                .replace(" ", "")
            )

            if normalized in id_keywords:
                remove_columns.append(column)

        if remove_columns:
            X = X.drop(columns=remove_columns)

        return X

    # =========================================================
    # ERROR
    # =========================================================

    def _error(self, message: str) -> Dict[str, Any]:
        return {
            "status": "ERROR",
            "results": [],
            "result_count": 0,
            "message": message,
        }