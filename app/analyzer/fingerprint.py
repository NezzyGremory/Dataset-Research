import hashlib
import json
import re

import pandas as pd


class DatasetFingerprint:
    """
    Generate a normalized fingerprint representation for a dataset.

    Fingerprint digunakan untuk merangkum karakteristik dataset
    dan menjadi identitas heuristik dataset.
    """

    def generate_representation(self, dataframe: pd.DataFrame) -> dict:
        rows = int(len(dataframe))
        columns_count = int(len(dataframe.columns))

        columns = []

        for column in dataframe.columns:
            series = dataframe[column]

            column_signature = self._analyze_column(
                column=column,
                series=series,
            )

            columns.append(column_signature)

        characteristics = self._build_characteristics(columns)

        target_candidates = [
            column["name"]
            for column in columns
            if column["target_candidate"]
        ]

        keywords = self._extract_keywords(dataframe)

        categories = self._build_categories(columns)

        return {
            "rows": rows,
            "columns": columns_count,
            "characteristics": characteristics,
            "target_candidates": target_candidates,
            "keywords": keywords,
            "categories": categories,
            "column_signature": columns,
        }

    def _analyze_column(
        self,
        column,
        series: pd.Series,
    ) -> dict:

        column_name = str(column)
        row_count = len(series)

        missing_count = int(series.isna().sum())

        missing_percentage = (
            round((missing_count / row_count) * 100, 4)
            if row_count > 0
            else 0.0
        )

        unique_count = int(series.nunique(dropna=True))

        unique_percentage = (
            round((unique_count / row_count) * 100, 4)
            if row_count > 0
            else 0.0
        )

        dtype = str(series.dtype)

        semantic_type = self._detect_semantic_type(
            column_name,
            series,
        )

        potential_id = self._is_potential_id(
            column_name,
            series,
            unique_count,
        )

        target_score, target_reasons = (
            self._calculate_target_score(
                column_name,
                series,
                unique_count,
                potential_id,
            )
        )

        signature = {
            "name": column_name,
            "normalized_name": self._normalize_column_name(
                column_name
            ),
            "dtype": dtype,
            "semantic_type": semantic_type,
            "unique_count": unique_count,
            "unique_percentage": unique_percentage,
            "missing_count": missing_count,
            "missing_percentage": missing_percentage,
            "potential_id": potential_id,
            "target_candidate": target_score >= 60,
            "target_score": target_score,
            "target_reasons": target_reasons,
        }

        numeric_statistics = (
            self._get_numeric_statistics(series)
        )

        if numeric_statistics is not None:
            signature["numeric_statistics"] = (
                numeric_statistics
            )

        categorical_statistics = (
            self._get_categorical_statistics(series)
        )

        if categorical_statistics is not None:
            signature["categorical_statistics"] = (
                categorical_statistics
            )

        return signature

    def _build_characteristics(
        self,
        columns: list,
    ) -> dict:

        characteristics = {
            "numeric_columns": 0,
            "categorical_columns": 0,
            "datetime_columns": 0,
            "text_columns": 0,
            "boolean_columns": 0,
            "potential_id_columns": 0,
        }

        for column in columns:
            semantic_type = column["semantic_type"]

            if semantic_type == "numeric":
                characteristics["numeric_columns"] += 1

            elif semantic_type == "categorical":
                characteristics["categorical_columns"] += 1

            elif semantic_type == "datetime":
                characteristics["datetime_columns"] += 1

            elif semantic_type == "text":
                characteristics["text_columns"] += 1

            elif semantic_type == "boolean":
                characteristics["boolean_columns"] += 1

            if column["potential_id"]:
                characteristics["potential_id_columns"] += 1

        return characteristics

    def _extract_keywords(
        self,
        dataframe: pd.DataFrame,
    ) -> list:

        keywords = set()

        stop_words = {
            "id",
            "no",
            "num",
            "number",
            "value",
            "data",
            "column",
            "field",
            "the",
            "of",
            "and",
        }

        for column in dataframe.columns:

            normalized = self._normalize_column_name(
                str(column)
            )

            parts = normalized.split("_")

            for part in parts:

                if not part:
                    continue

                if part in stop_words:
                    continue

                if len(part) < 2:
                    continue

                keywords.add(part)

        return sorted(keywords)

    def _build_categories(
        self,
        columns: list,
    ) -> list:

        categories = set()

        semantic_types = {
            column["semantic_type"]
            for column in columns
        }

        if "numeric" in semantic_types:
            categories.add("numeric")

        if "categorical" in semantic_types:
            categories.add("categorical")

        if "text" in semantic_types:
            categories.add("text")

        if "datetime" in semantic_types:
            categories.add("time_related")

        if "boolean" in semantic_types:
            categories.add("boolean")

        return sorted(categories)

    def _normalize_column_name(
        self,
        column_name: str,
    ) -> str:

        normalized = column_name.strip().lower()

        normalized = re.sub(
            r"[^a-z0-9]+",
            "_",
            normalized,
        )

        normalized = re.sub(
            r"_+",
            "_",
            normalized,
        )

        return normalized.strip("_")

    def _detect_semantic_type(
        self,
        column_name: str,
        series: pd.Series,
    ) -> str:

        if pd.api.types.is_bool_dtype(series):
            return "boolean"

        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        if pd.api.types.is_numeric_dtype(series):

            # Numeric columns with very low cardinality
            # can behave like categorical features.
            unique_count = series.nunique(
                dropna=True
            )

            if unique_count <= 10:
                return "categorical"

            return "numeric"

        non_null = series.dropna()

        if len(non_null) == 0:
            return "text"

        unique_count = non_null.nunique()

        normalized_name = self._normalize_column_name(
            column_name
        )

        categorical_keywords = {
            "sex",
            "gender",
            "class",
            "category",
            "type",
            "status",
            "state",
            "country",
            "city",
            "department",
            "embarked",
            "group",
        }

        if normalized_name in categorical_keywords:
            return "categorical"

        if unique_count <= 20:
            return "categorical"

        average_length = (
            non_null.astype(str)
            .str.len()
            .mean()
        )

        if average_length >= 30:
            return "text"

        return "text"

    def _is_potential_id(
        self,
        column_name: str,
        series: pd.Series,
        unique_count: int,
    ) -> bool:

        row_count = len(series)

        if row_count == 0:
            return False

        normalized_name = self._normalize_column_name(
            column_name
        )

        id_keywords = {
            "id",
            "user_id",
            "customer_id",
            "student_id",
            "employee_id",
            "patient_id",
            "record_id",
            "transaction_id",
            "uuid",
            "identifier",
            "passengerid",
        }

        if normalized_name in id_keywords:
            return True

        if (
            unique_count == row_count
            and row_count > 10
        ):
            return True

        return False

    def _calculate_target_score(
        self,
        column_name: str,
        series: pd.Series,
        unique_count: int,
        potential_id: bool,
    ) -> tuple[int, list]:

        score = 0
        reasons = []

        normalized_name = self._normalize_column_name(
            column_name
        )

        target_keywords = {
            "target",
            "label",
            "class",
            "outcome",
            "result",
            "response",
            "prediction",
            "diagnosis",
            "price",
            "sales",
            "score",
            "churn",
            "default",
            "survived",
        }

        # --------------------------------------------------
        # 1. Target-like column name
        # --------------------------------------------------

        if normalized_name in target_keywords:

            score += 60

            reasons.append(
                "Nama kolom target-like."
            )

        elif any(
            normalized_name.startswith(
                keyword + "_"
            )
            for keyword in target_keywords
        ):

            score += 50

            reasons.append(
                "Nama kolom memiliki pola target-like."
            )

        elif any(
            normalized_name.endswith(
                "_" + keyword
            )
            for keyword in target_keywords
        ):

            score += 50

            reasons.append(
                "Nama kolom memiliki pola target-like."
            )

        # --------------------------------------------------
        # 2. Low cardinality
        # --------------------------------------------------

        row_count = len(series)

        if row_count > 0:

            unique_ratio = (
                unique_count / row_count
            )

            if unique_count == 2:

                score += 25

                reasons.append(
                    "Memiliki dua nilai unik, "
                    "cocok untuk binary classification."
                )

            elif (
                3 <= unique_count <= 10
                and unique_ratio <= 0.05
            ):

                score += 15

                reasons.append(
                    "Memiliki jumlah kategori rendah."
                )

        # --------------------------------------------------
        # 3. Exclude obvious ID
        # --------------------------------------------------

        if potential_id:

            score -= 70

            reasons.append(
                "Terindikasi sebagai ID, sehingga "
                "tidak diprioritaskan sebagai target."
            )

        # --------------------------------------------------
        # 4. Avoid constant columns
        # --------------------------------------------------

        if unique_count <= 1:

            score -= 100

            reasons.append(
                "Kolom konstan tidak cocok sebagai target."
            )

        score = max(0, min(score, 100))

        return score, reasons

    def _get_numeric_statistics(
        self,
        series: pd.Series,
    ) -> dict | None:

        if not pd.api.types.is_numeric_dtype(series):
            return None

        clean_series = series.dropna()

        if clean_series.empty:
            return None

        return {
            "min": self._safe_number(
                clean_series.min()
            ),
            "max": self._safe_number(
                clean_series.max()
            ),
            "mean": self._safe_number(
                clean_series.mean()
            ),
            "median": self._safe_number(
                clean_series.median()
            ),
            "std": self._safe_number(
                clean_series.std()
            ),
        }

    def _get_categorical_statistics(
        self,
        series: pd.Series,
    ) -> dict | None:

        if not (
            isinstance(
                series.dtype,
                pd.CategoricalDtype,
            )
            or pd.api.types.is_object_dtype(series)
            or pd.api.types.is_string_dtype(series)
        ):
            return None

        non_null = series.dropna()

        if non_null.empty:
            return None

        unique_count = non_null.nunique()

        if unique_count > 20:
            return None

        value_counts = (
            non_null.astype(str)
            .value_counts()
        )

        top_values = []

        for value, count in (
            value_counts.head(5).items()
        ):

            top_values.append(
                {
                    "value": value,
                    "count": int(count),
                }
            )

        return {
            "unique_count": int(unique_count),
            "top_values": top_values,
        }

    def _safe_number(
        self,
        value,
    ):

        if pd.isna(value):
            return None

        if hasattr(value, "item"):
            value = value.item()

        if isinstance(value, float):
            return round(value, 6)

        return value

    def generate_hash(
        self,
        representation: dict,
    ) -> str:

        normalized = json.dumps(
            representation,
            sort_keys=True,
            ensure_ascii=False,
        )

        return hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()

    def generate(
        self,
        dataframe: pd.DataFrame,
    ) -> dict:

        representation = (
            self.generate_representation(
                dataframe
            )
        )

        fingerprint = self.generate_hash(
            representation
        )

        return {
            "representation": representation,
            "fingerprint": fingerprint,
        }