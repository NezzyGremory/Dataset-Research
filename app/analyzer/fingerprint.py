import hashlib
import json
import re

import pandas as pd


class DatasetFingerprint:
    """
    Generate a normalized fingerprint representation for a dataset.

    Fingerprint digunakan untuk merangkum karakteristik dataset
    dan menjadi identitas heuristik dataset.

    Catatan:
    - Fingerprint bukan identitas absolut dataset.
    - Target detection bersifat heuristik.
    - Semantic type detection digunakan untuk membantu
      proses ML task detection dan dataset similarity.
    """

    # ==========================================================
    # MAIN
    # ==========================================================

    def generate_representation(
        self,
        dataframe: pd.DataFrame,
    ) -> dict:
        """
        Generate normalized dataset representation.
        """

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

        characteristics = self._build_characteristics(
            columns
        )

        target_candidates = [
            column["name"]
            for column in columns
            if column["target_candidate"]
        ]

        keywords = self._extract_keywords(
            dataframe
        )

        categories = self._build_categories(
            columns
        )

        return {
            "rows": rows,
            "columns": columns_count,
            "characteristics": characteristics,
            "target_candidates": target_candidates,
            "keywords": keywords,
            "categories": categories,
            "column_signature": columns,
        }

    # ==========================================================
    # COLUMN ANALYSIS
    # ==========================================================

    def _analyze_column(
        self,
        column,
        series: pd.Series,
    ) -> dict:
        """
        Analyze one dataset column.
        """

        column_name = str(column)
        row_count = len(series)

        missing_count = int(
            series.isna().sum()
        )

        missing_percentage = (
            round(
                (missing_count / row_count) * 100,
                4,
            )
            if row_count > 0
            else 0.0
        )

        unique_count = int(
            series.nunique(
                dropna=True
            )
        )

        unique_percentage = (
            round(
                (unique_count / row_count) * 100,
                4,
            )
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
            "normalized_name": (
                self._normalize_column_name(
                    column_name
                )
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
            self._get_numeric_statistics(
                series
            )
        )

        if numeric_statistics is not None:
            signature[
                "numeric_statistics"
            ] = numeric_statistics

        categorical_statistics = (
            self._get_categorical_statistics(
                series
            )
        )

        if categorical_statistics is not None:
            signature[
                "categorical_statistics"
            ] = categorical_statistics

        return signature

    # ==========================================================
    # CHARACTERISTICS
    # ==========================================================

    def _build_characteristics(
        self,
        columns: list,
    ) -> dict:
        """
        Build dataset-level characteristics.
        """

        characteristics = {
            "numeric_columns": 0,
            "categorical_columns": 0,
            "datetime_columns": 0,
            "text_columns": 0,
            "boolean_columns": 0,
            "potential_id_columns": 0,
        }

        for column in columns:

            semantic_type = column[
                "semantic_type"
            ]

            if semantic_type == "numeric":
                characteristics[
                    "numeric_columns"
                ] += 1

            elif semantic_type == "categorical":
                characteristics[
                    "categorical_columns"
                ] += 1

            elif semantic_type == "datetime":
                characteristics[
                    "datetime_columns"
                ] += 1

            elif semantic_type == "text":
                characteristics[
                    "text_columns"
                ] += 1

            elif semantic_type == "boolean":
                characteristics[
                    "boolean_columns"
                ] += 1

            if column["potential_id"]:
                characteristics[
                    "potential_id_columns"
                ] += 1

        return characteristics

    # ==========================================================
    # KEYWORDS
    # ==========================================================

    def _extract_keywords(
        self,
        dataframe: pd.DataFrame,
    ) -> list:
        """
        Extract basic keywords from column names.

        Keyword extraction di sini masih bersifat lightweight.
        NLP yang lebih kompleks dilakukan di module nlp/.
        """

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

            normalized = (
                self._normalize_column_name(
                    str(column)
                )
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

    # ==========================================================
    # CATEGORIES
    # ==========================================================

    def _build_categories(
        self,
        columns: list,
    ) -> list:
        """
        Build high-level dataset categories
        from semantic column types.
        """

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

    # ==========================================================
    # COLUMN NAME NORMALIZATION
    # ==========================================================

    def _normalize_column_name(
        self,
        column_name: str,
    ) -> str:
        """
        Normalize column name into a comparable format.
        """

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

    # ==========================================================
    # SEMANTIC TYPE DETECTION
    # ==========================================================

    def _detect_semantic_type(
        self,
        column_name: str,
        series: pd.Series,
    ) -> str:
        """
        Detect semantic type of a column.

        Important:
        Numeric columns are treated as numeric by default.

        Numeric columns are NOT automatically converted into
        categorical simply because they have a small number
        of unique values.

        Example:
            Pclass -> categorical
            Price  -> numeric
            Age    -> numeric
            Rating -> numeric

        Certain column names strongly indicate categorical
        meaning and are therefore handled as categorical.
        """

        normalized_name = (
            self._normalize_column_name(
                column_name
            )
        )

        # --------------------------------------------------
        # 1. BOOLEAN
        # --------------------------------------------------

        if pd.api.types.is_bool_dtype(series):
            return "boolean"

        # --------------------------------------------------
        # 2. DATETIME
        # --------------------------------------------------

        if pd.api.types.is_datetime64_any_dtype(
            series
        ):
            return "datetime"

        # --------------------------------------------------
        # 3. NUMERIC
        # --------------------------------------------------

        if pd.api.types.is_numeric_dtype(
            series
        ):

            # Numeric columns that semantically represent
            # categories.
            categorical_numeric_keywords = {
                "class",
                "category",
                "type",
                "status",
                "group",
                "level",
                "rank",
                "grade",
                "segment",
                "tier",
                "code",
            }

            if normalized_name in (
                categorical_numeric_keywords
            ):
                return "categorical"

            return "numeric"

        # --------------------------------------------------
        # 4. EMPTY / ALL NULL
        # --------------------------------------------------

        non_null = series.dropna()

        if len(non_null) == 0:
            return "text"

        # --------------------------------------------------
        # 5. NON-NUMERIC CATEGORICAL
        # --------------------------------------------------

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
            "level",
            "grade",
            "segment",
            "tier",
        }

        if normalized_name in (
            categorical_keywords
        ):
            return "categorical"

        unique_count = int(
            non_null.nunique()
        )

        # --------------------------------------------------
        # 6. LOW CARDINALITY STRING
        # --------------------------------------------------

        if unique_count <= 20:
            return "categorical"

        # --------------------------------------------------
        # 7. TEXT
        # --------------------------------------------------

        average_length = (
            non_null.astype(str)
            .str.len()
            .mean()
        )

        if average_length >= 30:
            return "text"

        return "text"

    # ==========================================================
    # POTENTIAL ID DETECTION
    # ==========================================================

    def _is_potential_id(
        self,
        column_name: str,
        series: pd.Series,
        unique_count: int,
    ) -> bool:
        """
        Detect whether a column is likely an identifier.
        """

        row_count = len(series)

        if row_count == 0:
            return False

        normalized_name = (
            self._normalize_column_name(
                column_name
            )
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
            "passenger_id",
            "account_id",
            "order_id",
            "product_id",
            "item_id",
            "invoice_id",
            "ticket_id",
        }

        # Explicit ID-like names
        if normalized_name in id_keywords:
            return True

        # Exact unique sequence is suspicious as ID
        if (
            unique_count == row_count
            and row_count > 10
        ):
            return True

        return False

    # ==========================================================
    # TARGET DETECTION
    # ==========================================================

    def _calculate_target_score(
        self,
        column_name: str,
        series: pd.Series,
        unique_count: int,
        potential_id: bool,
    ) -> tuple[int, list]:
        """
        Estimate whether a column is a possible ML target.

        This is a heuristic score, NOT a guarantee.
        """

        score = 0
        reasons = []

        normalized_name = (
            self._normalize_column_name(
                column_name
            )
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
            "y",
            "target_value",
            "target_class",
        }

        # --------------------------------------------------
        # 1. TARGET-LIKE NAME
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
            if keyword not in {
                "y",
            }
        ):

            score += 50

            reasons.append(
                "Nama kolom memiliki pola "
                "target-like."
            )

        elif any(
            normalized_name.endswith(
                "_" + keyword
            )
            for keyword in target_keywords
            if keyword not in {
                "y",
            }
        ):

            score += 50

            reasons.append(
                "Nama kolom memiliki pola "
                "target-like."
            )

        # --------------------------------------------------
        # 2. LOW CARDINALITY
        # --------------------------------------------------

        row_count = len(series)

        if row_count > 0:

            unique_ratio = (
                unique_count / row_count
            )

            # Binary classification candidate
            if unique_count == 2:

                score += 25

                reasons.append(
                    "Memiliki dua nilai unik, "
                    "cocok untuk binary classification."
                )

            # Multiclass classification candidate
            elif (
                3 <= unique_count <= 10
                and unique_ratio <= 0.05
            ):

                score += 15

                reasons.append(
                    "Memiliki jumlah kategori rendah."
                )

        # --------------------------------------------------
        # 3. EXCLUDE OBVIOUS ID
        # --------------------------------------------------

        if potential_id:

            score -= 70

            reasons.append(
                "Terindikasi sebagai ID, sehingga "
                "tidak diprioritaskan sebagai target."
            )

        # --------------------------------------------------
        # 4. CONSTANT COLUMN
        # --------------------------------------------------

        if unique_count <= 1:

            score -= 100

            reasons.append(
                "Kolom konstan tidak cocok "
                "sebagai target."
            )

        # --------------------------------------------------
        # 5. NUMERIC TARGET SUPPORT
        # --------------------------------------------------

        if pd.api.types.is_numeric_dtype(
            series
        ):

            # Numeric columns with enough variation
            # can be regression targets.
            if unique_count > 10:

                score += 10

                reasons.append(
                    "Kolom numerik dengan variasi "
                    "nilai cukup untuk kandidat regression."
                )

        score = max(
            0,
            min(score, 100),
        )

        return score, reasons

    # ==========================================================
    # NUMERIC STATISTICS
    # ==========================================================

    def _get_numeric_statistics(
        self,
        series: pd.Series,
    ) -> dict | None:
        """
        Generate numeric statistics.
        """

        if not pd.api.types.is_numeric_dtype(
            series
        ):
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

    # ==========================================================
    # CATEGORICAL STATISTICS
    # ==========================================================

    def _get_categorical_statistics(
        self,
        series: pd.Series,
    ) -> dict | None:
        """
        Generate categorical statistics
        for low-cardinality non-numeric columns.
        """

        if not (
            isinstance(
                series.dtype,
                pd.CategoricalDtype,
            )
            or pd.api.types.is_object_dtype(
                series
            )
            or pd.api.types.is_string_dtype(
                series
            )
        ):
            return None

        non_null = series.dropna()

        if non_null.empty:
            return None

        unique_count = int(
            non_null.nunique()
        )

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
            "unique_count": unique_count,
            "top_values": top_values,
        }

    # ==========================================================
    # SAFE NUMBER
    # ==========================================================

    def _safe_number(
        self,
        value,
    ):
        """
        Convert pandas/numpy numeric values
        into JSON-safe Python values.
        """

        if pd.isna(value):
            return None

        if hasattr(value, "item"):
            value = value.item()

        if isinstance(
            value,
            float,
        ):
            return round(
                value,
                6,
            )

        return value

    # ==========================================================
    # HASH
    # ==========================================================

    def generate_hash(
        self,
        representation: dict,
    ) -> str:
        """
        Generate SHA-256 hash from normalized representation.

        Hash digunakan sebagai identitas heuristik,
        bukan sebagai similarity score.
        """

        normalized = json.dumps(
            representation,
            sort_keys=True,
            ensure_ascii=False,
        )

        return hashlib.sha256(
            normalized.encode(
                "utf-8"
            )
        ).hexdigest()

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def generate(
        self,
        dataframe: pd.DataFrame,
    ) -> dict:
        """
        Generate complete fingerprint.
        """

        representation = (
            self.generate_representation(
                dataframe
            )
        )

        fingerprint = (
            self.generate_hash(
                representation
            )
        )

        return {
            "representation": representation,
            "fingerprint": fingerprint,
        }