from __future__ import annotations

from typing import Any, Dict, List


class ResearchQueryBuilder:
    """
    Research Query Builder.

    Membuat query akademik berdasarkan:

    - keyword dataset
    - research domain
    - target
    - ML task
    - recommended ML methods
    - nama kolom dataset

    Semua query bersifat rekomendasi/heuristik.

    Builder dibuat defensif agar tetap dapat menghasilkan query
    walaupun sebagian informasi dari pipeline tidak tersedia.
    """

    def build(
        self,
        fingerprint: Dict,
        domain_result: Dict | None = None,
        ml_result: Dict | None = None,
        keyword_result: Dict | None = None,
        max_queries: int = 5,
    ) -> List[str]:

        max_queries = max(
            1,
            int(max_queries),
        )

        fingerprint = (
            fingerprint
            if isinstance(fingerprint, dict)
            else {}
        )

        domain_result = (
            domain_result
            if isinstance(domain_result, dict)
            else {}
        )

        ml_result = (
            ml_result
            if isinstance(ml_result, dict)
            else {}
        )

        keyword_result = (
            keyword_result
            if isinstance(keyword_result, dict)
            else {}
        )

        # =====================================================
        # EXTRACT INFORMATION
        # =====================================================

        keywords = self._keywords(
            fingerprint=fingerprint,
            keyword_result=keyword_result,
        )

        domain = self._domain(
            domain_result
        )

        target = self._target(
            ml_result=ml_result,
            fingerprint=fingerprint,
        )

        task = self._task(
            ml_result
        )

        methods = self._methods(
            ml_result
        )

        columns = self._columns(
            fingerprint
        )

        queries: List[str] = []

        # =====================================================
        # 1. KEYWORDS + DOMAIN
        # =====================================================

        if keywords:

            parts = keywords[:5]

            if domain:
                parts.append(
                    domain
                )

            self._add_query(
                queries,
                " ".join(parts),
            )

        # =====================================================
        # 2. TARGET + PREDICTION
        # =====================================================

        if target:

            if keywords:

                self._add_query(
                    queries,
                    (
                        f"{target} prediction "
                        f"{' '.join(keywords[:3])}"
                    ),
                )

            else:

                self._add_query(
                    queries,
                    f"{target} prediction",
                )

        # =====================================================
        # 3. TASK + KEYWORDS
        # =====================================================

        if task:

            task_label = self._task_label(
                task
            )

            if keywords:

                self._add_query(
                    queries,
                    (
                        f"{task_label} "
                        f"{' '.join(keywords[:3])}"
                    ),
                )

            else:

                self._add_query(
                    queries,
                    task_label,
                )

        # =====================================================
        # 4. ML METHODS + KEYWORDS
        # =====================================================

        if methods:

            for method in methods:

                if keywords:

                    self._add_query(
                        queries,
                        (
                            f"{method} "
                            f"{' '.join(keywords[:3])}"
                        ),
                    )

                else:

                    self._add_query(
                        queries,
                        method,
                    )

                if len(queries) >= max_queries:
                    break

        # =====================================================
        # 5. DOMAIN + TARGET
        # =====================================================

        if domain and target:

            self._add_query(
                queries,
                (
                    f"{domain} "
                    f"{target} prediction"
                ),
            )

        # =====================================================
        # 6. DOMAIN + TASK
        # =====================================================

        if domain and task:

            self._add_query(
                queries,
                (
                    f"{domain} "
                    f"{self._task_label(task)}"
                ),
            )

        # =====================================================
        # 7. KEYWORDS SAJA
        # =====================================================

        if keywords:

            self._add_query(
                queries,
                " ".join(
                    keywords[:5]
                ),
            )

        # =====================================================
        # 8. COLUMN-BASED FALLBACK
        # =====================================================

        if columns:

            self._add_query(
                queries,
                " ".join(
                    columns[:6]
                ),
            )

        # =====================================================
        # 9. DOMAIN SAJA
        # =====================================================

        if domain:

            self._add_query(
                queries,
                domain,
            )

        # =====================================================
        # 10. LAST RESORT
        # =====================================================

        if not queries:

            fallback = (
                self._dataset_name(
                    fingerprint
                )
            )

            if fallback:

                self._add_query(
                    queries,
                    fallback,
                )

        return queries[:max_queries]

    # =========================================================
    # ADD QUERY
    # =========================================================

    @staticmethod
    def _add_query(
        queries: List[str],
        query: Any,
    ) -> None:

        if query is None:
            return

        query = str(
            query
        ).strip()

        if not query:
            return

        # Bersihkan whitespace.
        query = " ".join(
            query.split()
        )

        # Hindari query terlalu pendek.
        if len(query) < 2:
            return

        normalized = query.lower()

        for existing in queries:

            if existing.lower() == normalized:
                return

        queries.append(
            query
        )

    # =========================================================
    # KEYWORDS
    # =========================================================

    @staticmethod
    def _keywords(
        fingerprint: Dict,
        keyword_result: Dict | None = None,
    ) -> List[str]:

        candidates: List[Any] = []

        # -----------------------------------------------------
        # PRIORITAS 1:
        # KeywordExtractor
        # -----------------------------------------------------

        if keyword_result:

            extracted = (
                keyword_result.get(
                    "keywords",
                    [],
                )
            )

            candidates.extend(
                ResearchQueryBuilder._as_list(
                    extracted
                )
            )

            # Beberapa implementasi mungkin menggunakan
            # extracted_keywords.
            if not candidates:

                extracted = (
                    keyword_result.get(
                        "extracted_keywords",
                        [],
                    )
                )

                candidates.extend(
                    ResearchQueryBuilder._as_list(
                        extracted
                    )
                )

        # -----------------------------------------------------
        # PRIORITAS 2:
        # fingerprint.representation
        # -----------------------------------------------------

        if not candidates:

            representation = (
                fingerprint.get(
                    "representation",
                    fingerprint,
                )
            )

            if isinstance(
                representation,
                dict,
            ):

                extracted = (
                    representation.get(
                        "keywords",
                        [],
                    )
                )

                candidates.extend(
                    ResearchQueryBuilder._as_list(
                        extracted
                    )
                )

        # -----------------------------------------------------
        # PRIORITAS 3:
        # fingerprint langsung
        # -----------------------------------------------------

        if not candidates:

            extracted = (
                fingerprint.get(
                    "keywords",
                    [],
                )
            )

            candidates.extend(
                ResearchQueryBuilder._as_list(
                    extracted
                )
            )

        # -----------------------------------------------------
        # NORMALIZE
        # -----------------------------------------------------

        result: List[str] = []

        seen = set()

        for keyword in candidates:

            if isinstance(
                keyword,
                dict,
            ):

                value = (
                    keyword.get("keyword")
                    or keyword.get("term")
                    or keyword.get("name")
                    or keyword.get("text")
                )

            else:

                value = keyword

            if value is None:
                continue

            value = str(
                value
            ).strip()

            value = " ".join(
                value.split()
            )

            if not value:
                continue

            normalized = value.lower()

            if normalized in seen:
                continue

            seen.add(
                normalized
            )

            result.append(
                value
            )

        return result

    # =========================================================
    # DOMAIN
    # =========================================================

    @staticmethod
    def _domain(
        domain_result: Dict | None,
    ) -> str | None:

        if not domain_result:
            return None

        domain = (
            domain_result.get(
                "primary_domain"
            )
            or domain_result.get(
                "domain"
            )
            or domain_result.get(
                "predicted_domain"
            )
            or domain_result.get(
                "research_domain"
            )
        )

        if isinstance(
            domain,
            dict,
        ):

            domain = (
                domain.get("name")
                or domain.get("label")
                or domain.get("domain")
            )

        if not domain:
            return None

        return str(
            domain
        ).strip()

    # =========================================================
    # TARGET
    # =========================================================

    @staticmethod
    def _target(
        ml_result: Dict | None,
        fingerprint: Dict,
    ) -> str | None:

        # -----------------------------------------------------
        # PRIORITAS 1:
        # ml_result target langsung
        # -----------------------------------------------------

        if ml_result:

            target = (
                ml_result.get(
                    "target"
                )
                or ml_result.get(
                    "target_name"
                )
                or ml_result.get(
                    "target_column"
                )
            )

            if target:

                if isinstance(
                    target,
                    dict,
                ):

                    target = (
                        target.get("name")
                        or target.get("column")
                    )

                if target:

                    return str(
                        target
                    ).strip()

        # -----------------------------------------------------
        # PRIORITAS 2:
        # primary_task
        # -----------------------------------------------------

        if ml_result:

            primary = (
                ml_result.get(
                    "primary_task"
                )
            )

            if isinstance(
                primary,
                dict,
            ):

                target = (
                    primary.get(
                        "target"
                    )
                    or primary.get(
                        "target_name"
                    )
                    or primary.get(
                        "target_column"
                    )
                )

                if target:

                    return str(
                        target
                    ).strip()

        # -----------------------------------------------------
        # PRIORITAS 3:
        # fingerprint representation
        # -----------------------------------------------------

        representation = (
            fingerprint.get(
                "representation",
                fingerprint,
            )
        )

        if not isinstance(
            representation,
            dict,
        ):
            representation = {}

        candidates = (
            representation.get(
                "target_candidates",
                [],
            )
        )

        if isinstance(
            candidates,
            dict,
        ):

            candidates = [
                candidates
            ]

        if isinstance(
            candidates,
            list,
        ):

            for candidate in candidates:

                if isinstance(
                    candidate,
                    dict,
                ):

                    name = (
                        candidate.get(
                            "name"
                        )
                        or candidate.get(
                            "column"
                        )
                        or candidate.get(
                            "target"
                        )
                    )

                    if name:

                        return str(
                            name
                        ).strip()

                elif candidate:

                    value = str(
                        candidate
                    ).strip()

                    if value:
                        return value

        return None

    # =========================================================
    # TASK
    # =========================================================

    @staticmethod
    def _task(
        ml_result: Dict | None,
    ) -> str | None:

        if not ml_result:
            return None

        # -----------------------------------------------------
        # primary_task
        # -----------------------------------------------------

        primary = (
            ml_result.get(
                "primary_task"
            )
        )

        if isinstance(
            primary,
            dict,
        ):

            task = (
                primary.get(
                    "task"
                )
                or primary.get(
                    "type"
                )
                or primary.get(
                    "name"
                )
            )

            if task:

                return str(
                    task
                ).strip()

        elif isinstance(
            primary,
            str,
        ):

            return primary.strip()

        # -----------------------------------------------------
        # task langsung
        # -----------------------------------------------------

        task = (
            ml_result.get(
                "task"
            )
            or ml_result.get(
                "task_type"
            )
            or ml_result.get(
                "ml_task"
            )
        )

        if task:

            return str(
                task
            ).strip()

        return None

    # =========================================================
    # TASK LABEL
    # =========================================================

    @staticmethod
    def _task_label(
        task: str,
    ) -> str:

        normalized = (
            str(task)
            .strip()
            .lower()
            .replace(
                " ",
                "_",
            )
            .replace(
                "-",
                "_",
            )
        )

        labels = {

            "binary_classification":
                "binary classification",

            "multiclass_classification":
                "multiclass classification",

            "multi_class_classification":
                "multiclass classification",

            "classification":
                "classification",

            "regression":
                "regression",

            "clustering":
                "clustering",

            "anomaly_detection":
                "anomaly detection",

            "anomaly":
                "anomaly detection",

            "time_series_forecasting":
                "time series forecasting",

            "forecasting":
                "time series forecasting",
        }

        return labels.get(
            normalized,
            str(task).replace(
                "_",
                " ",
            ),
        )

    # =========================================================
    # METHODS
    # =========================================================

    @staticmethod
    def _methods(
        ml_result: Dict | None,
    ) -> List[str]:

        if not ml_result:
            return []

        # -----------------------------------------------------
        # recommendation
        # -----------------------------------------------------

        recommendation = (
            ml_result.get(
                "recommendation",
                {},
            )
        )

        if not isinstance(
            recommendation,
            dict,
        ):

            recommendation = {}

        recommendations = (
            recommendation.get(
                "recommendations",
                [],
            )
        )

        # -----------------------------------------------------
        # fallback key
        # -----------------------------------------------------

        if not recommendations:

            recommendations = (
                ml_result.get(
                    "recommendations",
                    [],
                )
            )

        if not isinstance(
            recommendations,
            list,
        ):
            return []

        methods: List[str] = []

        for item in recommendations[:5]:

            if isinstance(
                item,
                dict,
            ):

                method = (
                    item.get(
                        "method"
                    )
                    or item.get(
                        "method_id"
                    )
                    or item.get(
                        "name"
                    )
                )

            else:

                method = item

            if not method:
                continue

            value = str(
                method
            ).strip()

            if not value:
                continue

            if value.lower() in {
                m.lower()
                for m in methods
            }:
                continue

            methods.append(
                value
            )

        return methods

    # =========================================================
    # COLUMNS
    # =========================================================

    @staticmethod
    def _columns(
        fingerprint: Dict,
    ) -> List[str]:

        candidates = []

        # -----------------------------------------------------
        # fingerprint.columns
        # -----------------------------------------------------

        columns = (
            fingerprint.get(
                "columns",
                [],
            )
        )

        if isinstance(
            columns,
            list,
        ):

            candidates.extend(
                columns
            )

        # -----------------------------------------------------
        # representation.columns
        # -----------------------------------------------------

        if not candidates:

            representation = (
                fingerprint.get(
                    "representation",
                    {},
                )
            )

            if isinstance(
                representation,
                dict,
            ):

                columns = (
                    representation.get(
                        "columns",
                        [],
                    )
                )

                if isinstance(
                    columns,
                    list,
                ):

                    candidates.extend(
                        columns
                    )

        result = []

        seen = set()

        for column in candidates:

            if isinstance(
                column,
                dict,
            ):

                name = (
                    column.get(
                        "name"
                    )
                    or column.get(
                        "column"
                    )
                )

            else:

                name = column

            if not name:
                continue

            value = str(
                name
            ).strip()

            if not value:
                continue

            normalized = value.lower()

            if normalized in seen:
                continue

            seen.add(
                normalized
            )

            result.append(
                value
            )

        return result

    # =========================================================
    # DATASET NAME
    # =========================================================

    @staticmethod
    def _dataset_name(
        fingerprint: Dict,
    ) -> str | None:

        if not isinstance(
            fingerprint,
            dict,
        ):
            return None

        candidates = [
            fingerprint.get(
                "dataset_name"
            ),
            fingerprint.get(
                "name"
            ),
            fingerprint.get(
                "filename"
            ),
            fingerprint.get(
                "file_name"
            ),
            fingerprint.get(
                "title"
            ),
        ]

        representation = (
            fingerprint.get(
                "representation",
                {},
            )
        )

        if isinstance(
            representation,
            dict,
        ):

            candidates.extend(
                [
                    representation.get(
                        "dataset_name"
                    ),
                    representation.get(
                        "name"
                    ),
                    representation.get(
                        "filename"
                    ),
                    representation.get(
                        "title"
                    ),
                ]
            )

        for value in candidates:

            if not value:
                continue

            value = str(
                value
            ).strip()

            if not value:
                continue

            return value

        return None

    # =========================================================
    # AS LIST
    # =========================================================

    @staticmethod
    def _as_list(
        value,
    ) -> List[Any]:

        if value is None:
            return []

        if isinstance(
            value,
            str,
        ):

            return [
                value
            ]

        if isinstance(
            value,
            dict,
        ):

            return list(
                value.keys()
            )

        if isinstance(
            value,
            (list, tuple, set),
        ):

            return list(
                value
            )

        return []