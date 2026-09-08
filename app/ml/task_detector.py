from __future__ import annotations

from typing import Any


class MLTaskDetector:
    """
    Mendeteksi kemungkinan jenis machine learning berdasarkan
    fingerprint dataset.

    Semua hasil bersifat ESTIMATION / HEURISTIC.
    Detector tidak mengklaim bahwa task yang dipilih adalah
    satu-satunya task yang benar.
    """

    STATUS = "ESTIMATION"

    TASK_LABELS = {
        "binary_classification": "Binary Classification",
        "multiclass_classification": "Multiclass Classification",
        "regression": "Regression",
        "clustering": "Clustering",
        "anomaly_detection": "Anomaly Detection",
        "time_series_forecasting": "Time-Series Forecasting",
    }

    def detect(self, fingerprint: dict[str, Any]) -> dict[str, Any]:
        """
        Mendeteksi kemungkinan task ML dari fingerprint.

        Parameters
        ----------
        fingerprint:
            Hasil DatasetFingerprint.generate().

        Returns
        -------
        dict
            Hasil deteksi task ML.
        """

        representation = fingerprint.get("representation", fingerprint)

        columns = representation.get("column_signature", [])
        target_candidates = representation.get("target_candidates", [])
        characteristics = representation.get("characteristics", {})

        # Normalisasi struktur agar detector tahan terhadap
        # beberapa variasi format fingerprint.
        columns = self._normalize_columns(columns)
        target_candidates = self._normalize_target_candidates(
            target_candidates
        )

        tasks: list[dict[str, Any]] = []

        # ---------------------------------------------------------
        # 1. SUPERVISED LEARNING
        # ---------------------------------------------------------

        target_result = self._detect_supervised_task(
            columns=columns,
            target_candidates=target_candidates,
        )

        if target_result:
            tasks.append(target_result)

        # ---------------------------------------------------------
        # 2. UNSUPERVISED LEARNING
        # ---------------------------------------------------------

        numeric_count = self._get_numeric_count(
            columns,
            characteristics,
        )

        datetime_count = self._get_datetime_count(
            columns,
            characteristics,
        )

        if numeric_count >= 2:
            tasks.append(
                self._make_task(
                    task="clustering",
                    score=65,
                    target=None,
                    reasons=[
                        f"Dataset memiliki {numeric_count} fitur numerik.",
                        "Clustering dapat dilakukan tanpa target.",
                    ],
                )
            )

            tasks.append(
                self._make_task(
                    task="anomaly_detection",
                    score=55,
                    target=None,
                    reasons=[
                        f"Dataset memiliki {numeric_count} fitur numerik.",
                        (
                            "Struktur data memungkinkan deteksi "
                            "observasi yang tidak umum."
                        ),
                    ],
                )
            )

        # ---------------------------------------------------------
        # 3. TIME SERIES
        # ---------------------------------------------------------

        if datetime_count >= 1:
            tasks.append(
                self._make_task(
                    task="time_series_forecasting",
                    score=70,
                    target=None,
                    reasons=[
                        "Dataset memiliki kolom bertipe datetime.",
                        (
                            "Struktur dataset berpotensi digunakan "
                            "untuk analisis berdasarkan waktu."
                        ),
                    ],
                )
            )

        # ---------------------------------------------------------
        # 4. SORTING
        # ---------------------------------------------------------

        tasks.sort(
            key=lambda item: item.get("score", 0),
            reverse=True,
        )

        # ---------------------------------------------------------
        # 5. PRIMARY TASK
        # ---------------------------------------------------------

        primary_task = self._select_primary_task(tasks)

        return {
            "status": self.STATUS,
            "primary_task": primary_task,
            "tasks": tasks,
            "task_count": len(tasks),
            "message": (
                "Jenis machine learning merupakan estimasi "
                "berdasarkan struktur dataset dan kandidat target."
            ),
        }

    # =============================================================
    # SUPERVISED TASK DETECTION
    # =============================================================

    def _detect_supervised_task(
        self,
        columns: list[dict[str, Any]],
        target_candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """
        Mendeteksi classification/regression berdasarkan kandidat target.
        """

        if not target_candidates:
            return None

        # Kandidat dengan score tertinggi diprioritaskan.
        target_candidates = sorted(
            target_candidates,
            key=lambda candidate: self._candidate_score(candidate),
            reverse=True,
        )

        for candidate in target_candidates:
            target_name = self._candidate_name(candidate)

            if not target_name:
                continue

            column = self._find_column(
                columns,
                target_name,
            )

            if not column:
                continue

            task = self._detect_task_from_column(column)

            if task is not None:
                return task

        return None

    def _detect_task_from_column(
        self,
        column: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Menentukan classification/regression dari sebuah target.
        """

        name = (
            column.get("name")
            or column.get("normalized_name")
            or ""
        )

        semantic_type = str(
            column.get("semantic_type", "")
        ).lower()

        dtype = str(
            column.get("dtype", "")
        ).lower()

        unique_count = self._get_unique_count(column)

        # ---------------------------------------------------------
        # BOOLEAN
        # ---------------------------------------------------------

        if semantic_type == "boolean" or dtype == "bool":
            return self._make_task(
                task="binary_classification",
                score=100,
                target=name,
                reasons=[
                    "Target bertipe boolean.",
                    "Target hanya memiliki dua kemungkinan nilai.",
                    "Struktur target sesuai dengan binary classification.",
                ],
            )

        # ---------------------------------------------------------
        # CATEGORICAL
        # ---------------------------------------------------------

        if semantic_type == "categorical":
            if unique_count == 2:
                return self._make_task(
                    task="binary_classification",
                    score=100,
                    target=name,
                    reasons=[
                        "Target bersifat kategorikal.",
                        "Target memiliki dua kelas.",
                        (
                            "Struktur target sesuai dengan "
                            "binary classification."
                        ),
                    ],
                )

            if 3 <= unique_count <= 20:
                return self._make_task(
                    task="multiclass_classification",
                    score=100,
                    target=name,
                    reasons=[
                        "Target bersifat kategorikal.",
                        (
                            f"Target memiliki {unique_count} "
                            "kelas."
                        ),
                        (
                            "Struktur target sesuai dengan "
                            "multiclass classification."
                        ),
                    ],
                )

        # ---------------------------------------------------------
        # NUMERIC
        # ---------------------------------------------------------

        if self._is_numeric_column(column):
            # Numeric dengan hanya dua nilai:
            # contoh 0/1 untuk classification.
            if unique_count == 2:
                return self._make_task(
                    task="binary_classification",
                    score=100,
                    target=name,
                    reasons=[
                        "Target bersifat numerik.",
                        "Target hanya memiliki dua nilai unik.",
                        (
                            "Struktur target sesuai dengan "
                            "binary classification."
                        ),
                    ],
                )

            # Numeric discrete dengan sedikit kelas.
            if 3 <= unique_count <= 20:
                if self._numeric_target_looks_discrete(column):
                    return self._make_task(
                        task="multiclass_classification",
                        score=95,
                        target=name,
                        reasons=[
                            "Target bersifat numerik.",
                            (
                                f"Target memiliki {unique_count} "
                                "nilai unik."
                            ),
                            (
                                "Nilai target terlihat diskrit "
                                "dan berpotensi merepresentasikan kelas."
                            ),
                        ],
                    )

            # Numeric kontinu → regression.
            return self._make_task(
                task="regression",
                score=100,
                target=name,
                reasons=[
                    "Target bersifat numerik.",
                    "Target memiliki karakteristik kontinu.",
                ],
            )

        # ---------------------------------------------------------
        # TEXT
        # ---------------------------------------------------------

        if semantic_type == "text":
            if unique_count == 2:
                return self._make_task(
                    task="binary_classification",
                    score=80,
                    target=name,
                    reasons=[
                        "Target berupa teks/kategori.",
                        "Target memiliki dua nilai unik.",
                    ],
                )

            if 3 <= unique_count <= 20:
                return self._make_task(
                    task="multiclass_classification",
                    score=75,
                    target=name,
                    reasons=[
                        "Target berupa teks/kategori.",
                        (
                            f"Target memiliki {unique_count} "
                            "nilai unik."
                        ),
                    ],
                )

        return None

    # =============================================================
    # PRIMARY TASK SELECTION
    # =============================================================

    def _select_primary_task(
        self,
        tasks: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """
        Memilih task utama.

        Prioritas:
        1. supervised task dengan target
        2. time-series
        3. clustering
        4. anomaly detection
        """

        if not tasks:
            return None

        supervised_tasks = [
            task
            for task in tasks
            if task.get("target") is not None
            and task.get("task")
            in {
                "binary_classification",
                "multiclass_classification",
                "regression",
            }
        ]

        if supervised_tasks:
            return max(
                supervised_tasks,
                key=lambda task: task.get("score", 0),
            )

        time_series_tasks = [
            task
            for task in tasks
            if task.get("task") == "time_series_forecasting"
        ]

        if time_series_tasks:
            return max(
                time_series_tasks,
                key=lambda task: task.get("score", 0),
            )

        return max(
            tasks,
            key=lambda task: task.get("score", 0),
        )

    # =============================================================
    # NORMALIZATION
    # =============================================================

    def _normalize_columns(
        self,
        columns: Any,
    ) -> list[dict[str, Any]]:
        """
        Menormalisasi column_signature.
        """

        if isinstance(columns, list):
            return [
                column
                for column in columns
                if isinstance(column, dict)
            ]

        if isinstance(columns, dict):
            normalized = []

            for name, info in columns.items():
                if isinstance(info, dict):
                    item = dict(info)
                    item.setdefault("name", name)
                    normalized.append(item)

            return normalized

        return []

    def _normalize_target_candidates(
        self,
        candidates: Any,
    ) -> list[dict[str, Any]]:
        """
        Menormalisasi target_candidates.

        Mendukung format:
        - [{"name": "Survived", "score": 100}]
        - ["Survived"]
        - {"Survived": 100}
        """

        if not candidates:
            return []

        if isinstance(candidates, dict):
            result = []

            for name, score in candidates.items():
                result.append(
                    {
                        "name": name,
                        "score": score,
                    }
                )

            return result

        if isinstance(candidates, list):
            result = []

            for candidate in candidates:
                if isinstance(candidate, dict):
                    result.append(candidate)

                elif isinstance(candidate, str):
                    result.append(
                        {
                            "name": candidate,
                            "score": 0,
                        }
                    )

            return result

        return []

    # =============================================================
    # COLUMN HELPERS
    # =============================================================

    def _find_column(
        self,
        columns: list[dict[str, Any]],
        target_name: str,
    ) -> dict[str, Any] | None:
        """
        Mencari kolom berdasarkan name atau normalized_name.
        """

        target = str(target_name).strip().lower()

        for column in columns:
            name = str(
                column.get("name", "")
            ).strip().lower()

            normalized_name = str(
                column.get("normalized_name", "")
            ).strip().lower()

            if target in {name, normalized_name}:
                return column

        return None

    def _candidate_name(
        self,
        candidate: dict[str, Any],
    ) -> str | None:
        return (
            candidate.get("name")
            or candidate.get("column")
            or candidate.get("column_name")
            or candidate.get("target")
        )

    def _candidate_score(
        self,
        candidate: dict[str, Any],
    ) -> float:
        for key in (
            "score",
            "target_score",
            "confidence",
        ):
            value = candidate.get(key)

            if isinstance(value, (int, float)):
                return float(value)

        return 0.0

    def _get_unique_count(
        self,
        column: dict[str, Any],
    ) -> int:
        value = column.get("unique_count", 0)

        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _is_numeric_column(
        self,
        column: dict[str, Any],
    ) -> bool:
        semantic_type = str(
            column.get("semantic_type", "")
        ).lower()

        dtype = str(
            column.get("dtype", "")
        ).lower()

        if semantic_type == "numeric":
            return True

        numeric_keywords = (
            "int",
            "float",
            "double",
            "decimal",
        )

        return any(
            keyword in dtype
            for keyword in numeric_keywords
        )

    def _numeric_target_looks_discrete(
        self,
        column: dict[str, Any],
    ) -> bool:
        """
        Menilai apakah numeric target kemungkinan merupakan kelas diskrit.
        """

        name = str(
            column.get("name")
            or column.get("normalized_name")
            or ""
        ).lower()

        discrete_keywords = {
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
            "label",
            "code",
        }

        name_tokens = set(
            name.replace("-", "_").split("_")
        )

        if name_tokens & discrete_keywords:
            return True

        minimum = column.get("min")
        maximum = column.get("max")

        try:
            minimum = float(minimum)
            maximum = float(maximum)

            if minimum >= 0 and maximum <= 20:
                return True

        except (TypeError, ValueError):
            pass

        return False

    # =============================================================
    # CHARACTERISTICS
    # =============================================================

    def _get_numeric_count(
        self,
        columns: list[dict[str, Any]],
        characteristics: dict[str, Any],
    ) -> int:
        value = characteristics.get("numeric_columns")

        if isinstance(value, int):
            return value

        if isinstance(value, list):
            return len(value)

        return sum(
            1
            for column in columns
            if self._is_numeric_column(column)
        )

    def _get_datetime_count(
        self,
        columns: list[dict[str, Any]],
        characteristics: dict[str, Any],
    ) -> int:
        value = characteristics.get("datetime_columns")

        if isinstance(value, int):
            return value

        if isinstance(value, list):
            return len(value)

        return sum(
            1
            for column in columns
            if str(
                column.get("semantic_type", "")
            ).lower()
            == "datetime"
        )

    # =============================================================
    # RESULT BUILDER
    # =============================================================

    def _make_task(
        self,
        task: str,
        score: float,
        target: str | None,
        reasons: list[str],
    ) -> dict[str, Any]:
        score = int(round(score))

        return {
            "task": task,
            "label": self.TASK_LABELS.get(
                task,
                task,
            ),
            "score": score,
            "confidence": score,
            "target": target,
            "status": self.STATUS,
            "reasons": reasons,
        }