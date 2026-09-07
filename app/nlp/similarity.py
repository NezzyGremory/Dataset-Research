from __future__ import annotations

from difflib import SequenceMatcher


class FingerprintSimilarity:
    """
    Compare two dataset fingerprints.

    Similarity yang dihasilkan bersifat struktural/heuristik,
    bukan bukti bahwa dua dataset berasal dari sumber yang sama.
    """

    def __init__(
        self,
        schema_weight: float = 0.40,
        keyword_weight: float = 0.25,
        target_weight: float = 0.25,
        size_weight: float = 0.10,
    ):
        self.schema_weight = schema_weight
        self.keyword_weight = keyword_weight
        self.target_weight = target_weight
        self.size_weight = size_weight

        total_weight = (
            schema_weight
            + keyword_weight
            + target_weight
            + size_weight
        )

        if abs(total_weight - 1.0) > 0.0001:
            raise ValueError(
                "Total similarity weight harus sama dengan 1.0."
            )

    def compare(
        self,
        fingerprint_a: dict,
        fingerprint_b: dict,
    ) -> dict:
        """
        Membandingkan dua fingerprint dataset.
        """

        representation_a = fingerprint_a.get(
            "representation",
            fingerprint_a,
        )

        representation_b = fingerprint_b.get(
            "representation",
            fingerprint_b,
        )

        schema_score = self._schema_similarity(
            representation_a,
            representation_b,
        )

        keyword_score = self._keyword_similarity(
            representation_a,
            representation_b,
        )

        target_score = self._target_similarity(
            representation_a,
            representation_b,
        )

        size_score = self._size_similarity(
            representation_a,
            representation_b,
        )

        overall_score = (
            schema_score * self.schema_weight
            + keyword_score * self.keyword_weight
            + target_score * self.target_weight
            + size_score * self.size_weight
        )

        explanation = self._build_explanation(
            schema_score=schema_score,
            keyword_score=keyword_score,
            target_score=target_score,
            size_score=size_score,
            overall_score=overall_score,
        )

        return {
            "overall_score": round(overall_score, 2),
            "schema_similarity": round(schema_score, 2),
            "keyword_similarity": round(keyword_score, 2),
            "target_similarity": round(target_score, 2),
            "size_similarity": round(size_score, 2),
            "weights": {
                "schema": self.schema_weight,
                "keyword": self.keyword_weight,
                "target": self.target_weight,
                "size": self.size_weight,
            },
            "explanation": explanation,
        }

    # ==========================================================
    # SCHEMA SIMILARITY
    # ==========================================================

    def _schema_similarity(
        self,
        representation_a: dict,
        representation_b: dict,
    ) -> float:

        columns_a = representation_a.get(
            "column_signature",
            [],
        )

        columns_b = representation_b.get(
            "column_signature",
            [],
        )

        if not columns_a or not columns_b:
            return 0.0

        types_a = [
            column.get("semantic_type")
            for column in columns_a
        ]

        types_b = [
            column.get("semantic_type")
            for column in columns_b
        ]

        type_score = self._multiset_similarity(
            types_a,
            types_b,
        )

        names_a = {
            column.get("normalized_name", "")
            for column in columns_a
            if column.get("normalized_name")
        }

        names_b = {
            column.get("normalized_name", "")
            for column in columns_b
            if column.get("normalized_name")
        }

        name_score = self._jaccard_similarity(
            names_a,
            names_b,
        )

        characteristics_a = representation_a.get(
            "characteristics",
            {},
        )

        characteristics_b = representation_b.get(
            "characteristics",
            {},
        )

        characteristic_score = self._characteristic_similarity(
            characteristics_a,
            characteristics_b,
        )

        score = (
            type_score * 0.40
            + name_score * 0.40
            + characteristic_score * 0.20
        )

        return score

    # ==========================================================
    # KEYWORD SIMILARITY
    # ==========================================================

    def _keyword_similarity(
        self,
        representation_a: dict,
        representation_b: dict,
    ) -> float:

        keywords_a = set(
            representation_a.get(
                "keywords",
                [],
            )
        )

        keywords_b = set(
            representation_b.get(
                "keywords",
                [],
            )
        )

        if not keywords_a and not keywords_b:
            return 100.0

        if not keywords_a or not keywords_b:
            return 0.0

        return self._jaccard_similarity(
            keywords_a,
            keywords_b,
        )

    # ==========================================================
    # TARGET SIMILARITY
    # ==========================================================

    def _target_similarity(
        self,
        representation_a: dict,
        representation_b: dict,
    ) -> float:

        targets_a = representation_a.get(
            "target_candidates",
            [],
        )

        targets_b = representation_b.get(
            "target_candidates",
            [],
        )

        if not targets_a and not targets_b:
            return 100.0

        if not targets_a or not targets_b:
            return 0.0

        normalized_a = {
            self._normalize_name(name)
            for name in targets_a
        }

        normalized_b = {
            self._normalize_name(name)
            for name in targets_b
        }

        exact_score = self._jaccard_similarity(
            normalized_a,
            normalized_b,
        )

        # Cek semantic type target
        target_types_a = self._target_types(
            representation_a,
        )

        target_types_b = self._target_types(
            representation_b,
        )

        type_score = self._jaccard_similarity(
            target_types_a,
            target_types_b,
        )

        score = (
            exact_score * 0.70
            + type_score * 0.30
        )

        return score

    # ==========================================================
    # SIZE SIMILARITY
    # ==========================================================

    def _size_similarity(
        self,
        representation_a: dict,
        representation_b: dict,
    ) -> float:

        rows_a = representation_a.get(
            "rows",
            0,
        )

        rows_b = representation_b.get(
            "rows",
            0,
        )

        columns_a = representation_a.get(
            "columns",
            0,
        )

        columns_b = representation_b.get(
            "columns",
            0,
        )

        row_score = self._numeric_similarity(
            rows_a,
            rows_b,
        )

        column_score = self._numeric_similarity(
            columns_a,
            columns_b,
        )

        return (
            row_score * 0.50
            + column_score * 0.50
        )

    # ==========================================================
    # HELPERS
    # ==========================================================

    def _jaccard_similarity(
        self,
        set_a: set,
        set_b: set,
    ) -> float:

        if not set_a and not set_b:
            return 100.0

        union = set_a | set_b

        if not union:
            return 100.0

        intersection = set_a & set_b

        return (
            len(intersection)
            / len(union)
        ) * 100

    def _multiset_similarity(
        self,
        values_a: list,
        values_b: list,
    ) -> float:

        counts_a = {}

        for value in values_a:
            counts_a[value] = (
                counts_a.get(value, 0) + 1
            )

        counts_b = {}

        for value in values_b:
            counts_b[value] = (
                counts_b.get(value, 0) + 1
            )

        keys = set(counts_a) | set(counts_b)

        if not keys:
            return 100.0

        intersection = 0
        union = 0

        for key in keys:
            count_a = counts_a.get(key, 0)
            count_b = counts_b.get(key, 0)

            intersection += min(
                count_a,
                count_b,
            )

            union += max(
                count_a,
                count_b,
            )

        if union == 0:
            return 100.0

        return (
            intersection
            / union
        ) * 100

    def _characteristic_similarity(
        self,
        characteristics_a: dict,
        characteristics_b: dict,
    ) -> float:

        keys = {
            "numeric_columns",
            "categorical_columns",
            "datetime_columns",
            "text_columns",
            "boolean_columns",
        }

        scores = []

        for key in keys:

            value_a = characteristics_a.get(
                key,
                0,
            )

            value_b = characteristics_b.get(
                key,
                0,
            )

            scores.append(
                self._numeric_similarity(
                    value_a,
                    value_b,
                )
            )

        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    def _numeric_similarity(
        self,
        value_a: float,
        value_b: float,
    ) -> float:

        if value_a == value_b:
            return 100.0

        if value_a == 0 and value_b == 0:
            return 100.0

        maximum = max(
            abs(value_a),
            abs(value_b),
        )

        if maximum == 0:
            return 100.0

        difference = abs(
            value_a - value_b
        )

        similarity = (
            1 - difference / maximum
        ) * 100

        return max(
            0.0,
            similarity,
        )

    def _target_types(
        self,
        representation: dict,
    ) -> set:

        targets = {
            self._normalize_name(name)
            for name in representation.get(
                "target_candidates",
                [],
            )
        }

        if not targets:
            return set()

        result = set()

        for column in representation.get(
            "column_signature",
            [],
        ):

            normalized_name = (
                self._normalize_name(
                    column.get("name", "")
                )
            )

            if normalized_name in targets:
                result.add(
                    column.get(
                        "semantic_type",
                        "unknown",
                    )
                )

        return result

    def _normalize_name(
        self,
        value: str,
    ) -> str:

        return (
            str(value)
            .strip()
            .lower()
            .replace(" ", "_")
        )

    def _build_explanation(
        self,
        schema_score: float,
        keyword_score: float,
        target_score: float,
        size_score: float,
        overall_score: float,
    ) -> list:

        explanation = []

        if overall_score >= 80:
            explanation.append(
                "Dataset memiliki kemiripan struktural yang sangat tinggi."
            )
        elif overall_score >= 60:
            explanation.append(
                "Dataset memiliki kemiripan struktural yang cukup tinggi."
            )
        elif overall_score >= 40:
            explanation.append(
                "Dataset memiliki kemiripan struktural sedang."
            )
        else:
            explanation.append(
                "Dataset memiliki kemiripan struktural rendah."
            )

        if schema_score >= 80:
            explanation.append(
                "Struktur dan tipe kolom sangat mirip."
            )
        elif schema_score >= 50:
            explanation.append(
                "Sebagian struktur kolom memiliki kemiripan."
            )
        else:
            explanation.append(
                "Struktur kolom cukup berbeda."
            )

        if keyword_score >= 80:
            explanation.append(
                "Keyword/nama kolom sangat mirip."
            )
        elif keyword_score >= 50:
            explanation.append(
                "Sebagian keyword dataset memiliki kemiripan."
            )
        else:
            explanation.append(
                "Keyword dataset relatif berbeda."
            )

        if target_score >= 80:
            explanation.append(
                "Kandidat target memiliki kemiripan tinggi."
            )
        elif target_score >= 50:
            explanation.append(
                "Kandidat target memiliki sebagian kemiripan."
            )
        else:
            explanation.append(
                "Kandidat target berbeda atau tidak terdeteksi."
            )

        return explanation