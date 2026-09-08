from __future__ import annotations

from collections import Counter
from typing import Dict, List


class ResearchGapAnalyzer:
    """
    Mendeteksi potential research gaps berdasarkan
    pola dari paper yang berhasil dianalisis.

    PENTING:
    Sistem tidak menyatakan gap sebagai fakta.
    Semua hasil diberi status POTENTIAL / HEURISTIC.
    """

    def analyze(
        self,
        papers: List[Dict],
        ml_result: Dict | None = None,
        landscape: Dict | None = None,
    ) -> Dict:

        if not papers:
            return {
                "status": "NO_DATA",
                "gaps": [],
                "summary": {
                    "gap_count": 0,
                    "papers_analyzed": 0,
                },
                "status_type": "HEURISTIC",
                "warning": (
                    "No papers available for gap analysis."
                ),
            }

        gaps = []

        gap_generators = [
            self._method_gap,
            self._underused_recommended_method,
            self._year_gap,
            self._dataset_gap,
            self._topic_gap,
        ]

        for generator in gap_generators:

            try:

                if generator == self._method_gap:
                    gap = generator(
                        papers,
                        ml_result,
                    )

                elif generator == self._underused_recommended_method:
                    gap = generator(
                        papers,
                        ml_result,
                    )

                else:
                    gap = generator(
                        papers
                    )

                if gap:
                    gaps.append(gap)

            except Exception:
                continue

        return {
            "status": "SUCCESS",
            "gaps": gaps,
            "summary": {
                "gap_count": len(gaps),
                "papers_analyzed": len(
                    papers
                ),
            },
            "status_type": "HEURISTIC",
            "warning": (
                "Potential gaps are heuristic signals "
                "based only on the analyzed literature. "
                "They are not confirmed research gaps."
            ),
        }

    def _method_gap(
        self,
        papers: List[Dict],
        ml_result: Dict | None,
    ) -> Dict | None:

        method_counts = Counter()

        aliases = {
            "Random Forest": [
                "random forest",
            ],
            "Logistic Regression": [
                "logistic regression",
            ],
            "Decision Tree": [
                "decision tree",
            ],
            "SVM": [
                "support vector machine",
                "svm",
            ],
            "KNN": [
                "knn",
                "k-nearest",
                "k nearest",
            ],
            "Gradient Boosting": [
                "gradient boosting",
            ],
            "Linear Regression": [
                "linear regression",
            ],
            "K-Means": [
                "k-means",
                "kmeans",
            ],
            "DBSCAN": [
                "dbscan",
            ],
        }

        for paper in papers:

            text = (
                f"{paper.get('title', '')} "
                f"{paper.get('abstract', '')}"
            ).lower()

            for method, method_aliases in aliases.items():

                if any(
                    alias in text
                    for alias in method_aliases
                ):
                    method_counts[method] += 1

        if len(method_counts) < 2:
            return None

        dominant = (
            method_counts.most_common()
        )

        most_used = dominant[0]
        least_used = dominant[-1]

        if (
            most_used[1]
            == least_used[1]
        ):
            return None

        return {
            "type": "METHOD_DISTRIBUTION",
            "title": "Potential Method Gap",
            "description": (
                f"{most_used[0]} appears more "
                f"frequently than {least_used[0]} "
                "in the analyzed papers."
            ),
            "evidence": {
                "dominant_method": (
                    most_used[0]
                ),
                "dominant_count": (
                    most_used[1]
                ),
                "less_common_method": (
                    least_used[0]
                ),
                "less_common_count": (
                    least_used[1]
                ),
            },
        }

    def _underused_recommended_method(
        self,
        papers: List[Dict],
        ml_result: Dict | None,
    ) -> Dict | None:

        if not ml_result:
            return None

        recommendation = ml_result.get(
            "recommendation",
            {},
        )

        recommendations = (
            recommendation.get(
                "recommendations",
                [],
            )
        )

        if not recommendations:
            return None

        text = " ".join(
            (
                f"{paper.get('title', '')} "
                f"{paper.get('abstract', '')}"
            )
            for paper in papers
        ).lower()

        method_aliases = {
            "random_forest_classifier": [
                "random forest",
            ],
            "random_forest_regressor": [
                "random forest",
            ],
            "logistic_regression": [
                "logistic regression",
            ],
            "decision_tree_classifier": [
                "decision tree",
            ],
            "decision_tree_regressor": [
                "decision tree",
            ],
            "svm_classifier": [
                "support vector machine",
                "svm",
            ],
            "knn_classifier": [
                "knn",
                "k-nearest",
                "k nearest",
            ],
            "gradient_boosting_regressor": [
                "gradient boosting",
            ],
            "linear_regression": [
                "linear regression",
            ],
            "kmeans": [
                "k-means",
                "kmeans",
            ],
            "dbscan": [
                "dbscan",
            ],
            "agglomerative_clustering": [
                "agglomerative clustering",
                "hierarchical clustering",
            ],
            "isolation_forest": [
                "isolation forest",
            ],
            "local_outlier_factor": [
                "local outlier factor",
                "lof",
            ],
            "one_class_svm": [
                "one class svm",
                "one-class svm",
            ],
        }

        for item in recommendations[:5]:

            method_id = item.get(
                "method_id"
            )

            if not method_id:
                continue

            aliases = method_aliases.get(
                method_id,
                [
                    method_id.replace(
                        "_",
                        " ",
                    )
                ],
            )

            matched = any(
                alias in text
                for alias in aliases
            )

            if not matched:

                method_name = item.get(
                    "method",
                    method_id,
                )

                return {
                    "type": (
                        "UNDERUSED_RECOMMENDED_METHOD"
                    ),
                    "title": (
                        "Potential Method Opportunity"
                    ),
                    "description": (
                        f"{method_name} is recommended "
                        "for the dataset but was not "
                        "detected in the analyzed papers."
                    ),
                    "evidence": {
                        "method_id": method_id,
                        "method": method_name,
                        "recommended_score": item.get(
                            "score"
                        ),
                    },
                }

        return None

    @staticmethod
    def _year_gap(
        papers: List[Dict],
    ) -> Dict | None:

        years = [
            paper.get("year")
            for paper in papers
            if isinstance(
                paper.get("year"),
                int,
            )
        ]

        if len(years) < 3:
            return None

        latest = max(years)

        recent_count = sum(
            1
            for year in years
            if year >= latest - 1
        )

        recent_ratio = (
            recent_count
            / len(years)
        )

        if recent_ratio > 0.30:
            return None

        return {
            "type": "TEMPORAL",
            "title": "Potential Temporal Gap",
            "description": (
                "Relatively few analyzed papers "
                "belong to the most recent "
                "publication period."
            ),
            "evidence": {
                "latest_year": latest,
                "recent_papers": recent_count,
                "total_papers": len(years),
                "recent_ratio": round(
                    recent_ratio * 100,
                    2,
                ),
            },
        }

    @staticmethod
    def _dataset_gap(
        papers: List[Dict],
    ) -> Dict | None:

        dataset_terms = Counter()

        known_datasets = [
            "titanic",
            "mnist",
            "cifar",
            "imagenet",
            "iris",
            "kaggle",
            "uci",
            "census",
        ]

        for paper in papers:

            text = (
                f"{paper.get('title', '')} "
                f"{paper.get('abstract', '')}"
            ).lower()

            for dataset in known_datasets:

                if dataset in text:
                    dataset_terms[
                        dataset
                    ] += 1

        if not dataset_terms:
            return None

        dominant = (
            dataset_terms.most_common(1)[0]
        )

        if dominant[1] < 2:
            return None

        return {
            "type": "DATASET_CONCENTRATION",
            "title": "Potential Dataset Gap",
            "description": (
                "The analyzed literature appears "
                "concentrated around a limited "
                "number of known datasets."
            ),
            "evidence": {
                "dominant_dataset": (
                    dominant[0]
                ),
                "paper_count": dominant[1],
            },
        }

    @staticmethod
    def _topic_gap(
        papers: List[Dict],
    ) -> Dict | None:

        if len(papers) < 5:
            return None

        abstracts = [
            paper.get(
                "abstract",
                "",
            ).lower()
            for paper in papers
        ]

        words = Counter()

        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "using",
            "this",
            "that",
            "from",
            "are",
            "was",
            "were",
            "have",
            "has",
            "into",
            "based",
            "study",
            "model",
            "data",
            "machine",
            "learning",
            "using",
            "method",
            "results",
            "research",
        }

        for abstract in abstracts:

            for word in abstract.split():

                word = word.strip(
                    ".,;:!?()[]{}\"'"
                )

                if (
                    len(word) >= 5
                    and word not in stopwords
                ):
                    words[word] += 1

        if not words:
            return None

        rare_words = [
            word
            for word, count
            in words.items()
            if count == 1
        ]

        if not rare_words:
            return None

        return {
            "type": "TOPIC_DIVERSITY",
            "title": (
                "Potential Topic Opportunity"
            ),
            "description": (
                "Several concepts appear only "
                "once in the analyzed abstracts "
                "and may represent less-explored "
                "directions."
            ),
            "evidence": {
                "rare_concepts": rare_words[:10],
            },
        }