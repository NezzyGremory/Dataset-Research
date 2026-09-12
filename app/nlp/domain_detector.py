from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


class DomainDetector:
    """
    Heuristic domain detector yang bersifat dataset-agnostic.

    Detector menggabungkan keyword yang diberikan engine dengan metadata
    kolom/dataframe bila tersedia. Hasilnya adalah ESTIMASI domain, bukan
    klasifikasi domain yang pasti.

    Kompatibel dengan beberapa cara pemanggilan:
        detector.detect(keywords)
        detector.detect(dataframe=dataframe, fingerprint=fingerprint)
        detector.detect(dataframe, fingerprint)
        detector.detect(fingerprint=fingerprint)
    """

    DOMAIN_RULES = {
        "healthcare": {
            "keywords": {
                "patient", "diagnosis", "disease", "medical", "health",
                "hospital", "symptom", "treatment", "doctor", "clinic",
                "blood", "heart", "cancer", "diabetes", "medicine",
                "drug", "therapy", "surgery", "mental", "covid",
            }
        },
        "finance": {
            "keywords": {
                "price", "stock", "finance", "financial", "revenue",
                "profit", "sales", "market", "investment", "loan",
                "credit", "transaction", "bank", "income", "expense",
                "payment", "fraud", "debt", "interest", "portfolio",
            }
        },
        "business": {
            "keywords": {
                "customer", "churn", "sales", "product", "company",
                "employee", "business", "marketing", "purchase", "order",
                "retention", "revenue", "client", "service", "complaint",
                "subscription", "segment", "campaign",
            }
        },
        "education": {
            "keywords": {
                "student", "school", "university", "education", "grade",
                "score", "exam", "course", "teacher", "academic",
                "learning", "class", "semester", "attendance", "major",
                "study", "dropout", "graduate",
            }
        },
        "transportation": {
            "keywords": {
                "passenger", "vehicle", "car", "bus", "train", "flight",
                "airport", "transport", "trip", "fare", "route", "distance",
                "traffic", "driver", "travel", "station", "departure",
                "arrival", "fuel",
            }
        },
        "environment": {
            "keywords": {
                "temperature", "rainfall", "weather", "climate", "humidity",
                "air", "pollution", "environment", "wind", "water", "soil",
                "forecast", "drought", "flood", "emission", "carbon",
                "precipitation", "ecosystem",
            }
        },
        "agriculture": {
            "keywords": {
                "crop", "plant", "soil", "farm", "agriculture", "yield",
                "harvest", "seed", "fertilizer", "irrigation", "farmer",
                "rice", "wheat", "corn", "pesticide", "livestock", "field",
                "cultivation", "production",
            }
        },
        "technology": {
            "keywords": {
                "computer", "software", "hardware", "network", "internet",
                "server", "system", "device", "sensor", "machine", "algorithm",
                "model", "database", "application", "app", "programming",
                "cyber", "security", "cloud", "api", "technology",
            }
        },
        "social_science": {
            "keywords": {
                "gender", "age", "income", "population", "survey", "social",
                "behavior", "community", "marriage", "employment", "occupation",
                "demographic", "family", "household", "society", "political",
                "attitude", "perception",
            }
        },
    }

    # Beberapa alias umum agar nama kolom natural tetap terbaca sebagai domain.
    KEYWORD_ALIASES = {
        "medical": "healthcare",
        "doctor": "healthcare",
        "hospital": "healthcare",
        "patient": "healthcare",
        "student": "education",
        "school": "education",
        "university": "education",
        "customer": "business",
        "churn": "business",
        "transaction": "finance",
        "bank": "finance",
        "passenger": "transportation",
        "vehicle": "transportation",
        "temperature": "environment",
        "rainfall": "environment",
        "crop": "agriculture",
        "farm": "agriculture",
        "software": "technology",
        "computer": "technology",
        "gender": "social_science",
        "population": "social_science",
    }

    STOPWORDS = {
        "id", "idx", "index", "row", "column", "col", "number", "num",
        "value", "values", "data", "record", "records", "name", "date",
        "time", "timestamp", "label", "target", "class", "category",
    }

    def detect(
        self,
        keywords: Iterable[str] | None = None,
        dataframe: Any | None = None,
        fingerprint: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Detect domain from keywords and optional dataset metadata."""
        extracted_keywords = self._collect_keywords(
            keywords=keywords,
            dataframe=dataframe,
            fingerprint=fingerprint,
        )

        scores: dict[str, float] = defaultdict(float)
        matched_keywords: dict[str, set[str]] = defaultdict(set)

        # Primary evidence: explicit keywords.
        for keyword in extracted_keywords:
            normalized = self._normalize(keyword)
            if not normalized or normalized in self.STOPWORDS:
                continue

            for domain, rule in self.DOMAIN_RULES.items():
                if normalized in rule["keywords"]:
                    scores[domain] += 1.0
                    matched_keywords[domain].add(normalized)

        # Secondary evidence: compound/substring matches from column names.
        column_names = self._extract_column_names(dataframe, fingerprint)
        for column in column_names:
            normalized_column = self._normalize(column)
            tokens = self._tokenize(normalized_column)

            for token in tokens:
                if token in self.STOPWORDS:
                    continue
                for domain, rule in self.DOMAIN_RULES.items():
                    if token in rule["keywords"]:
                        # Column-name evidence is intentionally weaker than
                        # explicit keyword evidence.
                        scores[domain] += 0.75
                        matched_keywords[domain].add(token)

        if not scores:
            return {
                "status": "ESTIMATION",
                "domains": [],
                "primary_domain": None,
                "confidence": 0.0,
                "matched_keywords": [],
                "keywords_used": sorted(extracted_keywords),
                "message": (
                    "Domain belum dapat diperkirakan berdasarkan keyword "
                    "dan metadata dataset yang tersedia."
                ),
            }

        # Normalisasi skor ke confidence total 100%.
        total_score = sum(scores.values())
        domain_results = []

        for domain, score in scores.items():
            confidence = (score / total_score) * 100 if total_score else 0.0
            domain_results.append(
                {
                    "domain": domain,
                    "score": round(score, 2),
                    "confidence": round(confidence, 2),
                    "matched_keywords": sorted(matched_keywords[domain]),
                    "status": "ESTIMATION",
                }
            )

        domain_results.sort(
            key=lambda item: (item["confidence"], item["score"]),
            reverse=True,
        )

        primary = domain_results[0]

        return {
            "status": "ESTIMATION",
            "domains": domain_results,
            "primary_domain": primary["domain"],
            "confidence": primary["confidence"],
            "matched_keywords": primary["matched_keywords"],
            "keywords_used": sorted(extracted_keywords),
            "message": (
                "Domain merupakan estimasi heuristik berdasarkan keyword "
                "dan metadata dataset."
            ),
        }

    def _collect_keywords(
        self,
        keywords: Iterable[str] | None,
        dataframe: Any | None,
        fingerprint: dict[str, Any] | None,
    ) -> set[str]:
        result: set[str] = set()

        if keywords:
            result.update(self._normalize(value) for value in keywords if value)

        # Ambil keyword dari fingerprint jika engine tidak mengirimkan
        # keyword secara eksplisit.
        if isinstance(fingerprint, dict):
            for key in (
                "keywords",
                "top_keywords",
                "semantic_keywords",
                "domain_keywords",
            ):
                value = fingerprint.get(key)
                if isinstance(value, str):
                    result.add(self._normalize(value))
                elif isinstance(value, (list, tuple, set)):
                    for item in value:
                        if isinstance(item, str):
                            result.add(self._normalize(item))
                        elif isinstance(item, dict):
                            for field in ("keyword", "term", "word"):
                                candidate = item.get(field)
                                if isinstance(candidate, str):
                                    result.add(self._normalize(candidate))

        # Tambahkan token dari nama kolom sebagai fallback.
        for column in self._extract_column_names(dataframe, fingerprint):
            result.update(self._tokenize(self._normalize(column)))

        return {item for item in result if item}

    def _extract_column_names(
        self,
        dataframe: Any | None,
        fingerprint: dict[str, Any] | None,
    ) -> list[str]:
        columns: list[str] = []

        if dataframe is not None:
            try:
                columns.extend(str(column) for column in dataframe.columns)
            except Exception:
                pass

        if isinstance(fingerprint, dict):
            possible_columns = fingerprint.get("columns")
            if isinstance(possible_columns, dict):
                columns.extend(str(key) for key in possible_columns.keys())
            elif isinstance(possible_columns, (list, tuple, set)):
                columns.extend(str(item) for item in possible_columns)

            # Beberapa profiler menyimpan metadata kolom dengan key lain.
            for key in ("column_names", "features", "feature_names"):
                value = fingerprint.get(key)
                if isinstance(value, (list, tuple, set)):
                    columns.extend(str(item) for item in value)

        # Deduplicate sambil menjaga urutan.
        seen = set()
        result = []
        for column in columns:
            if column not in seen:
                seen.add(column)
                result.append(column)
        return result

    @staticmethod
    def _normalize(value: Any) -> str:
        text = str(value).strip().lower()
        for char in "-./\\":
            text = text.replace(char, " ")
        text = " ".join(text.split())
        return text

    @staticmethod
    def _tokenize(value: str) -> list[str]:
        if not value:
            return []
        separators = "_ -./\\()[]{}:"
        normalized = value
        for char in separators:
            normalized = normalized.replace(char, " ")
        return [token for token in normalized.split() if token]
