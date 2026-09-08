from __future__ import annotations

from collections import defaultdict


class DomainDetector:
    """
    Heuristic domain detector.

    Hasil domain bersifat ESTIMATION,
    bukan klasifikasi domain yang pasti.
    """

    DOMAIN_RULES = {
        "healthcare": {
            "keywords": {
                "patient",
                "diagnosis",
                "disease",
                "medical",
                "health",
                "hospital",
                "symptom",
                "treatment",
                "doctor",
                "clinic",
                "blood",
                "heart",
                "cancer",
                "diabetes",
            }
        },
        "finance": {
            "keywords": {
                "price",
                "stock",
                "finance",
                "financial",
                "revenue",
                "profit",
                "sales",
                "market",
                "investment",
                "loan",
                "credit",
                "transaction",
                "bank",
                "income",
            }
        },
        "business": {
            "keywords": {
                "customer",
                "churn",
                "sales",
                "product",
                "company",
                "employee",
                "business",
                "marketing",
                "purchase",
                "order",
                "retention",
                "revenue",
            }
        },
        "education": {
            "keywords": {
                "student",
                "school",
                "university",
                "education",
                "grade",
                "score",
                "exam",
                "course",
                "teacher",
                "academic",
                "learning",
            }
        },
        "transportation": {
            "keywords": {
                "passenger",
                "vehicle",
                "car",
                "bus",
                "train",
                "flight",
                "airport",
                "transport",
                "trip",
                "fare",
                "route",
                "distance",
            }
        },
        "environment": {
            "keywords": {
                "temperature",
                "rainfall",
                "weather",
                "climate",
                "humidity",
                "air",
                "pollution",
                "environment",
                "wind",
                "water",
                "soil",
            }
        },
        "agriculture": {
            "keywords": {
                "crop",
                "plant",
                "soil",
                "farm",
                "agriculture",
                "yield",
                "harvest",
                "seed",
                "fertilizer",
                "irrigation",
            }
        },
        "technology": {
            "keywords": {
                "computer",
                "software",
                "hardware",
                "network",
                "internet",
                "server",
                "system",
                "device",
                "sensor",
                "machine",
                "algorithm",
                "model",
            }
        },
        "social_science": {
            "keywords": {
                "gender",
                "age",
                "income",
                "population",
                "survey",
                "social",
                "behavior",
                "community",
                "marriage",
                "employment",
                "occupation",
            }
        },
    }

    def detect(
        self,
        keywords: list[str],
    ) -> dict:

        normalized_keywords = {
            keyword.strip().lower()
            for keyword in keywords
            if keyword
        }

        scores = defaultdict(float)
        matched_keywords = defaultdict(list)

        for domain, rule in self.DOMAIN_RULES.items():

            domain_keywords = rule["keywords"]

            for keyword in normalized_keywords:

                if keyword in domain_keywords:

                    scores[domain] += 1

                    matched_keywords[domain].append(
                        keyword
                    )

        if not scores:

            return {
                "status": "ESTIMATION",
                "domains": [],
                "primary_domain": None,
                "confidence": 0.0,
                "message": (
                    "Domain belum dapat diperkirakan "
                    "berdasarkan keyword yang tersedia."
                ),
            }

        total_score = sum(
            scores.values()
        )

        domain_results = []

        for domain, score in scores.items():

            confidence = (
                score / total_score
            ) * 100

            domain_results.append(
                {
                    "domain": domain,
                    "score": round(score, 2),
                    "confidence": round(
                        confidence,
                        2,
                    ),
                    "matched_keywords": sorted(
                        matched_keywords[domain]
                    ),
                    "status": "ESTIMATION",
                }
            )

        domain_results.sort(
            key=lambda item: item["confidence"],
            reverse=True,
        )

        primary = domain_results[0]

        return {
            "status": "ESTIMATION",
            "domains": domain_results,
            "primary_domain": primary["domain"],
            "confidence": primary["confidence"],
            "message": (
                "Domain merupakan estimasi heuristik "
                "berdasarkan keyword dataset."
            ),
        }