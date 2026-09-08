from app.research.paper import Paper
from app.research.deduplication import PaperDeduplicator
from app.research.ranking import PaperRanker


def test_paper_model():
    paper = Paper(
        title="Machine Learning for Passenger Survival Prediction",
        authors=["John Doe"],
        abstract="A study using random forest classification.",
        year=2024,
        doi="https://doi.org/10.1234/ABC",
        source="OpenAlex",
    )

    assert paper.title
    assert paper.normalized_doi == "10.1234/abc"
    assert paper.normalized_title.startswith(
        "machine learning"
    )

    print("Paper model: PASSED")


def test_deduplication():
    paper1 = Paper(
        title="Machine Learning for Survival Prediction",
        doi="10.1234/test",
        abstract="Abstract A",
        source="OpenAlex",
    )

    paper2 = Paper(
        title="Machine Learning for Survival Prediction",
        doi="https://doi.org/10.1234/TEST",
        abstract="Abstract B",
        source="Crossref",
    )

    deduplicator = PaperDeduplicator()

    result = deduplicator.deduplicate(
        [paper1, paper2]
    )

    assert len(result) == 1
    assert result[0].abstract == "Abstract A"
    assert "Crossref" in result[0].source

    print("Deduplication: PASSED")


def test_ranking():
    papers = [
        {
            "title": (
                "Random Forest for Passenger "
                "Survival Prediction"
            ),
            "abstract": (
                "Machine learning classification "
                "for passenger survival."
            ),
            "year": 2024,
        },
        {
            "title": "Weather Forecasting Using Neural Networks",
            "abstract": (
                "A study about weather prediction."
            ),
            "year": 2024,
        },
    ]

    fingerprint = {
        "representation": {
            "keywords": [
                "passenger",
                "survival",
                "classification",
            ],
            "target_candidates": [
                {"name": "Survived"}
            ],
        }
    }

    domain_result = {
        "primary_domain": "transportation"
    }

    ml_result = {
        "primary_task": {
            "task": "binary_classification",
            "target": "Survived",
        },
        "recommendation": {
            "recommendations": [
                {
                    "method_id": "random_forest_classifier",
                    "method": "Random Forest Classifier",
                },
                {
                    "method_id": "logistic_regression",
                    "method": "Logistic Regression",
                },
            ]
        },
    }

    ranker = PaperRanker()

    result = ranker.rank(
        papers=papers,
        fingerprint=fingerprint,
        domain_result=domain_result,
        ml_result=ml_result,
    )

    assert len(result) == 2
    assert result[0]["relevance_score"] >= result[1]["relevance_score"]
    assert result[0]["rank"] == 1
    assert "score_breakdown" in result[0]
    assert "reasons" in result[0]

    print("Paper ranking: PASSED")


if __name__ == "__main__":
    print("=== ACADEMIC RESEARCH ENGINE ===")

    test_paper_model()
    test_deduplication()
    test_ranking()

    print()
    print("================================")
    print("Semua Academic Research Engine test berhasil.")
    print("================================")