from app.ml.method_recommender import MethodRecommender


def test_classification_integration():
    recommender = MethodRecommender()

    ml_result = {
        "primary_task": {
            "task": "binary_classification",
            "label": "Binary Classification",
            "target": "Survived",
            "score": 100,
        },
        "dataset": {
            "rows": 891,
            "columns": 5,
            "numeric_features": 3,
        },
    }

    result = recommender.recommend(ml_result)

    assert result["status"] == "RECOMMENDATION"
    assert result["recommendation_count"] > 0

    first = result["recommendations"][0]

    assert "method_id" in first
    assert "method" in first
    assert "score" in first

    # Knowledge Base harus terintegrasi
    assert "knowledge_base" in first
    assert "description" in first
    assert "strengths" in first
    assert "limitations" in first
    assert "preprocessing" in first

    print("Top method:", first["method"])
    print("Score:", first["score"])
    print("Knowledge Base:", first["knowledge_base"]["name"])


def test_regression_integration():
    recommender = MethodRecommender()

    ml_result = {
        "primary_task": {
            "task": "regression",
            "label": "Regression",
            "target": "Price",
            "score": 100,
        },
        "dataset": {
            "rows": 10000,
            "columns": 20,
            "numeric_features": 15,
        },
    }

    result = recommender.recommend(ml_result)

    assert result["recommendation_count"] > 0

    for recommendation in result["recommendations"]:
        assert recommendation["knowledge_base"] is not None
        assert recommendation["description"]
        assert recommendation["category"] == "Regression"

    print("Regression integration: PASSED")


def test_unknown_task():
    recommender = MethodRecommender()

    ml_result = {
        "primary_task": {
            "task": "unknown_task",
            "label": "Unknown Task",
            "target": None,
            "score": 0,
        },
        "dataset": {
            "rows": 100,
            "columns": 5,
            "numeric_features": 3,
        },
    }

    result = recommender.recommend(ml_result)

    assert result["recommendations"] == []
    assert result["recommendation_count"] == 0

    print("Unknown task integration: PASSED")


if __name__ == "__main__":
    print("=== METHOD KNOWLEDGE BASE INTEGRATION ===")

    test_classification_integration()
    test_regression_integration()
    test_unknown_task()

    print()
    print("================================")
    print("Method integration test berhasil.")
    print("================================")