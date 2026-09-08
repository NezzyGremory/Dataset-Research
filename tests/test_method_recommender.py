from app.ml.method_recommender import MethodRecommender


def build_fingerprint(
    rows=1000,
    columns=5,
    numeric_features=3,
    categorical_features=0,
    missing_columns=0,
    target="Survived",
    target_unique=2,
    target_type="numeric",
):

    signature = []

    for i in range(numeric_features):
        signature.append(
            {
                "name": f"feature_{i}",
                "dtype": "float64",
                "semantic_type": "numeric",
                "missing_count": 0,
            }
        )

    for i in range(categorical_features):
        signature.append(
            {
                "name": f"category_{i}",
                "dtype": "object",
                "semantic_type": "categorical",
                "missing_count": 0,
            }
        )

    while len(signature) < columns:
        signature.append(
            {
                "name": f"extra_{len(signature)}",
                "dtype": "float64",
                "semantic_type": "numeric",
                "missing_count": 0,
            }
        )

    return {
        "representation": {
            "rows": rows,
            "columns": columns,
            "column_signature": signature,
            "target_candidates": [
                {
                    "name": target,
                    "semantic_type": target_type,
                    "unique_count": target_unique,
                }
            ],
        }
    }


def test_small_binary_dataset():

    recommender = MethodRecommender()

    task_result = {
        "primary_task": {
            "task": "binary_classification",
            "target": "Survived",
        }
    }

    fingerprint = build_fingerprint(
        rows=300,
        columns=5,
        numeric_features=4,
        target_unique=2,
    )

    result = recommender.recommend(
        task_result,
        fingerprint,
    )

    assert result["methods"]

    top_method = result["methods"][0]

    print(
        "Small dataset top method:",
        top_method["method"],
        top_method["score"],
    )

    assert "score" in top_method
    assert "adjustments" in top_method
    assert "reasons" in top_method


def test_large_high_dimensional_dataset():

    recommender = MethodRecommender()

    task_result = {
        "primary_task": {
            "task": "binary_classification",
            "target": "Survived",
        }
    }

    fingerprint = build_fingerprint(
        rows=10000,
        columns=25,
        numeric_features=20,
        target_unique=2,
    )

    result = recommender.recommend(
        task_result,
        fingerprint,
    )

    assert result["methods"]

    random_forest = next(
        method
        for method in result["methods"]
        if method["method"] == "Random Forest Classifier"
    )

    print(
        "Random Forest score:",
        random_forest["score"],
    )

    assert random_forest["score"] > 92
    assert random_forest["adjustments"]


def test_regression_intelligent_scoring():

    recommender = MethodRecommender()

    task_result = {
        "primary_task": {
            "task": "regression",
            "target": "Price",
        }
    }

    fingerprint = build_fingerprint(
        rows=10000,
        columns=15,
        numeric_features=12,
        target_unique=5000,
        target_type="numeric",
    )

    result = recommender.recommend(
        task_result,
        fingerprint,
    )

    assert result["methods"]

    print(
        "Regression ranking:"
    )

    for method in result["methods"]:
        print(
            method["method"],
            method["score"],
            method["adjustments"],
        )

    assert result["methods"][0]["score"] >= result["methods"][-1]["score"]


def test_unknown_task():

    recommender = MethodRecommender()

    task_result = {
        "primary_task": {
            "task": "unknown_task",
        }
    }

    result = recommender.recommend(task_result)

    assert result["methods"] == []


if __name__ == "__main__":

    test_small_binary_dataset()
    test_large_high_dimensional_dataset()
    test_regression_intelligent_scoring()
    test_unknown_task()

    print()
    print("================================")
    print("Intelligent Method Scoring test berhasil.")
    print("================================")