import pandas as pd

from app.ml.evaluation import MLEvaluator


def test_binary_classification():
    data = pd.DataFrame(
        {
            "Age": [20, 30, 40, 25, 35, 50, 28, 45, 32, 22],
            "Fare": [10, 20, 30, 15, 25, 50, 12, 40, 22, 11],
            "Sex": [
                "male",
                "female",
                "male",
                "female",
                "male",
                "male",
                "female",
                "male",
                "female",
                "female",
            ],
            "Survived": [
                0,
                1,
                0,
                1,
                0,
                0,
                1,
                0,
                1,
                1,
            ],
        }
    )

    evaluator = MLEvaluator()

    result = evaluator.evaluate(
        dataframe=data,
        target="Survived",
        task="binary_classification",
    )

    assert result["status"] == "EVALUATION"
    assert result["result_count"] > 0

    for item in result["results"]:
        assert "method_id" in item
        assert "metrics" in item
        assert "accuracy" in item["metrics"]
        assert "precision" in item["metrics"]
        assert "recall" in item["metrics"]
        assert "f1_score" in item["metrics"]

    print("Binary classification evaluation: PASSED")

    print("\nRanking:")
    for item in result["results"]:
        print(
            item["method_id"],
            item["metrics"],
        )


def test_regression():
    data = pd.DataFrame(
        {
            "Area": [
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                110,
            ],
            "Rooms": [
                1,
                2,
                2,
                3,
                3,
                4,
                4,
                5,
                5,
                6,
            ],
            "Price": [
                100,
                150,
                200,
                260,
                300,
                360,
                410,
                480,
                530,
                600,
            ],
        }
    )

    evaluator = MLEvaluator()

    result = evaluator.evaluate(
        dataframe=data,
        target="Price",
        task="regression",
    )

    assert result["status"] == "EVALUATION"
    assert result["result_count"] > 0

    for item in result["results"]:
        assert "mae" in item["metrics"]
        assert "rmse" in item["metrics"]
        assert "r2_score" in item["metrics"]

    print("\nRegression evaluation: PASSED")

    print("\nRanking:")
    for item in result["results"]:
        print(
            item["method_id"],
            item["metrics"],
        )


def test_unknown_task():

    data = pd.DataFrame(
        {
            "A": [1, 2, 3, 4],
            "B": [5, 6, 7, 8],
        }
    )

    evaluator = MLEvaluator()

    result = evaluator.evaluate(
        dataframe=data,
        target="A",
        task="unknown_task",
    )

    assert result["status"] == "ERROR"

    print("Unknown task handling: PASSED")


if __name__ == "__main__":
    print("=== ML METHOD EVALUATION ===")

    test_binary_classification()
    test_regression()
    test_unknown_task()

    print()
    print("================================")
    print("Semua ML Evaluation test berhasil.")
    print("================================")