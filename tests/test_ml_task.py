from app.ml.task_detector import MLTaskDetector


def make_fingerprint(
    columns,
    target_candidates,
    rows=100,
):
    """
    Membuat fingerprint sederhana untuk kebutuhan testing.
    """

    return {
        "status": "SUCCESS",
        "representation": {
            "rows": rows,
            "columns": len(columns),
            "column_signature": columns,
            "target_candidates": target_candidates,
            "characteristics": {
                "numeric_columns": sum(
                    1 for column in columns
                    if column.get("semantic_type") == "numeric"
                ),
                "categorical_columns": sum(
                    1 for column in columns
                    if column.get("semantic_type") == "categorical"
                ),
                "datetime_columns": sum(
                    1 for column in columns
                    if column.get("semantic_type") == "datetime"
                ),
            },
        },
    }


def find_task(result, task_name):
    """
    Mencari task tertentu dari hasil detector.
    """

    for task in result.get("tasks", []):
        if task.get("task") == task_name:
            return task

    return None


def test_binary_classification():
    print("\n=== BINARY CLASSIFICATION TEST ===")

    fingerprint = make_fingerprint(
        columns=[
            {
                "name": "Age",
                "normalized_name": "age",
                "dtype": "int64",
                "semantic_type": "numeric",
                "unique_count": 70,
                "potential_id": False,
            },
            {
                "name": "Fare",
                "normalized_name": "fare",
                "dtype": "float64",
                "semantic_type": "numeric",
                "unique_count": 90,
                "potential_id": False,
            },
            {
                "name": "Survived",
                "normalized_name": "survived",
                "dtype": "int64",
                "semantic_type": "numeric",
                "unique_count": 2,
                "potential_id": False,
            },
        ],
        target_candidates=[
            {
                "name": "Survived",
                "score": 100,
                "reasons": ["target-like name"],
            }
        ],
    )

    detector = MLTaskDetector()
    result = detector.detect(fingerprint)

    print(result)

    task = result["primary_task"]

    assert task["task"] == "binary_classification"
    assert task["target"] == "Survived"

    print("Binary classification detection: PASSED")


def test_multiclass_classification():
    print("\n=== MULTICLASS CLASSIFICATION TEST ===")

    fingerprint = make_fingerprint(
        columns=[
            {
                "name": "Age",
                "normalized_name": "age",
                "dtype": "int64",
                "semantic_type": "numeric",
                "unique_count": 60,
                "potential_id": False,
            },
            {
                "name": "Fare",
                "normalized_name": "fare",
                "dtype": "float64",
                "semantic_type": "numeric",
                "unique_count": 80,
                "potential_id": False,
            },
            {
                "name": "Class",
                "normalized_name": "class",
                "dtype": "int64",
                "semantic_type": "categorical",
                "unique_count": 3,
                "potential_id": False,
            },
        ],
        target_candidates=[
            {
                "name": "Class",
                "score": 100,
                "reasons": ["target-like name"],
            }
        ],
    )

    detector = MLTaskDetector()
    result = detector.detect(fingerprint)

    print(result)

    task = result["primary_task"]

    assert task["task"] == "multiclass_classification"
    assert task["target"] == "Class"

    print("Multiclass classification detection: PASSED")


def test_regression():
    print("\n=== REGRESSION TEST ===")

    fingerprint = make_fingerprint(
        columns=[
            {
                "name": "Area",
                "normalized_name": "area",
                "dtype": "float64",
                "semantic_type": "numeric",
                "unique_count": 95,
                "potential_id": False,
            },
            {
                "name": "Rooms",
                "normalized_name": "rooms",
                "dtype": "numeric",
                "semantic_type": "numeric",
                "unique_count": 8,
                "potential_id": False,
            },
            {
                "name": "Price",
                "normalized_name": "price",
                "dtype": "float64",
                "semantic_type": "numeric",
                "unique_count": 100,
                "potential_id": False,
            },
        ],
        target_candidates=[
            {
                "name": "Price",
                "score": 100,
                "reasons": ["target-like name"],
            }
        ],
    )

    detector = MLTaskDetector()
    result = detector.detect(fingerprint)

    print(result)

    task = result["primary_task"]

    assert task["task"] == "regression"
    assert task["target"] == "Price"

    print("Regression detection: PASSED")


def test_clustering():
    print("\n=== CLUSTERING TEST ===")

    fingerprint = make_fingerprint(
        columns=[
            {
                "name": "Age",
                "normalized_name": "age",
                "dtype": "int64",
                "semantic_type": "numeric",
                "unique_count": 50,
                "potential_id": False,
            },
            {
                "name": "Income",
                "normalized_name": "income",
                "dtype": "float64",
                "semantic_type": "numeric",
                "unique_count": 90,
                "potential_id": False,
            },
            {
                "name": "Spending",
                "normalized_name": "spending",
                "dtype": "float64",
                "semantic_type": "numeric",
                "unique_count": 80,
                "potential_id": False,
            },
        ],
        target_candidates=[],
    )

    detector = MLTaskDetector()
    result = detector.detect(fingerprint)

    print(result)

    task = find_task(result, "clustering")

    assert task is not None
    assert task["task"] == "clustering"

    print("Clustering detection: PASSED")


def test_anomaly_detection():
    print("\n=== ANOMALY DETECTION TEST ===")

    fingerprint = make_fingerprint(
        columns=[
            {
                "name": "Temperature",
                "normalized_name": "temperature",
                "dtype": "float64",
                "semantic_type": "numeric",
                "unique_count": 90,
                "potential_id": False,
            },
            {
                "name": "Pressure",
                "normalized_name": "pressure",
                "dtype": "float64",
                "semantic_type": "numeric",
                "unique_count": 95,
                "potential_id": False,
            },
            {
                "name": "Humidity",
                "normalized_name": "humidity",
                "dtype": "float64",
                "semantic_type": "numeric",
                "unique_count": 85,
                "potential_id": False,
            },
        ],
        target_candidates=[],
    )

    detector = MLTaskDetector()
    result = detector.detect(fingerprint)

    print(result)

    task = find_task(result, "anomaly_detection")

    assert task is not None
    assert task["task"] == "anomaly_detection"

    print("Anomaly detection: PASSED")


def test_time_series_forecasting():
    print("\n=== TIME-SERIES FORECASTING TEST ===")

    fingerprint = make_fingerprint(
        columns=[
            {
                "name": "Date",
                "normalized_name": "date",
                "dtype": "datetime64[ns]",
                "semantic_type": "datetime",
                "unique_count": 100,
                "potential_id": False,
            },
            {
                "name": "Sales",
                "normalized_name": "sales",
                "dtype": "float64",
                "semantic_type": "numeric",
                "unique_count": 90,
                "potential_id": False,
            },
        ],
        target_candidates=[],
    )

    detector = MLTaskDetector()
    result = detector.detect(fingerprint)

    print(result)

    task = find_task(result, "time_series_forecasting")

    assert task is not None
    assert task["task"] == "time_series_forecasting"

    print("Time-series forecasting detection: PASSED")


if __name__ == "__main__":
    test_binary_classification()
    test_multiclass_classification()
    test_regression()
    test_clustering()
    test_anomaly_detection()
    test_time_series_forecasting()

    print("\n================================")
    print("Semua ML task test berhasil.")
    print("================================")