from app.ml.method_info import (
    METHOD_INFO,
    get_all_methods,
    get_method_info,
    get_methods_for_task,
)


def test_method_info_exists():

    assert METHOD_INFO

    print(
        "Jumlah method dalam knowledge base:",
        len(METHOD_INFO),
    )


def test_random_forest_info():

    info = get_method_info(
        "random_forest_classifier"
    )

    assert info is not None

    assert info["name"] == "Random Forest Classifier"

    assert (
        "binary_classification"
        in info["suitable_tasks"]
    )

    assert info["nonlinear"] is True

    assert info["scaling_required"] is False

    assert info["strengths"]

    assert info["limitations"]

    assert info["preprocessing"]

    print(
        "Random Forest knowledge base: PASSED"
    )


def test_classification_methods():

    methods = get_methods_for_task(
        "binary_classification"
    )

    assert methods

    assert (
        "random_forest_classifier"
        in methods
    )

    assert (
        "logistic_regression"
        in methods
    )

    print(
        "Classification methods:",
        len(methods),
    )


def test_regression_methods():

    methods = get_methods_for_task(
        "regression"
    )

    assert methods

    assert (
        "linear_regression"
        in methods
    )

    assert (
        "random_forest_regressor"
        in methods
    )

    print(
        "Regression methods:",
        len(methods),
    )


def test_clustering_methods():

    methods = get_methods_for_task(
        "clustering"
    )

    assert methods

    assert "kmeans" in methods
    assert "dbscan" in methods

    print(
        "Clustering methods:",
        len(methods),
    )


def test_anomaly_methods():

    methods = get_methods_for_task(
        "anomaly_detection"
    )

    assert methods

    assert (
        "isolation_forest"
        in methods
    )

    assert (
        "local_outlier_factor"
        in methods
    )

    print(
        "Anomaly methods:",
        len(methods),
    )


def test_unknown_method():

    info = get_method_info(
        "method_yang_tidak_ada"
    )

    assert info is None

    print(
        "Unknown method handling: PASSED"
    )


def test_all_methods_have_required_fields():

    required_fields = [
        "name",
        "category",
        "suitable_tasks",
        "description",
        "strengths",
        "limitations",
        "preprocessing",
        "interpretability",
        "scaling_required",
        "nonlinear",
    ]

    for method_id, info in METHOD_INFO.items():

        for field in required_fields:

            assert field in info, (
                f"{method_id} missing field: {field}"
            )

    print(
        "Required fields validation: PASSED"
    )


if __name__ == "__main__":

    test_method_info_exists()
    test_random_forest_info()
    test_classification_methods()
    test_regression_methods()
    test_clustering_methods()
    test_anomaly_methods()
    test_unknown_method()
    test_all_methods_have_required_fields()

    print()
    print("================================")
    print("Semua Method Knowledge Base test berhasil.")
    print("================================")