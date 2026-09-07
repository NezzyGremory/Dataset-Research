from app.analyzer.fingerprint import DatasetFingerprint
from app.nlp.similarity import FingerprintSimilarity

import pandas as pd


def create_titanic_like_dataset():
    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4, 5],
            "Survived": [0, 1, 1, 0, 1],
            "Pclass": [3, 1, 3, 1, 2],
            "Name": [
                "Person A",
                "Person B",
                "Person C",
                "Person D",
                "Person E",
            ],
            "Sex": [
                "male",
                "female",
                "female",
                "male",
                "female",
            ],
            "Age": [
                22,
                38,
                26,
                35,
                28,
            ],
            "Fare": [
                7.25,
                71.28,
                7.92,
                53.10,
                13.00,
            ],
        }
    )


def create_different_dataset():
    return pd.DataFrame(
        {
            "Product": [
                "Laptop",
                "Mouse",
                "Keyboard",
                "Monitor",
                "Printer",
            ],
            "Price": [
                500,
                20,
                30,
                200,
                100,
            ],
            "Stock": [
                10,
                50,
                30,
                15,
                20,
            ],
            "Category": [
                "Computer",
                "Accessory",
                "Accessory",
                "Computer",
                "Office",
            ],
        }
    )


def test_same_dataset_similarity():
    fingerprint_engine = DatasetFingerprint()
    similarity_engine = FingerprintSimilarity()

    dataframe = create_titanic_like_dataset()

    fingerprint = fingerprint_engine.generate(
        dataframe
    )

    result = similarity_engine.compare(
        fingerprint,
        fingerprint,
    )

    assert result["overall_score"] == 100.0
    assert result["schema_similarity"] == 100.0
    assert result["keyword_similarity"] == 100.0
    assert result["target_similarity"] == 100.0
    assert result["size_similarity"] == 100.0


def test_different_dataset_similarity():
    fingerprint_engine = DatasetFingerprint()
    similarity_engine = FingerprintSimilarity()

    dataframe_a = create_titanic_like_dataset()
    dataframe_b = create_different_dataset()

    fingerprint_a = fingerprint_engine.generate(
        dataframe_a
    )

    fingerprint_b = fingerprint_engine.generate(
        dataframe_b
    )

    result = similarity_engine.compare(
        fingerprint_a,
        fingerprint_b,
    )

    print("\nSimilarity Result:")
    print(result)

    assert 0 <= result["overall_score"] <= 100
    assert result["overall_score"] < 100

    if __name__ == "__main__":
    test_same_dataset_similarity()
    test_different_dataset_similarity()

    print("\nSemua test similarity berhasil.")