from app.analyzer.fingerprint import DatasetFingerprint
from app.nlp.domain_detector import DomainDetector
from app.nlp.keyword_extractor import KeywordExtractor

import pandas as pd


def create_titanic_dataset():
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


def test_keyword_extractor():

    dataframe = create_titanic_dataset()

    fingerprint_engine = DatasetFingerprint()

    fingerprint = fingerprint_engine.generate(
        dataframe
    )

    extractor = KeywordExtractor()

    result = extractor.extract(
        dataframe=dataframe,
        fingerprint=fingerprint,
        filename="titanic.csv",
    )

    print("\n=== KEYWORD TEST ===")
    print(result)

    assert "passengerid" in result["keywords"]
    assert "survived" in result["keywords"]
    assert len(result["keywords"]) > 0

    print("Keyword extraction: PASSED")


def test_domain_detector():

    detector = DomainDetector()

    keywords = [
        "passenger",
        "survived",
        "fare",
        "transport",
        "vehicle",
    ]

    result = detector.detect(
        keywords
    )

    print("\n=== DOMAIN TEST ===")
    print(result)

    assert result["primary_domain"] == "transportation"
    assert result["confidence"] > 0

    print("Domain detection: PASSED")


if __name__ == "__main__":
    test_keyword_extractor()
    test_domain_detector()

    print("\n================================")
    print("Semua test NLP berhasil.")
    print("================================")