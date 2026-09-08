from app.research.gap_analyzer import (
    ResearchGapAnalyzer,
)
from app.research.landscape import (
    ResearchLandscapeAnalyzer,
)


def sample_papers():

    return [
        {
            "title": (
                "Passenger Survival Using Random Forest"
            ),
            "abstract": (
                "Random forest classification "
                "for passenger survival."
            ),
            "year": 2024,
            "venue": "Journal A",
            "source": "OpenAlex",
        },
        {
            "title": (
                "Passenger Survival Prediction"
            ),
            "abstract": (
                "Logistic regression for "
                "survival prediction."
            ),
            "year": 2023,
            "venue": "Journal B",
            "source": "OpenAlex",
        },
        {
            "title": (
                "Passenger Classification "
                "with Random Forest"
            ),
            "abstract": (
                "Random forest classification "
                "using Kaggle Titanic dataset."
            ),
            "year": 2024,
            "venue": "Journal A",
            "source": "Crossref",
        },
        {
            "title": (
                "Survival Prediction Using SVM"
            ),
            "abstract": (
                "Support vector machine for "
                "passenger prediction."
            ),
            "year": 2022,
            "venue": "Journal C",
            "source": "OpenAlex",
        },
        {
            "title": (
                "Passenger Prediction "
                "with Random Forest"
            ),
            "abstract": (
                "Random forest model "
                "for passenger survival."
            ),
            "year": 2024,
            "venue": "Journal A",
            "source": "OpenAlex",
        },
    ]


def test_landscape():

    papers = sample_papers()

    analyzer = (
        ResearchLandscapeAnalyzer()
    )

    result = analyzer.analyze(
        papers
    )

    assert result["status"] == "SUCCESS"

    assert result[
        "paper_count"
    ] == 5

    assert len(
        result[
            "publication_years"
        ]
    ) > 0

    assert len(
        result["methods"]
    ) > 0

    assert (
        result[
            "summary"
        ][
            "dominant_method"
        ]
        == "Random Forest"
    )

    print(
        "Research Landscape: PASSED"
    )

    print(
        "Methods:"
    )

    for method in result[
        "methods"
    ]:

        print(
            f"- {method['name']}: "
            f"{method['percentage']}%"
        )


def test_gap():

    papers = sample_papers()

    landscape = (
        ResearchLandscapeAnalyzer()
        .analyze(papers)
    )

    analyzer = (
        ResearchGapAnalyzer()
    )

    result = analyzer.analyze(
        papers=papers,
        landscape=landscape,
    )

    assert (
        result["status"]
        == "SUCCESS"
    )

    assert "gaps" in result

    assert (
        result[
            "summary"
        ][
            "gap_count"
        ]
        >= 1
    )

    assert (
        result["status_type"]
        == "HEURISTIC"
    )

    print(
        "Research Gap Analyzer: PASSED"
    )

    for gap in result[
        "gaps"
    ]:

        print(
            f"- {gap['title']}: "
            f"{gap['description']}"
        )


if __name__ == "__main__":

    print(
        "=== RESEARCH LANDSCAPE & GAP ==="
    )

    test_landscape()

    print()

    test_gap()

    print()

    print(
        "================================"
    )

    print(
        "Semua Landscape & Gap "
        "test berhasil."
    )

    print(
        "================================"
    )