from __future__ import annotations

import pandas as pd

from app.research.intelligence import ResearchIntelligenceEngine


class MockSearchEngine:

    def search(self, query, limit=20):
        return {
            "status": "SUCCESS",
            "query": query,
            "papers": [
                {
                    "title": (
                        "Passenger Survival Prediction "
                        "Using Random Forest"
                    ),
                    "authors": ["Researcher A"],
                    "abstract": (
                        "Machine learning classification "
                        "for passenger survival prediction."
                    ),
                    "year": 2024,
                    "doi": "10.1234/survival",
                    "venue": "Machine Learning Journal",
                    "url": "https://example.com/paper",
                    "citation_count": 25,
                    "source": "Mock",
                    "external_id": "mock-1",
                    "keywords": [
                        "machine learning",
                        "survival",
                    ],
                }
            ],
            "result_count": 1,
            "sources": ["Mock"],
            "message": "Mock search",
        }


class MockDomainDetector:

    def detect(self, fingerprint):
        return {
            "status": "ESTIMATION",
            "primary_domain": "transportation",
            "confidence": 85.0,
        }


class MockKeywordExtractor:

    def extract(self, dataframe, fingerprint):
        return {
            "status": "SUCCESS",
            "keywords": [
                "passenger",
                "survival",
                "classification",
            ],
            "frequencies": {},
        }


def test_research_intelligence():

    print()
    print("Menyiapkan dataset...")

    dataframe = pd.DataFrame({
        "PassengerId": [1, 2, 3, 4, 5, 6],
        "Age": [22, 30, 35, 40, 28, 50],
        "Fare": [10, 20, 30, 40, 25, 50],
        "Survived": [1, 0, 1, 0, 1, 0],
    })

    print("Dataset shape:", dataframe.shape)

    fingerprint = {
        "representation": {
            "keywords": [
                "passenger",
                "survival",
                "classification",
            ],
            "target_candidates": [
                {
                    "name": "Survived",
                    "score": 90,
                }
            ],
        }
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
                    "score": 95,
                },
                {
                    "method_id": "logistic_regression",
                    "method": "Logistic Regression",
                    "score": 85,
                },
            ]
        },
    }

    print("Membuat Research Intelligence Engine...")

    engine = ResearchIntelligenceEngine(
        keyword_extractor=MockKeywordExtractor(),
        domain_detector=MockDomainDetector(),
        search_engine=MockSearchEngine(),
    )

    print("Menjalankan analysis...")

    result = engine.analyze(
        dataframe=dataframe,
        fingerprint=fingerprint,
        ml_result=ml_result,
        search_limit=10,
    )

    print("Analysis selesai.")

    # ==========================================================
    # BASIC RESULT
    # ==========================================================

    assert isinstance(result, dict), (
        "Result harus berupa dictionary."
    )

    assert result.get("status") == "SUCCESS", (
        "Research Intelligence gagal.\n\n"
        f"Status: {result.get('status')}\n"
        f"Error: {result.get('error')}\n\n"
        f"Result lengkap:\n{result}"
    )

    # ==========================================================
    # KEYWORDS
    # ==========================================================

    assert "keywords" in result, (
        "Result tidak memiliki field 'keywords'."
    )

    assert isinstance(result["keywords"], (dict, list)), (
        "Field 'keywords' harus berupa dict atau list."
    )

    # ==========================================================
    # DOMAIN
    # ==========================================================

    assert "domain" in result, (
        "Result tidak memiliki field 'domain'."
    )

    assert isinstance(result["domain"], dict), (
        "Field 'domain' harus berupa dictionary."
    )

    assert (
        result["domain"].get("primary_domain")
        == "transportation"
    ), (
        "Primary domain tidak sesuai."
    )

    # ==========================================================
    # QUERIES
    # ==========================================================

    assert "queries" in result, (
        "Result tidak memiliki field 'queries'."
    )

    assert isinstance(result["queries"], list), (
        "Field 'queries' harus berupa list."
    )

    assert len(result["queries"]) > 0, (
        "Minimal harus ada satu research query."
    )

    # ==========================================================
    # SEARCH
    # ==========================================================

    assert "search" in result, (
        "Result tidak memiliki field 'search'."
    )

    assert isinstance(result["search"], dict), (
        "Field 'search' harus berupa dictionary."
    )

    search_result = result["search"]

    result_count = search_result.get(
        "result_count",
        search_result.get("raw_paper_count", 0),
    )

    assert result_count > 0, (
        "Search engine seharusnya menemukan minimal satu paper."
    )

    # ==========================================================
    # PAPERS
    # ==========================================================

    assert "papers" in result, (
        "Result tidak memiliki field 'papers'."
    )

    assert isinstance(result["papers"], list), (
        "Field 'papers' harus berupa list."
    )

    assert len(result["papers"]) == 1, (
        "Mock search hanya menghasilkan satu paper."
    )

    top = result["papers"][0]

    assert isinstance(top, dict), (
        "Paper hasil ranking harus berupa dictionary."
    )

    # ==========================================================
    # PAPER TITLE
    # ==========================================================

    assert "title" in top, (
        "Paper tidak memiliki title."
    )

    assert top["title"] == (
        "Passenger Survival Prediction "
        "Using Random Forest"
    ), (
        "Judul paper tidak sesuai."
    )

    # ==========================================================
    # RELEVANCE SCORE
    # ==========================================================

    assert "relevance_score" in top, (
        "Paper tidak memiliki relevance_score."
    )

    assert isinstance(
        top["relevance_score"],
        (int, float),
    ), (
        "relevance_score harus berupa angka."
    )

    assert 0 <= top["relevance_score"] <= 100, (
        "relevance_score harus berada pada rentang 0-100."
    )

    # ==========================================================
    # RANK
    # ==========================================================

    assert "rank" in top, (
        "Paper tidak memiliki rank."
    )

    assert top["rank"] == 1, (
        "Paper pertama seharusnya memiliki rank 1."
    )

    # ==========================================================
    # SCORE BREAKDOWN
    # ==========================================================

    assert "score_breakdown" in top, (
        "Paper tidak memiliki score_breakdown."
    )

    breakdown = top["score_breakdown"]

    assert isinstance(breakdown, dict), (
        "score_breakdown harus berupa dictionary."
    )

    assert "keyword" in breakdown, (
        "score_breakdown tidak memiliki keyword score."
    )

    assert "domain" in breakdown, (
        "score_breakdown tidak memiliki domain score."
    )

    assert "target" in breakdown, (
        "score_breakdown tidak memiliki target score."
    )

    assert "method" in breakdown, (
        "score_breakdown tidak memiliki method score."
    )

    assert breakdown["method"] > 0, (
        "Paper menggunakan Random Forest, "
        "jadi method score seharusnya > 0."
    )

    # ==========================================================
    # REASONS
    # ==========================================================

    assert "reasons" in top, (
        "Paper tidak memiliki reasons."
    )

    assert isinstance(top["reasons"], list), (
        "reasons harus berupa list."
    )

    assert len(top["reasons"]) > 0, (
        "Minimal harus ada satu alasan relevansi."
    )

    # ==========================================================
    # LANDSCAPE
    # ==========================================================

    assert "landscape" in result, (
        "Result tidak memiliki field 'landscape'."
    )

    assert isinstance(result["landscape"], dict), (
        "landscape harus berupa dictionary."
    )

    assert (
        result["landscape"].get("status") == "SUCCESS"
    ), (
        "Research Landscape gagal."
    )

    # ==========================================================
    # TREND
    # ==========================================================

    assert "trend" in result, (
        "Result tidak memiliki field 'trend'."
    )

    assert isinstance(result["trend"], dict), (
        "trend harus berupa dictionary."
    )

    # ==========================================================
    # GAPS
    # ==========================================================

    assert "gaps" in result, (
        "Result tidak memiliki field 'gaps'."
    )

    assert isinstance(result["gaps"], dict), (
        "gaps harus berupa dictionary."
    )

    # ==========================================================
    # SUMMARY
    # ==========================================================

    assert "summary" in result, (
        "Result tidak memiliki field 'summary'."
    )

    assert isinstance(result["summary"], dict), (
        "summary harus berupa dictionary."
    )

    summary = result["summary"]

    assert summary.get("papers_found") == 1, (
        "Summary seharusnya menunjukkan 1 paper."
    )

    # ==========================================================
    # OUTPUT
    # ==========================================================

    print()
    print("Generated queries:")

    for query in result["queries"]:
        print(f"- {query}")

    print()
    print("Top paper:")
    print(top["title"])

    print()
    print("Relevance:", top["relevance_score"])

    print()
    print("Score breakdown:")
    for key, value in breakdown.items():
        print(f"- {key}: {value}")

    print()
    print("Reasons:")

    for reason in top["reasons"]:
        print(f"- {reason}")

    print()
    print("Summary:")
    print(summary)

    print()
    print("Research Intelligence: PASSED")


if __name__ == "__main__":

    print("=== RESEARCH INTELLIGENCE ENGINE ===")

    test_research_intelligence()

    print()
    print("================================")
    print("Semua Research Intelligence test berhasil.")
    print("================================")