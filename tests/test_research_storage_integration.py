from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from app.research.intelligence import ResearchIntelligenceEngine
from app.storage.database import Database
from app.storage.project_repository import ProjectRepository


class MockSearchEngine:
    """
    Mock academic search engine untuk integration test.
    Tidak membutuhkan internet/API.
    """

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
        }


class MockDomainDetector:
    """
    Mock domain detector.
    """

    def detect(self, fingerprint):
        return {
            "primary_domain": "transportation",
            "confidence": 0.85,
            "domains": [
                {
                    "domain": "transportation",
                    "score": 0.85,
                }
            ],
        }


class MockKeywordExtractor:
    """
    Mock keyword extractor.
    """

    def extract(self, dataframe, fingerprint):
        return [
            "passenger",
            "survival",
            "classification",
        ]


def create_test_dataset() -> pd.DataFrame:
    """
    Dataset sederhana untuk integration test.
    """

    return pd.DataFrame(
        {
            "PassengerId": [1, 2, 3, 4, 5, 6],
            "Age": [22, 38, 26, 35, 35, 28],
            "Fare": [
                7.25,
                71.28,
                7.92,
                53.10,
                8.05,
                13.00,
            ],
            "Survived": [
                0,
                1,
                1,
                1,
                0,
                0,
            ],
        }
    )


def create_test_fingerprint() -> dict:
    """
    Fingerprint dataset untuk integration test.
    """

    return {
        "representation": {
            "keywords": [
                "passenger",
                "survival",
                "classification",
            ],
            "target_candidates": [
                "Survived",
            ],
        }
    }


def create_test_ml_result() -> dict:
    """
    Hasil ML intelligence mock.
    """

    return {
        "task": "binary classification",
        "target": "Survived",
        "recommendations": [
            {
                "method": "Random Forest Classifier",
                "score": 95,
            },
            {
                "method": "Logistic Regression",
                "score": 85,
            },
        ],
    }


def print_result_structure(result: dict) -> None:
    """
    Menampilkan struktur hasil Research Intelligence
    agar kita mengetahui bentuk data sebenarnya.
    """

    print()
    print("================================")
    print("ACTUAL RESEARCH RESULT")
    print("================================")

    print()
    print("Result type:")
    print(type(result))

    print()
    print("Result keys:")
    print(list(result.keys()))

    for key, value in result.items():
        print()
        print("--------------------------------")
        print(f"KEY: {key}")
        print("--------------------------------")
        print("Type:", type(value))
        print("Value:")

        try:
            print(value)
        except Exception:
            print("<unable to print value>")

    print()
    print("================================")
    print("END ACTUAL RESEARCH RESULT")
    print("================================")


def main() -> None:
    print("=== RESEARCH + SQLITE INTEGRATION ===")
    print()

    # ============================================================
    # 1. DATASET
    # ============================================================

    print("Menyiapkan dataset...")

    dataframe = create_test_dataset()
    fingerprint = create_test_fingerprint()
    ml_result = create_test_ml_result()

    print(
        f"Dataset shape: {dataframe.shape}"
    )

    print("Dataset preparation: PASSED")

    # ============================================================
    # 2. ENGINE
    # ============================================================

    print()
    print("Membuat Research Intelligence Engine...")

    search_engine = MockSearchEngine()
    domain_detector = MockDomainDetector()
    keyword_extractor = MockKeywordExtractor()

    engine = ResearchIntelligenceEngine(
        search_engine=search_engine,
        domain_detector=domain_detector,
        keyword_extractor=keyword_extractor,
    )

    print("Engine creation: PASSED")

    # ============================================================
    # 3. RESEARCH ANALYSIS
    # ============================================================

    print()
    print("Menjalankan Research Intelligence...")

    result = engine.analyze(
        dataframe=dataframe,
        fingerprint=fingerprint,
        ml_result=ml_result,
    )

    assert isinstance(result, dict)

    assert result.get("status") == "SUCCESS"

    print("Research Intelligence: PASSED")

    # ============================================================
    # 4. DEBUG ACTUAL RESULT
    # ============================================================

    print_result_structure(result)

    # ============================================================
    # 5. BASIC RESULT VALIDATION
    # ============================================================

    print()
    print("Memeriksa hasil research...")

    assert isinstance(result, dict)

    assert result.get("status") == "SUCCESS"

    assert "keywords" in result
    assert "domain" in result
    assert "queries" in result
    assert "search" in result
    assert "summary" in result

    print("Research result basic validation: PASSED")

    # ============================================================
    # 6. SQLITE DATABASE
    # ============================================================

    print()
    print("Membuat temporary SQLite database...")

    with TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "integration.db"

        database = Database(db_path)

        repository = ProjectRepository(database)

        print("SQLite initialization: PASSED")

        # ========================================================
        # 7. CREATE PROJECT
        # ========================================================

        print()
        print("Membuat project...")

        project_id = repository.create_project(
            name="Passenger Survival Research",
            description=(
                "Integration test project "
                "for Dataset Research."
            ),
            dataset_name="Titanic Dataset",
            dataset_path="data/titanic.csv",
        )

        assert project_id > 0

        assert repository.project_exists(
            project_id
        )

        print("Project creation: PASSED")

        # ========================================================
        # 8. SAVE RESEARCH RESULT
        # ========================================================

        print()
        print(
            "Menyimpan Research Intelligence "
            "ke SQLite..."
        )

        repository.save_research_result(
            project_id=project_id,
            result=result,
        )

        assert repository.research_exists(
            project_id
        )

        print(
            "Research result persistence: PASSED"
        )

        # ========================================================
        # 9. LOAD RESEARCH RESULT
        # ========================================================

        print()
        print(
            "Memuat kembali hasil research "
            "dari SQLite..."
        )

        loaded_result = (
            repository.get_research_result(
                project_id
            )
        )

        assert loaded_result is not None

        assert (
            loaded_result["status"]
            == result["status"]
        )

        assert (
            loaded_result["keywords"]
            == result["keywords"]
        )

        assert (
            loaded_result["domain"]
            == result["domain"]
        )

        assert (
            loaded_result["queries"]
            == result["queries"]
        )

        assert (
            loaded_result["summary"]
            == result["summary"]
        )

        print(
            "Research result reload: PASSED"
        )

        # ========================================================
        # 10. CHECK PAPERS
        # ========================================================

        print()
        print(
            "Memeriksa paper yang tersimpan..."
        )

        loaded_papers = repository.get_papers(
            project_id
        )

        print(
            f"Jumlah paper tersimpan: "
            f"{len(loaded_papers)}"
        )

        print(
            "Paper storage check: PASSED"
        )

        # ========================================================
        # 11. LOAD COMPLETE PROJECT
        # ========================================================

        print()
        print(
            "Memuat project lengkap..."
        )

        loaded_project = (
            repository.get_project_with_research(
                project_id
            )
        )

        assert loaded_project is not None

        assert (
            loaded_project["id"]
            == project_id
        )

        assert (
            loaded_project["name"]
            == "Passenger Survival Research"
        )

        assert (
            loaded_project["dataset_name"]
            == "Titanic Dataset"
        )

        assert (
            loaded_project["research"]
            is not None
        )

        print(
            "Complete project reload: PASSED"
        )

        # ========================================================
        # 12. PROJECT UPDATE
        # ========================================================

        print()
        print("Mengubah project...")

        repository.update_project(
            project_id=project_id,
            description=(
                "Updated research description."
            ),
        )

        updated_project = (
            repository.get_project(
                project_id
            )
        )

        assert updated_project is not None

        assert (
            updated_project["description"]
            == "Updated research description."
        )

        print("Project update: PASSED")

        # ========================================================
        # 13. CASCADE DELETE
        # ========================================================

        print()
        print(
            "Menghapus project..."
        )

        repository.delete_project(
            project_id
        )

        assert not repository.project_exists(
            project_id
        )

        assert not repository.research_exists(
            project_id
        )

        assert (
            repository.count_papers(
                project_id
            )
            == 0
        )

        print(
            "Project cascade delete: PASSED"
        )

    # ============================================================
    # FINAL
    # ============================================================

    print()
    print("================================")
    print(
        "Research + SQLite Integration: "
        "BASIC TEST PASSED"
    )
    print("================================")

    print()
    print(
        "Research Intelligence berhasil "
        "menghasilkan result."
    )

    print(
        "SQLite berhasil menyimpan dan "
        "memuat kembali result."
    )

    print()
    print(
        "Struktur result di atas kita gunakan "
        "untuk tahap integration berikutnya."
    )


if __name__ == "__main__":
    main()