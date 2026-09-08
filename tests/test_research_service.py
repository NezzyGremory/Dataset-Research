from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from app.research.intelligence import ResearchIntelligenceEngine
from app.storage.database import Database
from app.storage.project_repository import ProjectRepository
from app.storage.research_service import ResearchService


class MockSearchEngine:

    def search(self, query, limit=20):
        return {
            "status": "SUCCESS",
            "query": query,
            "papers": [
                {
                    "title": "Passenger Survival Prediction Using Random Forest",
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

    def detect(self, text):
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

    def extract(self, dataframe, fingerprint):
        return [
            "passenger",
            "survival",
            "classification",
        ]


def build_service(tmp_path: Path) -> ResearchService:
    db_path = tmp_path / "service_test.db"

    database = Database(db_path)

    repository = ProjectRepository(database)

    engine = ResearchIntelligenceEngine(
        search_engine=MockSearchEngine(),
        domain_detector=MockDomainDetector(),
        keyword_extractor=MockKeywordExtractor(),
    )

    return ResearchService(
        engine=engine,
        repository=repository,
    )


def main():
    print()
    print("=== RESEARCH SERVICE TEST ===")
    print()

    with tempfile.TemporaryDirectory() as temp_dir:

        service = build_service(
            Path(temp_dir)
        )

        dataframe = pd.DataFrame(
            {
                "Survived": [1, 0, 1, 0],
                "Age": [22, 38, 26, 35],
                "Pclass": [3, 1, 3, 1],
            }
        )

        fingerprint = {
            "dataset_name": "Titanic",
            "rows": 4,
            "columns": 3,
            "target": "Survived",
            "problem_type": "classification",
        }

        ml_result = {
            "problem_type": "classification",
            "target_column": "Survived",
            "models": [
                {
                    "model": "Random Forest",
                    "accuracy": 0.85,
                }
            ],
        }

        # ------------------------------------------------------
        # CREATE PROJECT
        # ------------------------------------------------------

        project_id = service.create_project(
            name="Titanic Research",
            description="Research project test",
            dataset_name="Titanic.csv",
            dataset_path="datasets/Titanic.csv",
        )

        assert project_id > 0

        print("Project creation: PASSED")

        # ------------------------------------------------------
        # ANALYZE PROJECT
        # ------------------------------------------------------

        result = service.analyze_project(
            project_id=project_id,
            dataframe=dataframe,
            fingerprint=fingerprint,
            ml_result=ml_result,
        )

        assert result["status"] == "SUCCESS"
        assert "papers" in result
        assert "landscape" in result
        assert "trend" in result
        assert "gaps" in result

        print("Research analysis: PASSED")

        # ------------------------------------------------------
        # CHECK RESEARCH
        # ------------------------------------------------------

        assert service.research_exists(
            project_id
        )

        print("Research existence check: PASSED")

        # ------------------------------------------------------
        # CHECK PAPERS
        # ------------------------------------------------------

        paper_count = service.count_papers(
            project_id
        )

        print(
            f"Paper count: {paper_count}"
        )

        assert paper_count > 0

        print("Paper storage: PASSED")

        # ------------------------------------------------------
        # LOAD COMPLETE PROJECT
        # ------------------------------------------------------

        project = service.get_project_with_research(
            project_id
        )

        assert project is not None
        assert project["id"] == project_id
        assert project["research"] is not None

        research = project["research"]

        assert research["status"] == "SUCCESS"

        stored_papers = research["papers"]

        assert len(stored_papers) == paper_count

        print("Complete project reload: PASSED")

        # ------------------------------------------------------
        # CHECK PAPER DATA
        # ------------------------------------------------------

        first_paper = stored_papers[0]

        assert first_paper["title"] == (
            "Passenger Survival Prediction Using Random Forest"
        )

        assert first_paper["source"] == "Mock"

        assert first_paper["year"] == 2024

        assert first_paper["citation_count"] == 25

        print("Paper data integrity: PASSED")

        # ------------------------------------------------------
        # LIST PROJECTS
        # ------------------------------------------------------

        projects = service.list_projects()

        assert len(projects) == 1
        assert projects[0]["id"] == project_id

        print("Project listing: PASSED")

        # ------------------------------------------------------
        # UPDATE PROJECT
        # ------------------------------------------------------

        updated = service.update_project(
            project_id=project_id,
            name="Titanic Research Updated",
        )

        assert updated is True

        project = service.get_project(
            project_id
        )

        assert project is not None

        assert project["name"] == (
            "Titanic Research Updated"
        )

        print("Project update: PASSED")

        # ------------------------------------------------------
        # CHECK RESEARCH STILL EXISTS AFTER UPDATE
        # ------------------------------------------------------

        assert service.research_exists(
            project_id
        )

        assert service.count_papers(
            project_id
        ) == paper_count

        print(
            "Research persistence after update: PASSED"
        )

        # ------------------------------------------------------
        # DELETE PROJECT
        # ------------------------------------------------------

        deleted = service.delete_project(
            project_id
        )

        assert deleted is True

        assert not service.project_exists(
            project_id
        )

        print("Project delete: PASSED")

        # ------------------------------------------------------
        # CHECK CASCADE DELETE
        # ------------------------------------------------------

        assert not service.research_exists(
            project_id
        )

        assert service.count_papers(
            project_id
        ) == 0

        print("Cascade delete: PASSED")

    print()
    print("================================")
    print("Research Service: BASIC TEST PASSED")
    print("================================")
    print()


if __name__ == "__main__":
    main()