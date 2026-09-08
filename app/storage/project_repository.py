from __future__ import annotations

import json
from typing import Any

from app.storage.database import Database


class ProjectRepository:
    """
    Repository untuk mengelola Project dan Research Result.

    Struktur data Research Intelligence yang didukung:

    result = {
        "status": ...,
        "keywords": ...,
        "domain": ...,
        "queries": ...,
        "search": ...,
        "papers": [...],
        "top_papers": [...],
        "landscape": ...,
        "trend": ...,
        "gaps": ...,
        "summary": ...,
        "transparency": ...
    }
    """

    def __init__(self, database: Database):
        self.database = database

    # ============================================================
    # JSON HELPERS
    # ============================================================

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )

    @staticmethod
    def _from_json(
        value: str | None,
        default: Any = None,
    ) -> Any:
        if value is None:
            return default

        try:
            return json.loads(value)
        except (
            json.JSONDecodeError,
            TypeError,
        ):
            return default

    # ============================================================
    # PROJECT CRUD
    # ============================================================

    def create_project(
        self,
        name: str,
        description: str | None = None,
        dataset_name: str | None = None,
        dataset_path: str | None = None,
    ) -> int:

        return self.database.execute(
            """
            INSERT INTO projects (
                name,
                description,
                dataset_name,
                dataset_path
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                name,
                description,
                dataset_name,
                dataset_path,
            ),
        )

    def get_project(
        self,
        project_id: int,
    ) -> dict[str, Any] | None:

        row = self.database.fetch_one(
            """
            SELECT
                id,
                name,
                description,
                dataset_name,
                dataset_path,
                created_at,
                updated_at
            FROM projects
            WHERE id = ?
            """,
            (project_id,),
        )

        if row is None:
            return None

        return dict(row)

    def list_projects(self) -> list[dict[str, Any]]:

        rows = self.database.fetch_all(
            """
            SELECT
                id,
                name,
                description,
                dataset_name,
                dataset_path,
                created_at,
                updated_at
            FROM projects
            ORDER BY updated_at DESC, id DESC
            """
        )

        return [
            dict(row)
            for row in rows
        ]

    def update_project(
        self,
        project_id: int,
        name: str | None = None,
        description: str | None = None,
        dataset_name: str | None = None,
        dataset_path: str | None = None,
    ) -> None:

        current = self.get_project(project_id)

        if current is None:
            raise ValueError(
                f"Project dengan ID {project_id} tidak ditemukan."
            )

        final_name = (
            name
            if name is not None
            else current["name"]
        )

        final_description = (
            description
            if description is not None
            else current["description"]
        )

        final_dataset_name = (
            dataset_name
            if dataset_name is not None
            else current["dataset_name"]
        )

        final_dataset_path = (
            dataset_path
            if dataset_path is not None
            else current["dataset_path"]
        )

        self.database.execute(
            """
            UPDATE projects
            SET
                name = ?,
                description = ?,
                dataset_name = ?,
                dataset_path = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                final_name,
                final_description,
                final_dataset_name,
                final_dataset_path,
                project_id,
            ),
        )

    def delete_project(
        self,
        project_id: int,
    ) -> None:

        self.database.execute(
            """
            DELETE FROM projects
            WHERE id = ?
            """,
            (project_id,),
        )

    # ============================================================
    # RESEARCH RESULT
    # ============================================================

    def save_research_result(
        self,
        project_id: int,
        result: dict[str, Any],
    ) -> None:
        """
        Menyimpan hasil Research Intelligence.

        Paper utama diambil dari:

            result["papers"]

        Jika result["papers"] kosong, repository akan mencoba:

            result["top_papers"]

        Hal ini membuat storage lebih tahan terhadap variasi
        struktur result dari Research Intelligence Engine.
        """

        if not self.project_exists(project_id):
            raise ValueError(
                f"Project dengan ID {project_id} tidak ditemukan."
            )

        if not isinstance(result, dict):
            raise TypeError(
                "result harus berupa dictionary."
            )

        status = result.get("status")

        keywords = result.get(
            "keywords",
            [],
        )

        domain = result.get(
            "domain",
            {},
        )

        queries = result.get(
            "queries",
            [],
        )

        search = result.get(
            "search",
            {},
        )

        papers = result.get(
            "papers",
            [],
        )

        top_papers = result.get(
            "top_papers",
            [],
        )

        landscape = result.get(
            "landscape",
            {},
        )

        trend = result.get(
            "trend",
            {},
        )

        gaps = result.get(
            "gaps",
            {},
        )

        summary = result.get(
            "summary",
            {},
        )

        transparency = result.get(
            "transparency",
            {},
        )

        # --------------------------------------------------------
        # Normalisasi paper
        # --------------------------------------------------------

        if not isinstance(papers, list):
            papers = list(papers) if papers else []

        if not isinstance(top_papers, list):
            top_papers = (
                list(top_papers)
                if top_papers
                else []
            )

        # Jika papers kosong tetapi top_papers tersedia,
        # gunakan top_papers sebagai sumber penyimpanan.
        if not papers and top_papers:
            papers = top_papers

        # --------------------------------------------------------
        # Research Result
        # --------------------------------------------------------

        existing = self.database.fetch_one(
            """
            SELECT id
            FROM research_results
            WHERE project_id = ?
            """,
            (project_id,),
        )

        research_payload = self._json(
            {
                "search": search,
                "top_papers": top_papers,
                "transparency": transparency,
            }
        )

        if existing is None:

            self.database.execute(
                """
                INSERT INTO research_results (
                    project_id,
                    status,
                    keywords_json,
                    domain_json,
                    queries_json,
                    ml_result_json,
                    landscape_json,
                    trend_json,
                    gaps_json,
                    summary_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    status,
                    self._json(keywords),
                    self._json(domain),
                    self._json(queries),
                    research_payload,
                    self._json(landscape),
                    self._json(trend),
                    self._json(gaps),
                    self._json(summary),
                ),
            )

        else:

            self.database.execute(
                """
                UPDATE research_results
                SET
                    status = ?,
                    keywords_json = ?,
                    domain_json = ?,
                    queries_json = ?,
                    ml_result_json = ?,
                    landscape_json = ?,
                    trend_json = ?,
                    gaps_json = ?,
                    summary_json = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE project_id = ?
                """,
                (
                    status,
                    self._json(keywords),
                    self._json(domain),
                    self._json(queries),
                    research_payload,
                    self._json(landscape),
                    self._json(trend),
                    self._json(gaps),
                    self._json(summary),
                    project_id,
                ),
            )

        # --------------------------------------------------------
        # Papers
        # --------------------------------------------------------

        self.delete_papers(project_id)

        saved_paper_count = 0

        for paper in papers:

            if not isinstance(paper, dict):
                continue

            title = paper.get(
                "title",
                "",
            )

            if title is None:
                title = ""

            title = str(title).strip()

            if not title:
                continue

            paper_data = dict(paper)

            paper_data["title"] = title

            self.save_paper(
                project_id=project_id,
                paper=paper_data,
            )

            saved_paper_count += 1

        # --------------------------------------------------------
        # Update Project Timestamp
        # --------------------------------------------------------

        self.database.execute(
            """
            UPDATE projects
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (project_id,),
        )

    # ============================================================
    # GET RESEARCH RESULT
    # ============================================================

    def get_research_result(
        self,
        project_id: int,
    ) -> dict[str, Any] | None:

        row = self.database.fetch_one(
            """
            SELECT
                id,
                project_id,
                status,
                keywords_json,
                domain_json,
                queries_json,
                ml_result_json,
                landscape_json,
                trend_json,
                gaps_json,
                summary_json,
                created_at,
                updated_at
            FROM research_results
            WHERE project_id = ?
            """,
            (project_id,),
        )

        if row is None:
            return None

        ml_result = self._from_json(
            row["ml_result_json"],
            {},
        )

        if not isinstance(ml_result, dict):
            ml_result = {}

        result = {
            "id": row["id"],
            "project_id": row["project_id"],
            "status": row["status"],
            "keywords": self._from_json(
                row["keywords_json"],
                [],
            ),
            "domain": self._from_json(
                row["domain_json"],
                {},
            ),
            "queries": self._from_json(
                row["queries_json"],
                [],
            ),
            "search": ml_result.get(
                "search",
                {},
            ),
            "top_papers": ml_result.get(
                "top_papers",
                [],
            ),
            "transparency": ml_result.get(
                "transparency",
                {},
            ),
            "landscape": self._from_json(
                row["landscape_json"],
                {},
            ),
            "trend": self._from_json(
                row["trend_json"],
                {},
            ),
            "gaps": self._from_json(
                row["gaps_json"],
                {},
            ),
            "summary": self._from_json(
                row["summary_json"],
                {},
            ),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

        result["papers"] = self.get_papers(
            project_id
        )

        return result

    # ============================================================
    # PAPER STORAGE
    # ============================================================

    def save_paper(
        self,
        project_id: int,
        paper: dict[str, Any],
    ) -> int:

        if not isinstance(paper, dict):
            raise TypeError(
                "paper harus berupa dictionary."
            )

        title = paper.get(
            "title",
            "",
        )

        if title is None:
            title = ""

        title = str(title).strip()

        if not title:
            raise ValueError(
                "Paper harus memiliki title."
            )

        return self.database.execute(
            """
            INSERT INTO papers (
                project_id,
                title,
                authors_json,
                abstract,
                year,
                doi,
                venue,
                url,
                citation_count,
                source,
                external_id,
                keywords_json,
                relevance_score,
                score_breakdown_json,
                reasons_json,
                ranking_status,
                rank
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                project_id,
                title,
                self._json(
                    paper.get(
                        "authors",
                        [],
                    )
                ),
                paper.get("abstract"),
                paper.get("year"),
                paper.get("doi"),
                paper.get("venue"),
                paper.get("url"),
                paper.get(
                    "citation_count",
                    0,
                ),
                paper.get("source"),
                paper.get("external_id"),
                self._json(
                    paper.get(
                        "keywords",
                        [],
                    )
                ),
                paper.get(
                    "relevance_score"
                ),
                self._json(
                    paper.get(
                        "score_breakdown",
                        {},
                    )
                ),
                self._json(
                    paper.get(
                        "reasons",
                        [],
                    )
                ),
                paper.get(
                    "ranking_status"
                ),
                paper.get("rank"),
            ),
        )

    def get_papers(
        self,
        project_id: int,
    ) -> list[dict[str, Any]]:

        rows = self.database.fetch_all(
            """
            SELECT
                id,
                project_id,
                title,
                authors_json,
                abstract,
                year,
                doi,
                venue,
                url,
                citation_count,
                source,
                external_id,
                keywords_json,
                relevance_score,
                score_breakdown_json,
                reasons_json,
                ranking_status,
                rank,
                created_at
            FROM papers
            WHERE project_id = ?
            ORDER BY
                CASE
                    WHEN rank IS NULL THEN 1
                    ELSE 0
                END,
                rank ASC,
                relevance_score DESC,
                id ASC
            """,
            (project_id,),
        )

        papers = []

        for row in rows:

            papers.append(
                {
                    "id": row["id"],
                    "project_id": row["project_id"],
                    "title": row["title"],
                    "authors": self._from_json(
                        row["authors_json"],
                        [],
                    ),
                    "abstract": row["abstract"],
                    "year": row["year"],
                    "doi": row["doi"],
                    "venue": row["venue"],
                    "url": row["url"],
                    "citation_count": row[
                        "citation_count"
                    ],
                    "source": row["source"],
                    "external_id": row[
                        "external_id"
                    ],
                    "keywords": self._from_json(
                        row["keywords_json"],
                        [],
                    ),
                    "relevance_score": row[
                        "relevance_score"
                    ],
                    "score_breakdown": self._from_json(
                        row[
                            "score_breakdown_json"
                        ],
                        {},
                    ),
                    "reasons": self._from_json(
                        row["reasons_json"],
                        [],
                    ),
                    "ranking_status": row[
                        "ranking_status"
                    ],
                    "rank": row["rank"],
                    "created_at": row[
                        "created_at"
                    ],
                }
            )

        return papers

    def get_paper(
        self,
        paper_id: int,
    ) -> dict[str, Any] | None:

        row = self.database.fetch_one(
            """
            SELECT
                id,
                project_id,
                title,
                authors_json,
                abstract,
                year,
                doi,
                venue,
                url,
                citation_count,
                source,
                external_id,
                keywords_json,
                relevance_score,
                score_breakdown_json,
                reasons_json,
                ranking_status,
                rank,
                created_at
            FROM papers
            WHERE id = ?
            """,
            (paper_id,),
        )

        if row is None:
            return None

        return {
            "id": row["id"],
            "project_id": row["project_id"],
            "title": row["title"],
            "authors": self._from_json(
                row["authors_json"],
                [],
            ),
            "abstract": row["abstract"],
            "year": row["year"],
            "doi": row["doi"],
            "venue": row["venue"],
            "url": row["url"],
            "citation_count": row[
                "citation_count"
            ],
            "source": row["source"],
            "external_id": row[
                "external_id"
            ],
            "keywords": self._from_json(
                row["keywords_json"],
                [],
            ),
            "relevance_score": row[
                "relevance_score"
            ],
            "score_breakdown": self._from_json(
                row[
                    "score_breakdown_json"
                ],
                {},
            ),
            "reasons": self._from_json(
                row["reasons_json"],
                [],
            ),
            "ranking_status": row[
                "ranking_status"
            ],
            "rank": row["rank"],
            "created_at": row[
                "created_at"
            ],
        }

    def delete_papers(
        self,
        project_id: int,
    ) -> None:

        self.database.execute(
            """
            DELETE FROM papers
            WHERE project_id = ?
            """,
            (project_id,),
        )

    # ============================================================
    # COMPLETE PROJECT
    # ============================================================

    def get_project_with_research(
        self,
        project_id: int,
    ) -> dict[str, Any] | None:

        project = self.get_project(
            project_id
        )

        if project is None:
            return None

        project["research"] = (
            self.get_research_result(
                project_id
            )
        )

        return project

    # ============================================================
    # UTILITIES
    # ============================================================

    def project_exists(
        self,
        project_id: int,
    ) -> bool:

        row = self.database.fetch_one(
            """
            SELECT 1
            FROM projects
            WHERE id = ?
            LIMIT 1
            """,
            (project_id,),
        )

        return row is not None

    def research_exists(
        self,
        project_id: int,
    ) -> bool:

        row = self.database.fetch_one(
            """
            SELECT 1
            FROM research_results
            WHERE project_id = ?
            LIMIT 1
            """,
            (project_id,),
        )

        return row is not None

    def count_papers(
        self,
        project_id: int,
    ) -> int:

        row = self.database.fetch_one(
            """
            SELECT COUNT(*) AS count
            FROM papers
            WHERE project_id = ?
            """,
            (project_id,),
        )

        if row is None:
            return 0

        return int(row["count"])