from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class Database:
    """
    SQLite database manager untuk Dataset Research.

    Database menyimpan:
    - projects
    - research_results
    - papers

    Connection SQLite selalu ditutup secara eksplisit
    setelah operasi selesai.
    """

    def __init__(
        self,
        db_path: str | Path = "data/dataset_research.db",
    ):
        self.db_path = Path(db_path)

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize()

    # ============================================================
    # CONNECTION
    # ============================================================

    def connect(self) -> sqlite3.Connection:
        """
        Membuat koneksi SQLite baru.

        Connection DIKEMBALIKAN ke caller dan caller wajib
        menutupnya menggunakan connection.close().
        """

        connection = sqlite3.connect(
            str(self.db_path),
            timeout=30,
        )

        connection.row_factory = sqlite3.Row

        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        return connection

    # ============================================================
    # INITIALIZATION
    # ============================================================

    def _initialize(self) -> None:
        """
        Membuat seluruh tabel database jika belum tersedia.
        """

        connection = self.connect()

        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT,
                    dataset_name TEXT,
                    dataset_path TEXT,
                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS research_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    project_id INTEGER NOT NULL UNIQUE,

                    status TEXT,

                    keywords_json TEXT,
                    domain_json TEXT,
                    queries_json TEXT,
                    ml_result_json TEXT,
                    landscape_json TEXT,
                    trend_json TEXT,
                    gaps_json TEXT,
                    summary_json TEXT,

                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    updated_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY (project_id)
                        REFERENCES projects(id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    project_id INTEGER NOT NULL,

                    title TEXT NOT NULL,
                    authors_json TEXT,
                    abstract TEXT,
                    year INTEGER,
                    doi TEXT,
                    venue TEXT,
                    url TEXT,
                    citation_count INTEGER DEFAULT 0,
                    source TEXT,
                    external_id TEXT,

                    keywords_json TEXT,

                    relevance_score REAL,

                    score_breakdown_json TEXT,
                    reasons_json TEXT,

                    ranking_status TEXT,
                    rank INTEGER,

                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,

                    FOREIGN KEY (project_id)
                        REFERENCES projects(id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS
                    idx_papers_project_id
                ON papers(project_id);

                CREATE INDEX IF NOT EXISTS
                    idx_papers_project_rank
                ON papers(project_id, rank);

                CREATE INDEX IF NOT EXISTS
                    idx_research_project_id
                ON research_results(project_id);
                """
            )

            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            # SANGAT PENTING:
            # SQLite connection harus ditutup secara eksplisit.
            connection.close()

    # ============================================================
    # EXECUTE
    # ============================================================

    def execute(
        self,
        query: str,
        parameters: tuple[Any, ...] = (),
    ) -> int:
        """
        Menjalankan INSERT / UPDATE / DELETE.

        Returns:
            lastrowid jika tersedia.
        """

        connection = self.connect()

        try:
            cursor = connection.execute(
                query,
                parameters,
            )

            lastrowid = cursor.lastrowid

            connection.commit()

            return int(lastrowid or 0)

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

    # ============================================================
    # FETCH ONE
    # ============================================================

    def fetch_one(
        self,
        query: str,
        parameters: tuple[Any, ...] = (),
    ) -> sqlite3.Row | None:
        """
        Mengambil satu row dari database.
        """

        connection = self.connect()

        try:
            cursor = connection.execute(
                query,
                parameters,
            )

            row = cursor.fetchone()

            return row

        finally:
            connection.close()

    # ============================================================
    # FETCH ALL
    # ============================================================

    def fetch_all(
        self,
        query: str,
        parameters: tuple[Any, ...] = (),
    ) -> list[sqlite3.Row]:
        """
        Mengambil seluruh row dari database.
        """

        connection = self.connect()

        try:
            cursor = connection.execute(
                query,
                parameters,
            )

            rows = cursor.fetchall()

            return rows

        finally:
            connection.close()

    # ============================================================
    # DELETE DATABASE
    # ============================================================

    def delete_database(self) -> None:
        """
        Menghapus database.

        Digunakan untuk testing/reset database.
        """

        # Pastikan tidak ada connection dari object ini
        # yang masih aktif.
        #
        # Karena setiap operasi database menggunakan
        # connection lokal dan selalu close() di finally,
        # file aman untuk dihapus.

        if self.db_path.exists():
            self.db_path.unlink()

        self._initialize()