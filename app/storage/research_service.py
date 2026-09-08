from __future__ import annotations

from typing import Any

import pandas as pd

from app.research.intelligence import ResearchIntelligenceEngine
from app.storage.project_repository import ProjectRepository


class ResearchService:
    """
    Application/service layer untuk Dataset Research.

    Tugas utama:
    - Membuat project
    - Menjalankan Research Intelligence
    - Menyimpan hasil research ke SQLite
    - Mengambil project beserta hasil research
    - Menghapus project

    GUI nantinya cukup berkomunikasi dengan class ini.
    """

    def __init__(
        self,
        engine: ResearchIntelligenceEngine,
        repository: ProjectRepository,
    ):
        self.engine = engine
        self.repository = repository

    # ==========================================================
    # PROJECT
    # ==========================================================

    def create_project(
        self,
        name: str,
        description: str | None = None,
        dataset_name: str | None = None,
        dataset_path: str | None = None,
    ) -> int:
        """
        Membuat project baru dan mengembalikan project_id.
        """

        name = name.strip()

        if not name:
            raise ValueError("Project name tidak boleh kosong.")

        return self.repository.create_project(
            name=name,
            description=description,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
        )

    def get_project(
        self,
        project_id: int,
    ) -> dict[str, Any] | None:
        """
        Mengambil project berdasarkan ID.
        """

        return self.repository.get_project(
            project_id
        )

    def get_project_with_research(
        self,
        project_id: int,
    ) -> dict[str, Any] | None:
        """
        Mengambil project lengkap beserta hasil research.
        """

        return self.repository.get_project_with_research(
            project_id
        )

    def list_projects(self) -> list[dict[str, Any]]:
        """
        Mengambil seluruh project.
        """

        return self.repository.list_projects()

    def update_project(
        self,
        project_id: int,
        name: str | None = None,
        description: str | None = None,
        dataset_name: str | None = None,
        dataset_path: str | None = None,
    ) -> bool:
        """
        Mengupdate informasi project.
        """

        if name is not None:
            name = name.strip()

            if not name:
                raise ValueError(
                    "Project name tidak boleh kosong."
                )

        return self.repository.update_project(
            project_id=project_id,
            name=name,
            description=description,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
        )

    def delete_project(
        self,
        project_id: int,
    ) -> bool:
        """
        Menghapus project beserta seluruh data research
        dan paper yang terhubung.
        """

        return self.repository.delete_project(
            project_id
        )

    # ==========================================================
    # RESEARCH
    # ==========================================================

    def analyze_project(
        self,
        project_id: int,
        dataframe: pd.DataFrame,
        fingerprint: dict[str, Any],
        ml_result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Menjalankan Research Intelligence untuk sebuah project
        kemudian menyimpan hasilnya ke SQLite.

        Pipeline:

        DataFrame
            ↓
        ResearchIntelligenceEngine
            ↓
        Research Result
            ↓
        SQLite
        """

        if not self.repository.project_exists(
            project_id
        ):
            raise ValueError(
                f"Project dengan ID {project_id} tidak ditemukan."
            )

        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(
                "dataframe harus berupa pandas.DataFrame."
            )

        result = self.engine.analyze(
            dataframe=dataframe,
            fingerprint=fingerprint,
            ml_result=ml_result,
        )

        self.repository.save_research_result(
            project_id=project_id,
            result=result,
        )

        return result

    def analyze_and_create_project(
        self,
        name: str,
        dataframe: pd.DataFrame,
        fingerprint: dict[str, Any],
        ml_result: dict[str, Any],
        description: str | None = None,
        dataset_name: str | None = None,
        dataset_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Membuat project lalu langsung menjalankan research.

        Cocok digunakan oleh GUI ketika user:
        1. Upload dataset
        2. Memberikan nama project
        3. Menekan tombol Research
        """

        project_id = self.create_project(
            name=name,
            description=description,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
        )

        try:
            result = self.analyze_project(
                project_id=project_id,
                dataframe=dataframe,
                fingerprint=fingerprint,
                ml_result=ml_result,
            )
        except Exception:
            # Jika research gagal, project yang baru dibuat
            # dibersihkan agar database tidak meninggalkan
            # project kosong.
            self.repository.delete_project(
                project_id
            )
            raise

        return {
            "project_id": project_id,
            "result": result,
        }

    def get_research_result(
        self,
        project_id: int,
    ) -> dict[str, Any] | None:
        """
        Mengambil hasil research dari project.
        """

        return self.repository.get_research_result(
            project_id
        )

    # ==========================================================
    # STATUS / UTILITIES
    # ==========================================================

    def project_exists(
        self,
        project_id: int,
    ) -> bool:
        """
        Mengecek apakah project tersedia.
        """

        return self.repository.project_exists(
            project_id
        )

    def research_exists(
        self,
        project_id: int,
    ) -> bool:
        """
        Mengecek apakah project sudah memiliki
        hasil research.
        """

        return self.repository.research_exists(
            project_id
        )

    def count_papers(
        self,
        project_id: int,
    ) -> int:
        """
        Menghitung jumlah paper yang tersimpan
        pada project.
        """

        return self.repository.count_papers(
            project_id
        )