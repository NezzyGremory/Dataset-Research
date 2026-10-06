from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DatasetVersion:
    """
    Representasi satu versi dataset dalam project.

    Setiap transformasi menghasilkan versi baru.
    Versi 0 selalu merupakan dataset original (raw).
    """

    id: int = 0
    project_id: int = 0
    version: int = 0
    label: str = "Raw Dataset"
    file_path: str = ""

    row_count: Optional[int] = None
    column_count: Optional[int] = None
    columns: List[str] = field(default_factory=list)
    dtypes: Dict[str, str] = field(default_factory=dict)
    file_hash: Optional[str] = None
    file_size_bytes: Optional[int] = None

    is_current: bool = False
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "version": self.version,
            "label": self.label,
            "file_path": self.file_path,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "columns": self.columns,
            "dtypes": self.dtypes,
            "file_hash": self.file_hash,
            "file_size_bytes": self.file_size_bytes,
            "is_current": self.is_current,
            "created_at": self.created_at,
        }


@dataclass
class TransformationRecord:
    """
    Satu entry dalam Research Trail.

    Mencatat operasi transformasi yang mengubah
    dataset dari satu versi ke versi berikutnya.
    """

    id: int = 0
    project_id: int = 0
    from_version: int = 0
    to_version: int = 0

    operation: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    impact: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None

    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "operation": self.operation,
            "parameters": self.parameters,
            "impact": self.impact,
            "description": self.description,
            "created_at": self.created_at,
        }
