"""Kontrak manifest, split terkunci, dan pemeriksaan label evaluasi."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

SPLIT_SEED = 20261008
MANIFEST_FIELDS = (
    "dataset_id", "dataset_name", "source_url", "target_name", "task_label",
    "domain_label", "label_source_url", "split", "split_seed",
)


class EvaluationInputError(ValueError):
    """Input evaluasi belum lengkap atau tidak konsisten."""


def locked_split(dataset_id: str, seed: int = SPLIT_SEED, test_fraction: float = 0.2) -> str:
    """Tetapkan DEV/TEST dari ID dataset dan seed, tanpa bergantung urutan file."""
    if not dataset_id.strip() or not 0 < test_fraction < 1:
        raise ValueError("dataset_id wajib diisi dan test_fraction harus di antara 0 dan 1.")
    digest = hashlib.sha256(f"{seed}:{dataset_id.strip()}".encode("utf-8")).digest()
    threshold = int(test_fraction * (2**64))
    return "test" if int.from_bytes(digest[:8], "big") < threshold else "dev"


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    """Baca CSV manifest dan validasi split serta sumber label resmi."""
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise EvaluationInputError(f"Manifest belum tersedia: {manifest_path}")
    with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(MANIFEST_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise EvaluationInputError("Kolom manifest belum lengkap: " + ", ".join(sorted(missing)))
        rows = list(reader)
    seen = set()
    for line, row in enumerate(rows, start=2):
        dataset_id = row["dataset_id"].strip()
        if not dataset_id:
            raise EvaluationInputError(f"dataset_id kosong pada baris {line}.")
        for field in ("dataset_name", "source_url", "target_name", "task_label", "domain_label"):
            if not row[field].strip():
                raise EvaluationInputError(f"{field} wajib diisi pada baris {line}.")
        if dataset_id in seen:
            raise EvaluationInputError(f"dataset_id duplikat pada baris {line}: {dataset_id}")
        seen.add(dataset_id)
        if not row["label_source_url"].strip():
            raise EvaluationInputError(f"label_source_url wajib berasal dari dokumentasi dataset (baris {line}).")
        expected = locked_split(dataset_id)
        if row["split"].strip().lower() != expected or row["split_seed"].strip() != str(SPLIT_SEED):
            raise EvaluationInputError(f"Split terkunci tidak cocok pada baris {line}; untuk {dataset_id}, gunakan {expected}/{SPLIT_SEED}.")
    return rows


def select_split(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    """Pilih subset manifest tanpa mengubah pembagian yang sudah terkunci."""
    if split not in {"dev", "test"}:
        raise ValueError("split harus dev atau test.")
    return [row for row in rows if row["split"].strip().lower() == split]


def require_human_labels(path: str | Path, required_fields: tuple[str, ...]) -> list[dict[str, str]]:
    """Muat label manusia; berhenti jelas jika file belum memiliki label."""
    label_path = Path(path)
    if not label_path.exists():
        raise EvaluationInputError(f"label manusia belum tersedia: {label_path}")
    with label_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing_columns = set(required_fields) - set(reader.fieldnames or [])
        if missing_columns:
            raise EvaluationInputError("label manusia belum tersedia: kolom belum lengkap (" + ", ".join(sorted(missing_columns)) + ")")
        rows = list(reader)
    valid_rows = [row for row in rows if all((row.get(field) or "").strip() for field in required_fields)]
    if not valid_rows:
        raise EvaluationInputError(f"label manusia belum tersedia: belum ada baris berlabel pada {label_path}")
    return valid_rows
