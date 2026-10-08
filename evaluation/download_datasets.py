"""Unduh salinan dataset OpenML sesuai manifest ground truth."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

from evaluation.protocol import EvaluationInputError, load_manifest, select_split

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "evaluation" / "manifests" / "dataset_ground_truth.csv"
DATA_ROOT = ROOT / "data" / "evaluation"


def sha256_file(path: Path) -> str:
    """Hitung SHA-256 file secara bertahap."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_frame(dataset) -> pd.DataFrame:
    """Bangun DataFrame salinan dari hasil fetch_openml."""
    if getattr(dataset, "frame", None) is not None:
        return dataset.frame.copy(deep=True)
    features = dataset.data.copy(deep=True)
    target = dataset.target
    if isinstance(target, pd.DataFrame):
        for column in target.columns:
            features[column] = target[column].to_numpy(copy=True)
    else:
        features[str(target.name or "target")] = target.to_numpy(copy=True)
    return features


def download_manifest(manifest_path: Path, split: str, offline: bool = False) -> list[dict]:
    """Ambil atau gunakan cache lokal, lalu catat hash salinan CSV."""
    rows = select_split(load_manifest(manifest_path), split)
    if not rows:
        raise EvaluationInputError(f"Manifest belum berisi dataset berlabel untuk split {split.upper()}.")
    destination_dir = DATA_ROOT / "datasets"
    destination_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for row in rows:
        dataset_id = row["dataset_id"].strip()
        target_name = row["target_name"].strip()
        if not target_name:
            raise EvaluationInputError(f"Ground truth target_name belum diisi: {dataset_id}")
        output = destination_dir / f"{dataset_id.replace(':', '_')}.csv"
        if not output.exists():
            if offline:
                raise EvaluationInputError(f"TIDAK DIJALANKAN: salinan offline belum tersedia untuk {dataset_id}: {output}")
            try:
                print(f"Mengunduh {split.upper()} {dataset_id} ({row['dataset_name']})...", flush=True)
                openml_id = int(dataset_id.split(":", 1)[1] if ":" in dataset_id else dataset_id)
                try:
                    bunch = fetch_openml(data_id=openml_id, as_frame=True, parser="pandas",
                                         n_retries=1, data_home=str(DATA_ROOT / "openml_cache"))
                except ValueError as parser_error:
                    if "sparse" not in str(parser_error).lower():
                        raise
                    bunch = fetch_openml(data_id=openml_id, as_frame=True, parser="liac-arff",
                                         n_retries=1, data_home=str(DATA_ROOT / "openml_cache"))
                frame = _dataset_frame(bunch)
            except Exception as error:
                raise EvaluationInputError(f"Unduhan OpenML gagal untuk {dataset_id}: {type(error).__name__}: {error}") from error
            if target_name not in frame.columns:
                raise EvaluationInputError(f"Target resmi '{target_name}' tidak ditemukan di {dataset_id}; metadata perlu diverifikasi.")
            # Simpan salinan lokal; sumber remote tidak pernah ditulis.
            frame.to_csv(output, index=False)
        digest_before = sha256_file(output)
        frame = pd.read_csv(output)
        if target_name not in frame.columns:
            raise EvaluationInputError(f"Salinan {dataset_id} tidak memiliki target {target_name}.")
        records.append({
            "dataset_id": dataset_id,
            "dataset_name": row["dataset_name"],
            "source_url": row["source_url"],
            "label_source_url": row["label_source_url"],
            "target_name": target_name,
            "task_label": row["task_label"],
            "domain_label": row["domain_label"],
            "split": split,
            "path": str(output.relative_to(ROOT)),
            "sha256_before": digest_before,
            "sha256_after_load": sha256_file(output),
            "hash_unchanged": digest_before == sha256_file(output),
            "rows": len(frame),
            "columns": len(frame.columns),
        })
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "offline": offline,
        "datasets": records,
    }
    output_manifest = DATA_ROOT / "manifests" / f"downloaded_{split}.json"
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dev", action="store_true")
    modes.add_argument("--test", action="store_true")
    parser.add_argument("--offline", action="store_true", help="Gunakan salinan lokal saja.")
    args = parser.parse_args()
    split = "test" if args.test else "dev"
    try:
        records = download_manifest(args.manifest, split, args.offline)
    except EvaluationInputError as error:
        print(str(error))
        return 2
    print(f"Split {split.upper()}: {len(records)} dataset; SHA-256 dicatat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
