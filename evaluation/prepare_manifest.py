"""Siapkan manifest benchmark 40 dataset dari metadata resmi OpenML."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import httpx

from evaluation.protocol import MANIFEST_FIELDS, SPLIT_SEED, locked_split

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "evaluation" / "manifests" / "dataset_ground_truth.csv"
API = "https://www.openml.org/api/v1/json/data/{}"

# Domain dilabel dari konteks/tags resmi OpenML. rationale disimpan agar bisa diaudit.
CURATED = {
    3: ("classification", "technology", "Games;Strategy"),
    6: ("classification", "technology", "Machine Learning"),
    11: ("classification", "social_science", "Psychology"),
    12: ("classification", "technology", "Pattern Recognition"),
    15: ("classification", "healthcare", "Health;Medicine"),
    23: ("classification", "social_science", "Sociology;Demographics"),
    28: ("classification", "technology", "Computer Vision"),
    29: ("classification", "finance", "Finance"),
    31: ("classification", "finance", "Economics;credit_scoring"),
    32: ("classification", "technology", "Computer Vision"),
    37: ("classification", "healthcare", "Health;Nutrition"),
    44: ("classification", "technology", "Computer Science;Information Retrieval"),
    46: ("classification", "healthcare", "Biology;Genetics"),
    50: ("classification", "technology", "Games;Strategy"),
    54: ("classification", "transportation", "Transport"),
    188: ("classification", "environment", "Environmental Science"),
    458: ("classification", "education", "Education;Research"),
    469: ("classification", "healthcare", "Health"),
    1063: ("classification", "technology", "Computer Science;Engineering"),
    1067: ("classification", "technology", "Computer Science;Engineering"),
    1068: ("classification", "technology", "Computer Science;Engineering"),
    1461: ("classification", "finance", "Banking;Finance"),
    1462: ("classification", "finance", "Finance;Security"),
    1464: ("classification", "healthcare", "Healthcare"),
    1468: ("classification", "business", "CNAE economic activity classification"),
    151: ("classification", "environment", "electricity;Sustainability"),
    1471: ("classification", "healthcare", "EEG;Neuroscience"),
    1472: ("regression", "environment", "Energy;Construction"),
    1475: ("classification", "technology", "AI Research;Computer Science"),
    1476: ("classification", "technology", "Sensing Technology;Chemistry"),
    1477: ("classification", "technology", "Sensing Technology;Chemistry"),
    40701: ("classification", "business", "Business;Customer Analytics"),
    40975: ("classification", "transportation", "Automobile;Transportation"),
    41138: ("classification", "technology", "UCI IDA industrial challenge"),
    41142: ("classification", "technology", "AutoML challenge dataset"),
    216: ("regression", "technology", "Aerospace;Control Theory;Robotics"),
    531: ("regression", "social_science", "Housing Economics;Urban Studies"),
    42225: ("regression", "business", "Diamond price regression dataset"),
    1049: ("classification", "technology", "PROMISE software defect dataset"),
    1050: ("classification", "technology", "PROMISE software defect dataset"),
}


def build_manifest_rows(timeout: float = 30.0) -> list[dict[str, str]]:
    """Ambil target/nama/tag langsung dari OpenML dan gabungkan label kurasi."""
    rows = []
    with httpx.Client(timeout=timeout, headers={"User-Agent": "DatasetResearchEvaluation/1.0"}) as client:
        for dataset_id, (task, domain, rationale) in CURATED.items():
            response = client.get(API.format(dataset_id))
            response.raise_for_status()
            metadata = response.json()["data_set_description"]
            if metadata.get("status") != "active" or metadata.get("visibility") != "public":
                raise ValueError(f"Dataset {dataset_id} tidak aktif atau tidak publik.")
            target = (metadata.get("default_target_attribute") or "").strip()
            if not target:
                raise ValueError(f"Target resmi OpenML tidak tersedia untuk {dataset_id}.")
            openml_url = f"https://www.openml.org/search?id={dataset_id}&type=data"
            rows.append({
                "dataset_id": f"openml:{dataset_id}",
                "dataset_name": metadata["name"],
                "source_url": metadata.get("original_data_url") or openml_url,
                "target_name": target,
                "task_label": task,
                "domain_label": domain,
                "label_source_url": API.format(dataset_id),
                "split": locked_split(f"openml:{dataset_id}"),
                "split_seed": str(SPLIT_SEED),
                "official_tags": ";".join(metadata.get("tag") or []),
                "domain_label_basis": rationale,
            })
    if len(rows) != len(CURATED) or len({row["dataset_id"] for row in rows}) != len(rows):
        raise ValueError("Manifest tidak memiliki 40 ID unik.")
    return rows


def write_manifest(output: Path, force: bool = False) -> int:
    """Tulis hasil setelah konfirmasi eksplisit bila file tujuan sudah berisi data."""
    if output.exists() and output.stat().st_size > 0 and not force:
        with output.open(encoding="utf-8-sig", newline="") as stream:
            existing = list(csv.DictReader(stream))
        if existing:
            raise FileExistsError(f"Manifest sudah berisi data; gunakan --force setelah meninjau: {output}")
    rows = build_manifest_rows()
    fields = (*MANIFEST_FIELDS, "official_tags", "domain_label_basis")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true", help="Timpa manifest berisi hanya setelah memeriksanya.")
    args = parser.parse_args()
    try:
        count = write_manifest(args.output, args.force)
    except (httpx.HTTPError, ValueError, FileExistsError, KeyError) as error:
        print(f"Gagal menyiapkan manifest: {type(error).__name__}: {error}")
        return 2
    print(f"Manifest berisi {count} dataset terverifikasi: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
