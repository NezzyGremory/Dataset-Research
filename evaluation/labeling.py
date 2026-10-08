"""Buat file penilaian blind dan simpan mapping sistem secara privat."""

from __future__ import annotations

import csv
import argparse
import hashlib
import json
import random
from pathlib import Path

from evaluation.protocol import EvaluationInputError
from evaluation.snapshots import load_snapshot

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "data" / "evaluation" / "labels"
PRIVATE_DIR = ROOT / "data" / "evaluation" / "private"


def _abstract_from_openalex(index: dict | None) -> str:
    """Rekonstruksi abstract inverted-index OpenAlex bila tersedia."""
    if not index:
        return ""
    words = []
    for token, positions in index.items():
        words.extend((int(position), token) for position in positions)
    return " ".join(token for _, token in sorted(words))


def _paper_rows(snapshot: dict) -> list[dict]:
    body = snapshot["response"]
    if snapshot["provider"] == "openalex":
        works = body.get("results", [])
        return [{
            "paper_id": item.get("doi") or item.get("id") or item.get("title", ""),
            "title": item.get("title") or "",
            "abstract": _abstract_from_openalex(item.get("abstract_inverted_index")),
            "publication_year": item.get("publication_year") or "",
        } for item in works]
    works = body.get("message", {}).get("items", [])
    rows = []
    for item in works:
        published = item.get("published-print") or item.get("published-online") or {}
        parts = published.get("date-parts", [[]])
        year = parts[0][0] if parts and parts[0] else ""
        rows.append({
            "paper_id": item.get("DOI") or item.get("URL") or "",
            "title": " ".join(item.get("title", [])),
            "abstract": item.get("abstract", ""),
            "publication_year": year,
        })
    return rows


def create_relevance_template(
    snapshot_paths: list[str | Path],
    candidate_records: list[dict] | None = None,
    seed: int = 20261008,
) -> tuple[Path, Path]:
    """Gabungkan pool aplikasi/baseline/provider dan siapkan label blind.

    Candidate record: query, system, rank, paper_id, title, abstract,
    publication_year. Paper yang sama pada query sama hanya dilabel satu kali.
    """
    pool: dict[tuple[str, str], dict] = {}
    for path in snapshot_paths:
        snapshot = load_snapshot(path)
        query_id = hashlib.sha256(snapshot["query"].encode("utf-8")).hexdigest()[:12]
        for rank, paper in enumerate(_paper_rows(snapshot), start=1):
            pool.setdefault((query_id, str(paper["paper_id"])), {
                "query_id": query_id, "query": snapshot["query"], **paper, "systems_ranks": {}
            })["systems_ranks"][f"{snapshot['provider']}_raw"] = rank
    for candidate in candidate_records or []:
        required = ("query", "system", "rank", "paper_id", "title")
        if any(not str(candidate.get(field, "")).strip() for field in required):
            raise EvaluationInputError("Candidate pool wajib berisi query, system, rank, paper_id, dan title.")
        query = str(candidate["query"])
        query_id = hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]
        key = (query_id, str(candidate["paper_id"]))
        item = pool.setdefault(key, {"query_id": query_id, "query": query,
            "paper_id": str(candidate["paper_id"]), "title": str(candidate["title"]),
            "abstract": str(candidate.get("abstract", "")),
            "publication_year": candidate.get("publication_year", ""), "systems_ranks": {}})
        system = str(candidate["system"])
        rank = int(candidate["rank"])
        previous = item["systems_ranks"].get(system)
        item["systems_ranks"][system] = min(previous, rank) if previous is not None else rank
        breakdown = candidate.get("score_breakdown")
        if isinstance(breakdown, dict):
            item.setdefault("systems_score_breakdowns", {})[system] = breakdown
    rows, mapping = [], {}
    for (query_id, paper_id), paper in pool.items():
        key_material = f"{paper_id}:{query_id}"
        blind_id = "B" + hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:12]
        for rater_id in ("rater_1", "rater_2"):
            rows.append({"blind_id": blind_id, "query_id": query_id, "title": paper["title"],
                         "abstract": paper["abstract"], "publication_year": paper["publication_year"],
                         "relevance_0_2": "", "rater_id": rater_id, "notes": ""})
        mapping[blind_id] = {"query_id": query_id, "query": paper["query"],
                             "paper_id": paper_id, "systems_ranks": paper["systems_ranks"],
                             "systems_score_breakdowns": paper.get("systems_score_breakdowns", {})}
    if not rows:
        raise EvaluationInputError("Snapshot tidak memiliki kandidat paper untuk dilabeli.")
    random.Random(seed).shuffle(rows)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    label_path = LABEL_DIR / "paper_relevance.csv"
    mapping_path = PRIVATE_DIR / "paper_relevance_key.json"
    if label_path.exists() or mapping_path.exists():
        raise EvaluationInputError("Template label/mapping sudah ada; file lama tidak ditimpa.")
    with label_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mapping_path.write_text(json.dumps({"seed": seed, "mapping": mapping}, indent=2, ensure_ascii=False), encoding="utf-8")
    return label_path, mapping_path


def create_gap_template(gap_records: list[dict], seed: int = 20261008) -> tuple[Path, Path]:
    """Buat template blind 2 penilai untuk kandidat gap hasil snapshot."""
    if not gap_records:
        raise EvaluationInputError("Tidak ada kandidat gap untuk dilabeli.")
    rows, mapping = [], {}
    for index, record in enumerate(gap_records):
        source_id = str(record.get("id") or index)
        blind_id = "G" + hashlib.sha256(f"{seed}:{source_id}".encode()).hexdigest()[:12]
        common = {"blind_id": blind_id, "topic_id": record.get("topic_id", ""),
                  "gap_title": record.get("title", ""), "gap_description": record.get("description", ""),
                  "evidence_summary": record.get("evidence", "")}
        for rater in ("rater_1", "rater_2"):
            rows.append({**common, "plausibility_1_5": "", "evidence_sufficiency_1_5": "", "rater_id": rater, "notes": ""})
        mapping[blind_id] = {"source_id": source_id, "system": record.get("system", "")}
    random.Random(seed).shuffle(rows)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    label_path = LABEL_DIR / "research_gap.csv"
    mapping_path = PRIVATE_DIR / "research_gap_key.json"
    if label_path.exists() or mapping_path.exists():
        raise EvaluationInputError("Template label/mapping sudah ada; file lama tidak ditimpa.")
    with label_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    mapping_path.write_text(json.dumps({"seed": seed, "mapping": mapping}, indent=2, ensure_ascii=False), encoding="utf-8")
    return label_path, mapping_path


def main() -> int:
    """Siapkan template label blind dari snapshot/cache dan output tersimpan."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", action="append", type=Path, default=[],
                        help="Path snapshot JSON; dapat diulang untuk tiap provider/query.")
    parser.add_argument("--candidate-pool", type=Path,
                        help="JSON array berisi output aplikasi dan baseline untuk ranking.")
    parser.add_argument("--gaps-json", type=Path,
                        help="JSON array berisi kandidat research gap dari sistem.")
    args = parser.parse_args()
    try:
        if args.gaps_json:
            records = json.loads(args.gaps_json.read_text(encoding="utf-8"))
            label_path, _ = create_gap_template(records)
            print(f"Template gap dibuat: {label_path.relative_to(ROOT)} ({len(records) * 2} baris penilaian)")
        else:
            candidates = json.loads(args.candidate_pool.read_text(encoding="utf-8")) if args.candidate_pool else []
            if not args.snapshot and not candidates:
                raise EvaluationInputError("Berikan --snapshot atau --candidate-pool.")
            label_path, _ = create_relevance_template(args.snapshot, candidates)
            print(f"Template relevansi dibuat: {label_path.relative_to(ROOT)}")
            print("Baris penilaian: dua penilai per paper unique dalam pool union.")
    except (EvaluationInputError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"TIDAK DIJALANKAN: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
