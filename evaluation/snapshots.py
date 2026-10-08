"""Ambil dan baca snapshot OpenAlex/Crossref bertanggal."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import httpx

from evaluation.protocol import EvaluationInputError

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "data" / "evaluation" / "snapshots"


def _slug(query: str) -> str:
    """Buat nama berkas yang stabil dan aman dari teks query."""
    readable = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:48] or "query"
    return f"{readable}-{hashlib.sha256(query.encode('utf-8')).hexdigest()[:10]}"


def capture_snapshot(provider: str, query: str, limit: int = 20, output_dir: Path = SNAPSHOT_DIR) -> Path:
    """Ambil respons provider resmi dan simpan metadata reproducibility."""
    if provider not in {"openalex", "crossref"} or not query.strip():
        raise ValueError("provider harus openalex/crossref dan query wajib diisi.")
    limit = min(max(int(limit), 1), 100)
    if provider == "openalex":
        url = "https://api.openalex.org/works"
        params = {"search": query, "per-page": limit, "page": 1}
    else:
        url = "https://api.crossref.org/works"
        params = {"query": query, "rows": limit, "offset": 0}
    response = httpx.get(url, params=params, headers={"User-Agent": "DatasetResearchEvaluation/1.0"}, timeout=30)
    response.raise_for_status()
    now = datetime.now(timezone.utc)
    record = {
        "schema_version": 1,
        "provider": provider,
        "query": query,
        "requested_url": str(response.request.url),
        "fetched_at_utc": now.isoformat(),
        "snapshot_date_utc": now.date().isoformat(),
        "python_version": platform.python_version(),
        "httpx_version": httpx.__version__,
        "response_sha256": hashlib.sha256(response.content).hexdigest(),
        "response_body_base64": base64.b64encode(response.content).decode("ascii"),
        "response": response.json(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{now.date().isoformat()}-{provider}-{_slug(query)}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_snapshot(path: str | Path) -> dict:
    """Baca snapshot lokal dan pastikan hash payload masih sama."""
    snapshot_path = Path(path)
    if not snapshot_path.exists():
        raise EvaluationInputError(f"TIDAK DIJALANKAN: snapshot offline tidak ditemukan: {snapshot_path}")
    record = json.loads(snapshot_path.read_text(encoding="utf-8"))
    response = record.get("response")
    raw_body = base64.b64decode(record.get("response_body_base64", ""))
    expected = record.get("response_sha256")
    if not isinstance(response, dict) or not expected or hashlib.sha256(raw_body).hexdigest() != expected:
        raise EvaluationInputError(f"Snapshot tidak valid: {snapshot_path}")
    record["loaded_response_sha256"] = hashlib.sha256(raw_body).hexdigest()
    record["response_hash_verifiable_after_json_serialization"] = True
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("openalex", "crossref"), required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    try:
        if args.offline:
            if not args.snapshot:
                raise EvaluationInputError("Mode --offline memerlukan --snapshot <file.json>.")
            record = load_snapshot(args.snapshot)
            print(f"Snapshot {record['provider']} {record['snapshot_date_utc']}: {record['query']}")
        else:
            path = capture_snapshot(args.provider, args.query, args.limit)
            print(f"Snapshot disimpan: {path.relative_to(ROOT)}")
    except (EvaluationInputError, httpx.HTTPError, ValueError) as error:
        print(f"TIDAK DIJALANKAN: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
