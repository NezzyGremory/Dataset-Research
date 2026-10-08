"""Jalankan evaluasi Dataset Research dan simpan hasil yang dapat direproduksi."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import scipy
import sklearn

from evaluation.detection import run_detection
from evaluation.deduplication import run_synthetic_dedup_check
from evaluation.label_evaluation import (score_gap_labels, score_paper_labels,
                                         score_ranking_ablation)
from evaluation.methods import run_method_evaluation
from evaluation.metrics import (bootstrap_ci, classification_metrics,
                                cohens_kappa, mcnemar_test,
                                wilcoxon_signed_rank)
from evaluation.performance import run_performance
from evaluation.protocol import (EvaluationInputError, load_manifest,
                                 require_human_labels, select_split)
from evaluation.quality import assert_quality_quick, run_quality_quick
from evaluation.robustness import run_robustness

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evaluation" / "manifests" / "dataset_ground_truth.csv"
DATA_ROOT = ROOT / "data" / "evaluation"
RESULTS_DIR = DATA_ROOT / "results"


def _metric_self_check() -> dict:
    """Pastikan utilitas metrik mengembalikan hasil konsisten pada contoh kecil."""
    classification = classification_metrics(["a", "b", "a", "b"], ["a", "a", "a", "b"])
    ci = bootstrap_ci([1, 0, 1, 1], n_resamples=1000, seed=42)
    paired = mcnemar_test([1, 1, 0, 0], [1, 1, 0, 1], [0, 1, 1, 0])
    rank_test = wilcoxon_signed_rank([0.8, 0.7, 0.9], [0.6, 0.7, 0.5])
    return {"classification_example": classification, "bootstrap_example": ci,
            "mcnemar_example": paired, "wilcoxon_example": rank_test,
            "cohens_kappa_example": cohens_kappa([0, 1, 1], [0, 1, 0])}


def _versions() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__, "httpx": httpx.__version__}


def _skip(reason: str) -> dict:
    return {"status": "TIDAK DIJALANKAN", "reason": reason}


def run_evaluation(split: str, quick: bool, offline: bool, selected: str) -> dict:
    """Jalankan komponen yang tersedia; komponen tanpa bukti input diberi status skip."""
    report = {
        "schema_version": 1,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "quick": quick,
        "offline": offline,
        "seed": 20261008,
        "versions": _versions(),
        "modules": {},
    }
    requested = {selected} if selected != "all" else {
        "metrics", "detection", "quality", "ranking", "gap", "robustness", "performance",
        "methods", "dedup", "ranking_ablation", "gap_temporal"
    }

    def attempt(name, function):
        try:
            report["modules"][name] = {"status": "SELESAI", "result": function()}
        except (EvaluationInputError, FileNotFoundError) as error:
            report["modules"][name] = _skip(str(error))
        except Exception as error:  # Keep full-run audit moving while exposing failures.
            report["modules"][name] = {"status": "GAGAL", "error_type": type(error).__name__, "error": str(error)}

    if "metrics" in requested:
        attempt("metrics", _metric_self_check)
    if "detection" in requested:
        def detection():
            rows = load_manifest(MANIFEST)
            if len(rows) < 40:
                raise EvaluationInputError(f"TIDAK DIJALANKAN: manifest berisi {len(rows)} dataset; minimal 40 dataset dengan target/task/domain terverifikasi.")
            if split == "test" and not select_split(rows, "dev"):
                raise EvaluationInputError("TIDAK DIJALANKAN: split DEV diperlukan untuk baseline.")
            return run_detection(MANIFEST, split)
        attempt("detection", detection)
    if "quality" in requested:
        def quality():
            result = run_quality_quick()
            assert_quality_quick(result)
            return result
        attempt("quality", quality)
    if "ranking" in requested:
        def ranking():
            labels = DATA_ROOT / "labels" / "paper_relevance.csv"
            mapping = DATA_ROOT / "private" / "paper_relevance_key.json"
            require_human_labels(labels, ("blind_id", "query_id", "relevance_0_2", "rater_id"))
            if not mapping.exists():
                raise EvaluationInputError(f"mapping blind privat belum tersedia: {mapping}")
            return score_paper_labels(labels, mapping)
        attempt("ranking", ranking)
    if "gap" in requested:
        def gaps():
            labels = DATA_ROOT / "labels" / "research_gap.csv"
            mapping = DATA_ROOT / "private" / "research_gap_key.json"
            require_human_labels(labels, ("blind_id", "plausibility_1_5", "evidence_sufficiency_1_5", "rater_id"))
            if not mapping.exists():
                raise EvaluationInputError(f"mapping blind privat belum tersedia: {mapping}")
            return score_gap_labels(labels, mapping)
        attempt("gap", gaps)
    if "robustness" in requested:
        attempt("robustness", lambda: run_robustness(quick=quick))
    if "performance" in requested:
        attempt("performance", lambda: run_performance(quick=quick))
    if "methods" in requested:
        attempt("methods", lambda: run_method_evaluation(MANIFEST, quick=quick))
    if "dedup" in requested:
        attempt("dedup", run_synthetic_dedup_check)
    if "ranking_ablation" in requested:
        def ranking_ablation():
            labels = DATA_ROOT / "labels" / "paper_relevance.csv"
            mapping = DATA_ROOT / "private" / "paper_relevance_key.json"
            require_human_labels(labels, ("blind_id", "query_id", "relevance_0_2", "rater_id"))
            if not mapping.exists():
                raise EvaluationInputError(f"mapping blind privat belum tersedia: {mapping}")
            return score_ranking_ablation(labels, mapping)
        attempt("ranking_ablation", ranking_ablation)
    deferred = {
        "gap_temporal": "TIDAK DIJALANKAN: belum tersedia corpus/snapshot bertanggal multi-tahun untuk retrodiksi Y ke Y+1/Y+2.",
    }
    for name in requested & deferred.keys():
        report["modules"][name] = _skip(deferred[name])
    report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    split = parser.add_mutually_exclusive_group(required=True)
    split.add_argument("--dev", action="store_true", help="Jalankan pada split DEV.")
    split.add_argument("--test", action="store_true", help="TEST final; perlu label manusia dan input lengkap.")
    parser.add_argument("--only", choices=("all", "metrics", "detection", "quality", "ranking", "gap", "robustness", "performance", "methods", "dedup", "ranking_ablation", "gap_temporal"), default="all")
    parser.add_argument("--quick", action="store_true", help="Profil cepat; ukuran benchmark hanya 1.000 baris.")
    parser.add_argument("--offline", action="store_true", help="Tidak mengakses internet; gunakan snapshot/cache lokal.")
    args = parser.parse_args()
    which = "test" if args.test else "dev"
    if args.test:
        try:
            if args.only != "all":
                raise EvaluationInputError("TEST final hanya dijalankan sekali melalui --only all.")
            manifest_rows = load_manifest(MANIFEST)
            if len(manifest_rows) < 40 or not select_split(manifest_rows, "test"):
                raise EvaluationInputError("TIDAK DIJALANKAN: TEST memerlukan minimal 40 dataset berlabel dan split TEST terkunci.")
            require_human_labels(DATA_ROOT / "labels" / "paper_relevance.csv", ("blind_id", "query_id", "relevance_0_2", "rater_id"))
            require_human_labels(DATA_ROOT / "labels" / "research_gap.csv", ("blind_id", "plausibility_1_5", "evidence_sufficiency_1_5", "rater_id"))
            for private_file in (DATA_ROOT / "private" / "paper_relevance_key.json",
                                 DATA_ROOT / "private" / "research_gap_key.json"):
                if not private_file.exists():
                    raise EvaluationInputError(f"TIDAK DIJALANKAN: mapping blind privat belum tersedia: {private_file}")
            test_records = DATA_ROOT / "manifests" / "downloaded_test.json"
            if not test_records.exists():
                raise EvaluationInputError(f"TIDAK DIJALANKAN: cache dataset TEST belum diunduh: {test_records}")
            lock = DATA_ROOT / "test_run_consumed.json"
            if lock.exists():
                raise EvaluationInputError("TEST final sudah pernah dijalankan. Evaluasi tidak mengulang split TEST.")
            if args.quick:
                raise EvaluationInputError("TEST tidak boleh memakai mode --quick.")
        except EvaluationInputError as error:
            print(str(error))
            return 2

    report = run_evaluation(which, args.quick, args.offline, args.only)
    required_final = {"metrics", "detection", "quality", "ranking", "gap", "robustness", "performance", "methods"}
    if args.test and any(report["modules"].get(name, {}).get("status") != "SELESAI" for name in required_final):
        print("TEST belum dikunci karena masih ada modul yang TIDAK DIJALANKAN atau GAGAL.")
        print(json.dumps(report["modules"], ensure_ascii=False, indent=2, default=str))
        return 1
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"{which}_{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if args.test:
        (DATA_ROOT / "test_run_consumed.json").write_text(
            json.dumps({"consumed_at_utc": report["finished_at_utc"], "result": str(path.relative_to(ROOT))}, indent=2),
            encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"Hasil tersimpan: {path.relative_to(ROOT)}")
    return 1 if any(item["status"] == "GAGAL" for item in report["modules"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
