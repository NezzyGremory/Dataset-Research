"""Benchmark dataset target/task/domain detectors against verified labels."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.ml.target_detector import MLTargetDetector
from app.ml.task_detector import MLTaskDetector
from app.nlp.domain_detector import DomainDetector
from app.nlp.keyword_extractor import KeywordExtractor
from app.analyzer.fingerprint import DatasetFingerprint
from evaluation.download_datasets import DATA_ROOT, sha256_file
from evaluation.metrics import bootstrap_ci, classification_metrics, majority_baseline, mcnemar_test
from evaluation.protocol import EvaluationInputError, load_manifest, select_split


def run_detection(manifest_path: Path, split: str) -> dict:
    """Run target/task/domain evaluation only for selected locked split."""
    all_rows = load_manifest(manifest_path)
    rows = select_split(all_rows, split)
    if not rows:
        raise EvaluationInputError(f"TIDAK DIJALANKAN: belum ada dataset ground truth pada split {split.upper()}.")
    dev_rows = select_split(all_rows, "dev")
    if split == "test" and not dev_rows:
        raise EvaluationInputError("TIDAK DIJALANKAN: DEV diperlukan untuk baseline sebelum TEST.")

    target_detector = MLTargetDetector()
    task_detector = MLTaskDetector(target_detector=target_detector)
    keyword_extractor = KeywordExtractor()
    domain_detector = DomainDetector()
    fingerprint_engine = DatasetFingerprint()
    outputs = []
    for row in rows:
        data_path = DATA_ROOT / "datasets" / f"{row['dataset_id'].replace(':', '_')}.csv"
        checksum_path = DATA_ROOT / "manifests" / f"downloaded_{split}.json"
        if not data_path.exists() or not checksum_path.exists():
            raise EvaluationInputError(f"TIDAK DIJALANKAN: data {row['dataset_id']} belum diunduh untuk {split.upper()}.")
        manifest = json.loads(checksum_path.read_text(encoding="utf-8"))
        record = next((item for item in manifest["datasets"] if item["dataset_id"] == row["dataset_id"]), None)
        actual_hash = sha256_file(data_path)
        if not record or actual_hash != record["sha256_before"] or actual_hash != record["sha256_after_load"]:
            raise EvaluationInputError(f"Hash dataset berubah atau tidak tercatat: {row['dataset_id']}")
        frame = pd.read_csv(data_path)
        fingerprint = fingerprint_engine.generate_representation(frame)
        target_candidates = target_detector.detect(dataframe=frame, fingerprint=fingerprint)
        target_names = [item.get("name") for item in target_candidates]
        ml_result = task_detector.detect(fingerprint, dataframe=frame)
        task_name = (ml_result.get("primary_task") or {}).get("task") or "unknown_task"
        # Ground truth is the broad task family; the app may refine classification
        # into binary/multiclass, which belongs to the same labeled family.
        if task_name in {"binary_classification", "multiclass_classification"}:
            task_name = "classification"
        keywords = keyword_extractor.extract(frame, fingerprint=fingerprint, filename=row["dataset_name"])
        domain = domain_detector.detect(keywords=keywords.get("keywords", []), dataframe=frame, fingerprint=fingerprint)
        target_name = row["target_name"]
        outputs.append({
            "dataset_id": row["dataset_id"],
            "target_true": target_name,
            "target_top1": target_names[0] if target_names else "<none>",
            "target_top3_hit": target_name in target_names[:3],
            "task_true": row["task_label"],
            "task_pred": task_name,
            "domain_true": row["domain_label"],
            "domain_pred": domain.get("primary_domain") or "<none>",
            "domain_confidence": domain.get("confidence", 0.0),
            "sha256_before": actual_hash,
            "sha256_after": sha256_file(data_path),
        })
    if any(row["sha256_before"] != row["sha256_after"] for row in outputs):
        raise AssertionError("Hash dataset berubah selama evaluasi.")

    task_truth = [item["task_true"] for item in outputs]
    task_pred = [item["task_pred"] for item in outputs]
    domain_truth = [item["domain_true"] for item in outputs]
    domain_pred = [item["domain_pred"] for item in outputs]
    task_baseline = majority_baseline([r["task_label"] for r in dev_rows] if split == "test" else [r["task_label"] for r in rows], len(rows))
    domain_baseline = majority_baseline([r["domain_label"] for r in dev_rows] if split == "test" else [r["domain_label"] for r in rows], len(rows))
    task_metrics = classification_metrics(task_truth, task_pred)
    domain_metrics = classification_metrics(domain_truth, domain_pred)
    task_base_metrics = classification_metrics(task_truth, task_baseline)
    domain_base_metrics = classification_metrics(domain_truth, domain_baseline)
    task_ci = bootstrap_ci([int(a == b) for a, b in zip(task_truth, task_pred)], seed=42)
    domain_ci = bootstrap_ci([int(a == b) for a, b in zip(domain_truth, domain_pred)], seed=42)
    return {
        "split": split,
        "dataset_count": len(outputs),
        "target": {
            "top1_accuracy": sum(x["target_true"] == x["target_top1"] for x in outputs) / len(outputs),
            "top3_hit_rate": sum(x["target_top3_hit"] for x in outputs) / len(outputs),
            "top3_hit_ci95": bootstrap_ci([int(x["target_top3_hit"]) for x in outputs], seed=42),
        },
        "task": {"application": task_metrics, "majority_baseline": task_base_metrics,
                 "paired_mcnemar": mcnemar_test(task_truth, task_pred, task_baseline), "accuracy_ci95": task_ci},
        "domain": {"application": domain_metrics, "majority_baseline": domain_base_metrics,
                   "paired_mcnemar": mcnemar_test(domain_truth, domain_pred, domain_baseline), "accuracy_ci95": domain_ci},
        "per_dataset": outputs,
    }
