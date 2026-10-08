import csv

import pytest

from evaluation.protocol import (
    EvaluationInputError,
    load_manifest,
    locked_split,
    require_human_labels,
)


def test_locked_split_is_stable_and_seeded():
    assert locked_split("openml:1590") == locked_split("openml:1590")
    ids = [f"openml:{index}" for index in range(30)]
    assert any(locked_split(dataset_id, seed=20261008) != locked_split(dataset_id, seed=42) for dataset_id in ids)


def test_manifest_rejects_split_drift(tmp_path):
    path = tmp_path / "manifest.csv"
    wrong_split = "dev" if locked_split("d1") == "test" else "test"
    fields = ["dataset_id", "dataset_name", "source_url", "target_name", "task_label", "domain_label", "label_source_url", "split", "split_seed"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"dataset_id": "d1", "dataset_name": "D", "source_url": "https://example.org/d", "target_name": "y", "task_label": "binary_classification", "domain_label": "healthcare", "label_source_url": "https://example.org/docs", "split": wrong_split, "split_seed": "20261008"})
    with pytest.raises(EvaluationInputError, match="Split terkunci"):
        load_manifest(path)


def test_human_label_loader_stops_when_labels_are_blank(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text("blind_id,relevance_0_2\npaper-1,\n", encoding="utf-8")
    with pytest.raises(EvaluationInputError, match="label manusia belum tersedia"):
        require_human_labels(path, ("blind_id", "relevance_0_2"))
