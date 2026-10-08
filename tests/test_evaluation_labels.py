import csv
import json

from evaluation.label_evaluation import score_gap_labels, score_paper_labels


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_paper_labels_score_blind_systems_by_rank_and_query(tmp_path):
    labels, mapping = tmp_path / "paper.csv", tmp_path / "paper_key.json"
    rows, private = [], {}
    for provider in ("openalex", "crossref"):
        for query_index in range(2):
            query_id = f"q{query_index}"
            blind_id = f"{provider}-{query_id}"
            private[blind_id] = {"provider": provider, "query_id": query_id, "rank": 1, "paper_id": blind_id}
            for rater, relevance in (("rater_1", 2), ("rater_2", 2)):
                rows.append({"blind_id": blind_id, "query_id": query_id,
                             "relevance_0_2": relevance, "rater_id": rater})
    _write_csv(labels, rows)
    mapping.write_text(json.dumps({"mapping": private}), encoding="utf-8")
    result = score_paper_labels(labels, mapping)
    assert result["item_count"] == 4
    assert result["inter_rater_cohens_kappa"] == 1.0
    assert result["systems"]["openalex"]["mean_mrr"] == 1.0
    assert result["paired_wilcoxon"]["n_pairs"] == 2


def test_gap_labels_report_agreement_without_claiming_ground_truth(tmp_path):
    labels, mapping = tmp_path / "gaps.csv", tmp_path / "gap_key.json"
    _write_csv(labels, [
        {"blind_id": "g1", "plausibility_1_5": 4, "evidence_sufficiency_1_5": 3, "rater_id": "rater_1"},
        {"blind_id": "g1", "plausibility_1_5": 4, "evidence_sufficiency_1_5": 3, "rater_id": "rater_2"},
    ])
    mapping.write_text(json.dumps({"mapping": {"g1": {"system": "application"}}}), encoding="utf-8")
    result = score_gap_labels(labels, mapping)
    assert result["candidate_count"] == 1
    assert result["inter_rater_cohens_kappa_plausibility"] == 1.0
    assert "tidak ada label gap benar/salah" in result["interpretation"]
