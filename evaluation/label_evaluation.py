"""Skor paper dan kandidat research gap setelah label manusia tersedia."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluation.metrics import (cohens_kappa, mean_reciprocal_rank,
                                ndcg_at_k, precision_at_k,
                                random_ranking_baseline, wilcoxon_signed_rank,
                                bootstrap_ci)
from evaluation.protocol import EvaluationInputError, require_human_labels


def score_paper_labels(label_path: Path, mapping_path: Path) -> dict:
    """Hitung metrik setiap sistem pada union pool label manusia blind."""
    required = ("blind_id", "query_id", "relevance_0_2", "rater_id")
    rows = require_human_labels(label_path, required)
    key = json.loads(mapping_path.read_text(encoding="utf-8"))
    by_item = defaultdict(list)
    for row in rows:
        try:
            rating = int(row["relevance_0_2"])
        except ValueError as error:
            raise EvaluationInputError("relevance_0_2 harus integer 0, 1, atau 2.") from error
        if rating not in (0, 1, 2):
            raise EvaluationInputError("relevance_0_2 harus integer 0, 1, atau 2.")
        by_item[row["blind_id"]].append((row["rater_id"], rating))
    relevance = {}
    ratings_a, ratings_b = [], []
    for blind_id, ratings in by_item.items():
        raters = [rater for rater, _ in ratings]
        if len(ratings) != 2 or len(set(raters)) != 2:
            raise EvaluationInputError(f"Tepat dua penilai berbeda wajib mengisi item {blind_id}.")
        ratings = sorted(ratings, key=lambda item: item[0])
        ratings_a.append(ratings[0][1]); ratings_b.append(ratings[1][1])
        relevance[blind_id] = float(np.mean([value for _, value in ratings]))
    papers_by_query = defaultdict(list)
    for blind_id, value in relevance.items():
        record = key["mapping"].get(blind_id)
        if not record:
            raise EvaluationInputError(f"ID blind tidak ditemukan di mapping privat: {blind_id}")
        papers_by_query[record["query_id"]].append({
            "blind_id": blind_id, "relevance": value,
            "systems_ranks": record.get("systems_ranks", {}),
            # Compatibility with the original single-provider mapping format.
            "legacy_system": record.get("provider"), "legacy_rank": record.get("rank"),
        })
    system_names = set()
    for papers in papers_by_query.values():
        for paper in papers:
            system_names.update(paper["systems_ranks"])
            if paper["legacy_system"]:
                system_names.add(paper["legacy_system"])
    scores = {}
    for system in sorted(system_names):
        query_scores = []
        for query_id, pool in sorted(papers_by_query.items()):
            ranked = []
            for paper in pool:
                rank = paper["systems_ranks"].get(system)
                if rank is None and paper["legacy_system"] == system:
                    rank = paper["legacy_rank"]
                if rank is not None:
                    ranked.append((int(rank), paper["relevance"]))
            if not ranked:
                continue
            ranked.sort(key=lambda item: item[0])
            values = [value for _, value in ranked]
            pool_values = [paper["relevance"] for paper in pool]
            random_order = random_ranking_baseline(len(values), seed=20261008)
            random_values = [values[index] for index in random_order]
            total_relevant = sum(value > 0 for value in pool_values)
            query_scores.append({"query_id": query_id, "candidate_count": len(values),
                                 "pool_relevant_count": total_relevant,
                                 "precision_at_10": precision_at_k(values, 10),
                                 "recall_at_10": (sum(value > 0 for value in values[:10]) / total_relevant) if total_relevant else 0.0,
                                 "mrr": mean_reciprocal_rank([values]),
                                 "ndcg_at_10": ndcg_at_k(values, 10, ideal_relevance=pool_values),
                                 "random_baseline_ndcg_at_10": ndcg_at_k(random_values, 10, ideal_relevance=pool_values)})
        scores[system] = {"query_count": len(query_scores), "per_query": query_scores,
                          "mean_precision_at_10": float(np.mean([x["precision_at_10"] for x in query_scores])) if query_scores else 0.0,
                          "mean_recall_at_10": float(np.mean([x["recall_at_10"] for x in query_scores])) if query_scores else 0.0,
                          "mean_mrr": float(np.mean([x["mrr"] for x in query_scores])) if query_scores else 0.0,
                          "mean_ndcg_at_10": float(np.mean([x["ndcg_at_10"] for x in query_scores])) if query_scores else 0.0,
                          "mean_random_baseline_ndcg_at_10": float(np.mean([x["random_baseline_ndcg_at_10"] for x in query_scores])) if query_scores else 0.0}
    paired = None
    if len(scores) >= 2:
        left, right = list(scores)[:2]
        a = {x["query_id"]: x["ndcg_at_10"] for x in scores[left]["per_query"]}
        b = {x["query_id"]: x["ndcg_at_10"] for x in scores[right]["per_query"]}
        common = sorted(a.keys() & b.keys())
        if len(common) >= 2:
            paired = {"systems": [left, right], "metric": "nDCG@10",
                      **wilcoxon_signed_rank([a[x] for x in common], [b[x] for x in common])}
    random_tests = {}
    for system, detail in scores.items():
        application = {x["query_id"]: x["ndcg_at_10"] for x in detail["per_query"]}
        random_scores = {x["query_id"]: x["random_baseline_ndcg_at_10"] for x in detail["per_query"]}
        common = sorted(application.keys() & random_scores.keys())
        if len(common) >= 2:
            random_tests[system] = {"metric": "nDCG@10", "baseline": "random_deterministic",
                                    **wilcoxon_signed_rank([application[x] for x in common],
                                                           [random_scores[x] for x in common])}
    return {"item_count": len(relevance), "inter_rater_cohens_kappa": cohens_kappa(ratings_a, ratings_b),
            "systems": scores, "paired_wilcoxon": paired,
            "system_vs_random_wilcoxon": random_tests}


def score_gap_labels(label_path: Path, mapping_path: Path) -> dict:
    """Ringkas dua rating manusia untuk kandidat gap; tidak mengarang gold gap."""
    required = ("blind_id", "plausibility_1_5", "evidence_sufficiency_1_5", "rater_id")
    rows = require_human_labels(label_path, required)
    key = json.loads(mapping_path.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for row in rows:
        try:
            plausibility = int(row["plausibility_1_5"])
            evidence = int(row["evidence_sufficiency_1_5"])
        except ValueError as error:
            raise EvaluationInputError("Rating gap harus integer 1–5.") from error
        if not 1 <= plausibility <= 5 or not 1 <= evidence <= 5:
            raise EvaluationInputError("Rating gap harus integer 1–5.")
        grouped[row["blind_id"]].append((row["rater_id"], plausibility, evidence))
    items, p_a, p_b, e_a, e_b = [], [], [], [], []
    for blind_id, ratings in grouped.items():
        raters = [rater for rater, _, _ in ratings]
        if len(ratings) != 2 or len(set(raters)) != 2:
            raise EvaluationInputError(f"Tepat dua penilai berbeda wajib mengisi kandidat gap {blind_id}.")
        ratings = sorted(ratings, key=lambda item: item[0])
        (r1, p1, e1), (r2, p2, e2) = ratings
        p_a.append(p1); p_b.append(p2); e_a.append(e1); e_b.append(e2)
        items.append({"blind_id": blind_id, "system": key["mapping"].get(blind_id, {}).get("system", "unknown"),
                      "plausibility_mean": (p1 + p2) / 2, "evidence_sufficiency_mean": (e1 + e2) / 2})
    per_system = {}
    for system in sorted({item["system"] for item in items}):
        selected = [item for item in items if item["system"] == system]
        plausibility = [item["plausibility_mean"] for item in selected]
        evidence = [item["evidence_sufficiency_mean"] for item in selected]
        per_system[system] = {"candidate_count": len(selected),
                              "mean_plausibility": bootstrap_ci(plausibility, seed=20261008),
                              "mean_evidence_sufficiency": bootstrap_ci(evidence, seed=20261008)}
    return {"candidate_count": len(items), "items": items, "per_system": per_system,
            "inter_rater_cohens_kappa_plausibility": cohens_kappa(p_a, p_b),
            "inter_rater_cohens_kappa_evidence": cohens_kappa(e_a, e_b),
            "interpretation": "Skor adalah penilaian kandidat; tidak ada label gap benar/salah yang diasumsikan."}


def score_ranking_ablation(label_path: Path, mapping_path: Path) -> dict:
    """Re-evaluate exported ranker components after removing one component at a time."""
    required = ("blind_id", "query_id", "relevance_0_2", "rater_id")
    rows = require_human_labels(label_path, required)
    key = json.loads(mapping_path.read_text(encoding="utf-8"))
    ratings = defaultdict(list)
    for row in rows:
        value = int(row["relevance_0_2"])
        if value not in (0, 1, 2):
            raise EvaluationInputError("relevance_0_2 harus integer 0, 1, atau 2.")
        ratings[row["blind_id"]].append((row["rater_id"], value))
    items_by_query = defaultdict(list)
    for blind_id, values in ratings.items():
        if len(values) != 2 or len({rater for rater, _ in values}) != 2:
            raise EvaluationInputError(f"Tepat dua penilai berbeda wajib mengisi item {blind_id}.")
        item = key.get("mapping", {}).get(blind_id)
        if not item:
            raise EvaluationInputError(f"ID blind tidak ditemukan di mapping privat: {blind_id}")
        items_by_query[item["query_id"]].append({
            "relevance": float(np.mean([value for _, value in values])),
            "systems_score_breakdowns": item.get("systems_score_breakdowns", {}),
        })
    components = {"semantic": 0.40, "dataset_name": 0.25, "schema": 0.20,
                  "keyword": 0.10, "metadata": 0.05}
    system_names = {system for items in items_by_query.values() for item in items
                    for system in item["systems_score_breakdowns"]}
    if not system_names:
        raise EvaluationInputError("Ranking ablation memerlukan score_breakdown ranker pada candidate_pool saat template dibuat.")
    per_system = {}
    for system in sorted(system_names):
        by_variant = {"full": defaultdict(list), **{f"without_{component}": defaultdict(list)
                                                    for component in components}}
        for query_id, items in sorted(items_by_query.items()):
            scored = {variant: [] for variant in by_variant}
            for item in items:
                breakdown = item["systems_score_breakdowns"].get(system)
                if not isinstance(breakdown, dict):
                    continue
                for variant in scored:
                    active = components if variant == "full" else {
                        name: weight for name, weight in components.items()
                        if variant != f"without_{name}"}
                    denominator = sum(active.values())
                    score = sum(float(breakdown.get(name, 0.0)) * weight
                                for name, weight in active.items()) / denominator
                    scored[variant].append((score, item["relevance"]))
            for variant, pairs in scored.items():
                if pairs:
                    pairs.sort(key=lambda pair: pair[0], reverse=True)
                    ideal = [relevance for _, relevance in sorted(pairs, key=lambda pair: pair[1], reverse=True)]
                    by_variant[variant][query_id].append(ndcg_at_k([rel for _, rel in pairs], 10, ideal_relevance=ideal))
        full = by_variant["full"]
        variant_summary = {}
        for variant, per_query in by_variant.items():
            scores = {query_id: float(np.mean(values)) for query_id, values in per_query.items()}
            common = sorted(scores.keys() & full.keys())
            variant_summary[variant] = {
                "query_count": len(scores), "mean_ndcg_at_10": float(np.mean(list(scores.values()))) if scores else 0.0,
                "delta_vs_full": float(np.mean([scores[q] - full[q][0] for q in common])) if common else None,
                "paired_wilcoxon_vs_full": (wilcoxon_signed_rank([full[q][0] for q in common],
                                                                  [scores[q] for q in common])
                                               if variant != "full" and len(common) >= 2 else None),
            }
        per_system[system] = variant_summary
    return {"systems": per_system,
            "interpretation": "Ablasi menghitung ulang peringkat dari score_breakdown yang diekspor; tidak mengubah bobot atau alur aplikasi."}
