"""Deterministic sanity checks for paper deduplication; no human labels assumed."""

from __future__ import annotations

from app.research.deduplication import PaperDeduplicator
from app.research.paper import Paper


def run_synthetic_dedup_check() -> dict:
    """Verify known DOI/title duplicates collapse while distinct papers remain."""
    cases = [
        ("doi_variant", [
            Paper(title="A Study of Data", doi="10.1234/ABC", abstract=""),
            Paper(title="A Study of Data: extended", doi="https://doi.org/10.1234/abc", abstract="Full abstract"),
            Paper(title="Different Work", doi="10.9999/unique"),
        ], 2),
        ("title_normalization", [
            Paper(title="A Robust Model for Data!"),
            Paper(title="a robust model for data"),
            Paper(title="A Different Model"),
        ], 2),
        ("external_id_priority", [
            Paper(title="Paper One", external_id="OPENALEX:W123"),
            Paper(title="Paper Two", external_id="openalex:w123"),
            Paper(title="Paper Two", external_id="other-id"),
        ], 2),
    ]
    results = []
    for name, papers, expected_count in cases:
        actual_count = len(PaperDeduplicator().deduplicate(papers))
        results.append({"case": name, "expected_unique_count": expected_count,
                        "actual_unique_count": actual_count, "passed": actual_count == expected_count})
    return {"fixture_type": "synthetic_known_cases", "case_count": len(results),
            "passed_count": sum(item["passed"] for item in results),
            "cases": results,
            "human_pairwise_precision_recall": "NOT_MEASURED: blind human duplicate-pair labels are required"}
