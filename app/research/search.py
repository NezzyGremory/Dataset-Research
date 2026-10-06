from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from app.research.crossref import CrossrefClient
from app.research.deduplication import PaperDeduplicator
from app.research.openalex import OpenAlexClient
from app.research.paper import Paper


class AcademicSearchEngine:
    """Multi-source academic search using open-access academic providers (OpenAlex & Crossref)."""

    def __init__(
        self,
        email: Optional[str] = None,
        openalex: Optional[OpenAlexClient] = None,
        crossref: Optional[CrossrefClient] = None,
        scholar: Any = None,
        scholar_api_key: Optional[str] = None,
        deduplicator: Optional[PaperDeduplicator] = None,
        **kwargs,
    ):
        self.openalex = openalex or OpenAlexClient(email=email)
        self.crossref = crossref or CrossrefClient(email=email)
        self.deduplicator = deduplicator or PaperDeduplicator()

    def search(
        self,
        query: str,
        limit: int = 20,
        use_crossref: bool = True,
        **kwargs,
    ) -> Dict:
        if not query or not query.strip():
            return {
                "status": "ERROR",
                "query": query,
                "papers": [],
                "result_count": 0,
                "sources": [],
                "provider_status": {},
                "message": "Query tidak boleh kosong.",
            }

        limit = max(1, min(limit, 100))

        all_papers: List[Paper] = []
        candidate_source_counts: Dict[str, int] = {}
        provider_status = {
            "OpenAlex": "ENABLED",
            "Crossref": "ENABLED" if use_crossref else "DISABLED",
        }

        # Collect candidates independently from OpenAlex and Crossref
        try:
            openalex_results = self.openalex.search(
                query=query,
                per_page=limit,
            )
            all_papers.extend(openalex_results)
            candidate_source_counts["OpenAlex"] = len(openalex_results)
        except Exception:
            provider_status["OpenAlex"] = "ERROR"

        if use_crossref:
            try:
                crossref_results = self.crossref.search(
                    query=query,
                    rows=limit,
                )
                all_papers.extend(crossref_results)
                candidate_source_counts["Crossref"] = len(crossref_results)
            except Exception:
                provider_status["Crossref"] = "ERROR"

        unique_papers = self.deduplicator.deduplicate(all_papers)
        unique_papers = unique_papers[:limit]

        sources = sorted(
            {
                paper.source
                for paper in unique_papers
                if paper.source
            }
        )

        active_sources = [
            source
            for source, status in provider_status.items()
            if status == "ENABLED"
        ]

        if not unique_papers:
            status = "PARTIAL" if any(
                value == "ERROR" for value in provider_status.values()
            ) else "SUCCESS"
        else:
            status = "SUCCESS"

        return {
            "status": status,
            "query": query,
            "papers": [paper.to_dict() for paper in unique_papers],
            "result_count": len(unique_papers),
            "sources": sources,
            "source_counts": candidate_source_counts,
            "provider_status": provider_status,
            "active_sources": active_sources,
            "candidate_count": len(all_papers),
            "message": (
                f"Ditemukan {len(unique_papers)} paper setelah deduplikasi "
                f"dari {len(all_papers)} kandidat."
            ),
        }
