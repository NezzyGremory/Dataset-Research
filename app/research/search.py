from __future__ import annotations

from typing import Dict, List, Optional

from app.research.crossref import CrossrefClient
from app.research.deduplication import PaperDeduplicator
from app.research.openalex import OpenAlexClient
from app.research.paper import Paper


class AcademicSearchEngine:

    def __init__(
        self,
        email: Optional[str] = None,
        openalex: Optional[OpenAlexClient] = None,
        crossref: Optional[CrossrefClient] = None,
        deduplicator: Optional[PaperDeduplicator] = None,
    ):
        self.openalex = (
            openalex
            or OpenAlexClient(email=email)
        )

        self.crossref = (
            crossref
            or CrossrefClient(email=email)
        )

        self.deduplicator = (
            deduplicator
            or PaperDeduplicator()
        )

    def search(
        self,
        query: str,
        limit: int = 20,
        use_crossref: bool = True,
    ) -> Dict:

        if not query or not query.strip():
            return {
                "status": "ERROR",
                "query": query,
                "papers": [],
                "result_count": 0,
                "sources": [],
                "message": "Query tidak boleh kosong.",
            }

        limit = max(1, min(limit, 100))

        openalex_results = self.openalex.search(
            query=query,
            per_page=limit,
        )

        all_papers = list(openalex_results)

        if use_crossref and len(all_papers) < limit:
            remaining = limit - len(all_papers)

            crossref_results = self.crossref.search(
                query=query,
                rows=remaining,
            )

            all_papers.extend(crossref_results)

        unique_papers = self.deduplicator.deduplicate(
            all_papers
        )

        unique_papers = unique_papers[:limit]

        sources = sorted(
            set(
                paper.source
                for paper in unique_papers
                if paper.source
            )
        )

        return {
            "status": "SUCCESS",
            "query": query,
            "papers": [
                paper.to_dict()
                for paper in unique_papers
            ],
            "result_count": len(unique_papers),
            "sources": sources,
            "message": (
                f"Ditemukan {len(unique_papers)} "
                f"paper setelah deduplikasi."
            ),
        }