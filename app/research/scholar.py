from __future__ import annotations

import os
import re
from typing import List, Optional

import httpx

from app.research.paper import Paper


class ScholarClient:
    """Optional Google Scholar search client through SerpApi.

    Google Scholar itself does not expose an official public API. This client
    uses SerpApi's Google Scholar endpoint when an API key is configured.
    Without a key, search() safely returns an empty list.
    """

    BASE_URL = "https://serpapi.com/search"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 20.0,
        hl: str = "en",
    ):
        self.api_key = (
            api_key
            or os.getenv("SERPAPI_API_KEY")
            or os.getenv("SERPAPI_KEY")
        )
        self.timeout = timeout
        self.hl = hl

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def search(
        self,
        query: str,
        rows: int = 20,
        start: int = 0,
    ) -> List[Paper]:
        if not query or not query.strip() or not self.enabled:
            return []

        params = {
            "engine": "google_scholar",
            "q": query.strip(),
            "api_key": self.api_key,
            "hl": self.hl,
            "num": min(max(rows, 1), 20),
            "start": max(start, 0),
        }

        try:
            response = httpx.get(
                self.BASE_URL,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError):
            return []

        if not isinstance(data, dict):
            return []

        results = data.get("organic_results", [])

        return [
            self._parse_result(item)
            for item in results
            if isinstance(item, dict) and item.get("title")
        ]

    def _parse_result(self, item: dict) -> Paper:
        publication_info = item.get("publication_info") or {}
        if not isinstance(publication_info, dict):
            publication_info = {}

        authors = self._parse_authors(publication_info.get("authors"))
        summary = str(publication_info.get("summary") or "")

        year = self._extract_year(summary)
        venue = self._extract_venue(summary, year)

        cited_by = (item.get("inline_links") or {}).get("cited_by") or {}
        citation_count = cited_by.get("total", 0) or 0

        result_id = item.get("result_id")
        url = item.get("link")

        doi = self._extract_doi(item)
        if doi and not url:
            url = f"https://doi.org/{doi}"

        # Google Scholar returns a search snippet rather than a guaranteed
        # publisher abstract. Keep it in Paper.abstract so downstream ranking
        # can use available text, but do not claim it is a formal abstract.
        snippet = str(item.get("snippet") or "")

        return Paper(
            title=str(item.get("title") or "Untitled"),
            authors=authors,
            abstract=snippet,
            year=year,
            doi=doi,
            venue=venue,
            url=url,
            citation_count=self._safe_int(citation_count),
            source="Google Scholar",
            external_id=result_id or url,
            keywords=[],
        )

    @staticmethod
    def _parse_authors(value) -> List[str]:
        if not isinstance(value, list):
            return []

        authors = []
        for author in value:
            if isinstance(author, dict):
                name = author.get("name")
            else:
                name = str(author)

            if name:
                authors.append(str(name).strip())

        return [name for name in authors if name]

    @staticmethod
    def _extract_year(text: str) -> Optional[int]:
        match = re.search(r"\b(19\d{2}|20\d{2}|21\d{2})\b", text)
        if not match:
            return None

        try:
            return int(match.group(1))
        except ValueError:
            return None

    @staticmethod
    def _extract_venue(summary: str, year: Optional[int]) -> Optional[str]:
        if not summary:
            return None

        parts = [part.strip() for part in summary.split("-")]
        if len(parts) < 2:
            return None

        # Scholar commonly formats this as:
        # "Authors - Journal, 2024 - publisher"
        candidate = parts[-2] if year and str(year) in parts[-2] else parts[-1]
        candidate = re.sub(r"\b(19\d{2}|20\d{2}|21\d{2})\b", "", candidate)
        candidate = candidate.strip(" ,")

        return candidate or None

    @classmethod
    def _extract_doi(cls, item: dict) -> Optional[str]:
        candidates = []

        link = item.get("link")
        if link:
            candidates.append(str(link))

        snippet = item.get("snippet")
        if snippet:
            candidates.append(str(snippet))

        for resource in item.get("resources") or []:
            if isinstance(resource, dict) and resource.get("link"):
                candidates.append(str(resource["link"]))

        doi_pattern = re.compile(
            r"(?:https?://(?:dx\.)?doi\.org/|doi:\s*)"
            r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
            re.IGNORECASE,
        )

        for text in candidates:
            match = doi_pattern.search(text)
            if match:
                return match.group(1).rstrip(".,;)")

        return None

    @staticmethod
    def _safe_int(value) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
