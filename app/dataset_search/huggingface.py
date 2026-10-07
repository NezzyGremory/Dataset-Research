from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote
import re
import time
from concurrent.futures import ThreadPoolExecutor

import httpx


@dataclass
class DatasetFile:
    filename: str
    size_bytes: Optional[int] = None

    @property
    def suffix(self) -> str:
        return Path(self.filename).suffix.lower().lstrip(".")

    @property
    def display_size(self) -> str:
        if self.size_bytes is None:
            return "Ukuran tidak tersedia"
        size = float(self.size_bytes)
        units = ["B", "KB", "MB", "GB", "TB"]
        index = 0
        while size >= 1024 and index < len(units) - 1:
            size /= 1024
            index += 1
        return f"{size:.1f} {units[index]}"


@dataclass
class DatasetSearchResult:
    dataset_id: str
    author: str = ""
    description: str = ""
    downloads: int = 0
    likes: int = 0
    last_modified: str = ""
    tags: list[str] = field(default_factory=list)
    files: list[DatasetFile] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    match_reason: str = ""
    relevance_score: float = 0.0

    @property
    def title(self) -> str:
        clean_id = self.dataset_id.replace("kaggle:", "")
        name = clean_id.split("/", 1)[-1]
        return name.replace("-", " ").replace("_", " ").strip()

    @property
    def source_label(self) -> str:
        if self.dataset_id.startswith("kaggle:"):
            return "Kaggle"
        return "Hugging Face"

    @property
    def url(self) -> str:
        if self.dataset_id.startswith("kaggle:"):
            clean_id = self.dataset_id.replace("kaggle:", "")
            return f"https://www.kaggle.com/datasets/{clean_id}"
        return f"https://huggingface.co/datasets/{self.dataset_id}"

    @property
    def formats(self) -> list[str]:
        preferred = {
            "csv",
            "tsv",
            "json",
            "jsonl",
            "parquet",
            "xlsx",
            "xls",
        }
        seen: list[str] = []
        for file in self.files:
            suffix = file.suffix
            if suffix in preferred and suffix not in seen:
                seen.append(suffix)
        return seen


class HuggingFaceDatasetClient:
    """Dataset discovery client with keyword expansion and local relevance scoring."""

    BASE_URL = "https://huggingface.co"
    API_URL = f"{BASE_URL}/api/datasets"
    DOWNLOAD_ROOT = f"{BASE_URL}/datasets"

    SUPPORTED_TABULAR = {
        ".csv",
        ".tsv",
        ".json",
        ".jsonl",
        ".parquet",
        ".xlsx",
        ".xls",
    }

    # Lightweight bilingual/domain vocabulary so users can search naturally
    # in Indonesian without knowing the exact English dataset title.
    KEYWORD_ALIASES: dict[str, tuple[str, ...]] = {
        "pertanian": ("agriculture", "agricultural", "farming", "crop", "crops", "plant", "rice"),
        "petani": ("farmer", "farmers", "agriculture", "farming", "crop"),
        "tanaman": ("plant", "plants", "crop", "crops", "agriculture"),
        "padi": ("rice", "paddy", "agriculture", "crop"),
        "jagung": ("corn", "maize", "agriculture", "crop"),
        "perkebunan": ("plantation", "agriculture", "farming"),
        "mahasiswa": ("student", "students", "university", "college", "higher education", "academic"),
        "siswa": ("student", "students", "school", "education", "academic"),
        "pelajar": ("student", "students", "education", "school"),
        "sekolah": ("school", "education", "student", "students"),
        "pendidikan": ("education", "educational", "student", "students", "school", "university"),
        "universitas": ("university", "college", "higher education", "student"),
        "kesehatan": ("health", "healthcare", "medical", "medicine", "clinical"),
        "penyakit": ("disease", "diseases", "medical", "healthcare", "clinical"),
        "rumah sakit": ("hospital", "healthcare", "medical", "clinical"),
        "diabetes": ("diabetes", "medical", "healthcare", "clinical"),
        "kanker": ("cancer", "oncology", "medical", "healthcare"),
        "jantung": ("heart", "cardiac", "cardiovascular", "medical"),
        "keuangan": ("finance", "financial", "banking", "stock", "credit"),
        "finansial": ("finance", "financial", "banking", "credit"),
        "bank": ("banking", "finance", "financial", "credit"),
        "saham": ("stock", "stocks", "finance", "financial", "market"),
        "ekonomi": ("economics", "economic", "finance", "financial"),
        "bisnis": ("business", "commerce", "company", "sales", "marketing"),
        "penjualan": ("sales", "sales data", "commerce", "retail"),
        "pelanggan": ("customer", "customers", "churn", "retail", "marketing"),
        "pemasaran": ("marketing", "customer", "sales", "commerce"),
        "transportasi": ("transportation", "traffic", "vehicle", "mobility", "road"),
        "lalu lintas": ("traffic", "transportation", "vehicle", "road"),
        "kendaraan": ("vehicle", "cars", "automotive", "transportation"),
        "lingkungan": ("environment", "environmental", "climate", "pollution", "ecology"),
        "iklim": ("climate", "weather", "environment", "temperature"),
        "cuaca": ("weather", "climate", "temperature", "precipitation"),
        "polusi": ("pollution", "air quality", "environment", "emissions"),
        "teknologi": ("technology", "software", "computer", "computing", "ai"),
        "komputer": ("computer", "computing", "software", "technology"),
        "kecerdasan buatan": ("artificial intelligence", "ai", "machine learning", "deep learning"),
        "machine learning": ("machine learning", "ml", "classification", "regression"),
        "gambar": ("image", "images", "vision", "computer vision", "image classification"),
        "citra": ("image", "images", "vision", "computer vision"),
        "teks": ("text", "nlp", "natural language", "language"),
        "bahasa": ("language", "nlp", "text", "linguistics"),
        "musik": ("music", "audio", "sound", "speech"),
        "suara": ("audio", "speech", "sound", "voice"),
        "olahraga": ("sports", "sport", "football", "soccer", "basketball"),
        "permainan": ("game", "gaming", "video games", "gameplay"),
        "energi": ("energy", "electricity", "power", "renewable energy"),
        "listrik": ("electricity", "power", "energy", "smart grid"),
        "perumahan": ("housing", "real estate", "property", "house prices"),
    }

    STOPWORDS = {
        "dataset", "data", "tentang", "untuk", "dengan", "dan", "atau",
        "yang", "dari", "pada", "di", "ke", "the", "of", "for", "and",
        "with", "about", "a", "an", "to", "in", "on",
    }

    def __init__(self, timeout: float = 12.0, max_results: int = 12):
        self.timeout = timeout
        self.max_results = max_results
        self.headers = {
            "User-Agent": "DatasetResearch/2.1 Dataset Search"
        }

        # Short in-process cache: repeated searches become instant during
        # the same application session without changing the UI.
        self.cache_ttl = 300.0
        self._search_cache = {}
        self._details_cache = {}

    def search(
        self,
        query: str,
        limit: Optional[int] = None,
    ) -> list[DatasetSearchResult]:
        original_query = self._normalize_text(query)
        if not original_query:
            return []

        limit = max(1, min(limit or self.max_results, 20))

        cached = self._search_cache.get(original_query)
        if cached:
            cached_at, cached_results = cached
            if time.monotonic() - cached_at < self.cache_ttl:
                return list(cached_results[:limit])

        search_terms = self._expand_query(original_query)[:8]

        candidates: dict[str, dict[str, Any]] = {}
        per_term_limit = min(max(limit, 8), 12)

        # Keep the original, reliable HF request shape. We only parallelize a
        # small number of search calls and do not use the more aggressive
        # expand/fan-out optimization that could trigger provider throttling.
        def fetch_term(term: str):
            with httpx.Client(
                timeout=self.timeout,
                headers=self.headers,
                follow_redirects=True,
            ) as client:
                try:
                    response = client.get(
                        self.API_URL,
                        params={
                            "search": term,
                            "limit": per_term_limit,
                            "sort": "downloads",
                            "direction": -1,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()
                    return term, data if isinstance(data, list) else []
                except (httpx.HTTPError, ValueError):
                    return term, []

        # A small worker pool reduces wall-clock time but avoids a burst of
        # requests that can hit the Hub's rate limits.
        worker_count = min(3, max(1, len(search_terms)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(fetch_term, term) for term in search_terms]
            for future in futures:
                try:
                    term, items = future.result()
                except Exception:
                    continue

                for item in items:
                    if not isinstance(item, dict):
                        continue
                    dataset_id = str(item.get("id") or "").strip()
                    if not dataset_id:
                        continue
                    existing = candidates.get(dataset_id)
                    if existing is None:
                        candidates[dataset_id] = {
                            "item": item,
                            "terms": {term},
                        }
                    else:
                        existing["terms"].add(term)

        if not candidates:
            self._search_cache[original_query] = (time.monotonic(), [])
            return []

        # Rank using whatever metadata the search endpoint already returned.
        ranked_candidates = sorted(
            candidates.values(),
            key=lambda candidate: self._quick_candidate_score(
                original_query,
                candidate["item"],
            ),
            reverse=True,
        )

        # Only enrich the top few results; the previous version fetched full
        # details for every candidate, which was the biggest latency source.
        top_candidates = ranked_candidates[:min(limit, 6)]

        def enrich(candidate):
            item = candidate["item"]
            dataset_id = str(item.get("id") or "")
            cached_detail = self._details_cache.get(dataset_id)
            if cached_detail:
                cached_at, detail = cached_detail
                if time.monotonic() - cached_at < self.cache_ttl:
                    return candidate, detail

            try:
                with httpx.Client(
                    timeout=self.timeout,
                    headers=self.headers,
                    follow_redirects=True,
                ) as client:
                    detail = self._get_details(client, dataset_id)
            except Exception:
                detail = {}
            return candidate, detail

        enriched = []
        with ThreadPoolExecutor(max_workers=min(3, len(top_candidates) or 1)) as executor:
            futures = [executor.submit(enrich, candidate) for candidate in top_candidates]
            for future in futures:
                try:
                    enriched.append(future.result())
                except Exception:
                    continue

        results: list[DatasetSearchResult] = []
        for candidate, details in enriched:
            result = self._parse_result(
                candidate["item"],
                details,
                original_query,
            )
            if result is not None:
                results.append(result)

        results.sort(
            key=lambda result: (
                result.relevance_score,
                result.downloads,
                result.likes,
            ),
            reverse=True,
        )

        results = results[:limit]
        self._search_cache[original_query] = (
            time.monotonic(),
            list(results),
        )
        return results

    def _quick_candidate_score(
        self,
        original_query: str,
        item: dict[str, Any],
    ) -> float:
        dataset_id = self._normalize_text(str(item.get("id") or ""))
        description = self._normalize_text(
            str(item.get("description") or "")
        )
        tags = item.get("tags") or []
        tag_text = " ".join(
            self._normalize_text(str(tag)) for tag in tags
        )

        query_tokens = [
            token
            for token in re.findall(r"[a-zA-Z0-9À-ÿ]+", original_query)
            if token not in self.STOPWORDS and len(token) > 1
        ]

        terms = list(query_tokens)
        for token in query_tokens:
            terms.extend(self.KEYWORD_ALIASES.get(token, ()))
        terms.extend(self.KEYWORD_ALIASES.get(original_query, ()))

        score = 0.0
        for term in terms:
            normalized = self._normalize_text(term)
            if not normalized:
                continue
            if normalized in dataset_id:
                score += 12.0
            if normalized in tag_text:
                score += 8.0
            if normalized in description:
                score += 5.0

        try:
            downloads = int(item.get("downloads", 0) or 0)
        except (TypeError, ValueError):
            downloads = 0
        return score + downloads / 1_000_000.0

    def _expand_query(self, query: str) -> list[str]:
        query = self._normalize_text(query)

        terms: list[str] = [query]

        # Try the whole phrase in the alias table first.
        if query in self.KEYWORD_ALIASES:
            terms.extend(self.KEYWORD_ALIASES[query])

        tokens = [
            token
            for token in re.findall(r"[a-zA-Z0-9À-ÿ_-]+", query)
            if token not in self.STOPWORDS and len(token) > 1
        ]

        # Search useful individual words as well as the original phrase.
        for token in tokens:
            terms.append(token)
            terms.extend(self.KEYWORD_ALIASES.get(token, ()))

        # Handle phrases such as "dataset pertanian", where "dataset" is noise.
        stripped = " ".join(tokens).strip()
        if stripped and stripped != query:
            terms.append(stripped)

        # Handle phrase aliases embedded in a larger query.
        for key, aliases in self.KEYWORD_ALIASES.items():
            if key in query and key != query:
                terms.extend(aliases[:4])

        clean: list[str] = []
        seen: set[str] = set()
        for term in terms:
            term = self._normalize_text(term)
            if not term or term in seen:
                continue
            seen.add(term)
            clean.append(term)

        return clean[:14]

    def _get_details(
        self,
        client: httpx.Client,
        dataset_id: str,
    ) -> dict[str, Any]:
        cached = self._details_cache.get(dataset_id)
        if cached:
            cached_at, data = cached
            if time.monotonic() - cached_at < self.cache_ttl:
                return dict(data)

        try:
            response = client.get(
                f"{self.API_URL}/{quote(dataset_id, safe='/')}",
                params={"full": "true"},
            )
            if response.status_code >= 400:
                return {}
            data = response.json()
            if not isinstance(data, dict):
                return {}
            self._details_cache[dataset_id] = (
                time.monotonic(),
                data,
            )
            return data
        except (httpx.HTTPError, ValueError):
            return {}

    def _parse_result(
        self,
        item: dict[str, Any],
        details: dict[str, Any],
        original_query: str,
    ) -> Optional[DatasetSearchResult]:
        source = details or item
        files: list[DatasetFile] = []

        for sibling in source.get("siblings") or []:
            if not isinstance(sibling, dict):
                continue
            filename = sibling.get("rfilename") or sibling.get("path")
            if not filename:
                continue
            suffix = Path(filename).suffix.lower()
            if suffix not in self.SUPPORTED_TABULAR:
                continue
            size_bytes = self._file_size(sibling)
            files.append(
                DatasetFile(
                    filename=filename,
                    size_bytes=size_bytes,
                )
            )

        card_data = source.get("cardData") or {}
        description = (
            source.get("description")
            or card_data.get("description")
            or ""
        )

        tags = source.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        tags = [str(tag) for tag in tags if tag]

        author = source.get("author") or item.get("author") or ""
        if not author and "/" in str(item.get("id", "")):
            author = str(item["id"]).split("/", 1)[0]

        title = self._clean_text(
            str(item.get("id") or source.get("id") or "")
            .split("/", 1)[-1]
            .replace("-", " ")
            .replace("_", " ")
        )
        clean_description = self._clean_description(str(description))
        score, matched_terms, reason = self._score_relevance(
            original_query,
            title,
            clean_description,
            tags,
            files,
        )

        # Reject obviously unrelated results when the only match is a very
        # weak token. We still allow direct title/token matches.
        if score <= 0:
            return None

        return DatasetSearchResult(
            dataset_id=str(
                item.get("id") or source.get("id") or ""
            ),
            author=str(author),
            description=clean_description,
            downloads=self._safe_int(
                source.get("downloads", item.get("downloads", 0))
            ),
            likes=self._safe_int(
                source.get("likes", item.get("likes", 0))
            ),
            last_modified=str(
                source.get(
                    "lastModified",
                    item.get("lastModified", "") or "",
                )
            ),
            tags=tags,
            files=files,
            matched_terms=matched_terms,
            match_reason=reason,
            relevance_score=score,
        )

    def _score_relevance(
        self,
        original_query: str,
        title: str,
        description: str,
        tags: list[str],
        files: list[DatasetFile],
    ) -> tuple[float, list[str], str]:
        title_text = self._normalize_text(title)
        description_text = self._normalize_text(description)
        tag_text = " ".join(self._normalize_text(tag) for tag in tags)
        file_text = " ".join(
            self._normalize_text(file.filename) for file in files
        )

        query_tokens = [
            token
            for token in re.findall(r"[a-zA-Z0-9À-ÿ]+", original_query)
            if token not in self.STOPWORDS and len(token) > 1
        ]

        candidate_terms: list[str] = list(query_tokens)
        for token in query_tokens:
            candidate_terms.extend(self.KEYWORD_ALIASES.get(token, ()))

        if original_query in self.KEYWORD_ALIASES:
            candidate_terms.extend(self.KEYWORD_ALIASES[original_query])

        matched: list[str] = []
        score = 0.0
        locations: list[str] = []

        for term in candidate_terms:
            normalized = self._normalize_text(term)
            if not normalized:
                continue

            hit_title = normalized in title_text
            hit_tags = normalized in tag_text
            hit_description = normalized in description_text
            hit_file = normalized in file_text

            if hit_title:
                score += 12.0
                locations.append("nama dataset")
            if hit_tags:
                score += 8.0
                locations.append("tag")
            if hit_description:
                score += 5.0
                locations.append("deskripsi")
            if hit_file:
                score += 2.0
                locations.append("nama file")

            if (hit_title or hit_tags or hit_description or hit_file) and normalized not in matched:
                matched.append(normalized)

        # Small boost for useful tabular files because the feature is built
        # for datasets that can flow into Dataset Research directly.
        if files:
            score += 1.0

        # Prefer direct phrase matches over scattered token matches.
        if original_query and original_query in title_text:
            score += 10.0
        elif original_query and original_query in description_text:
            score += 4.0

        if not matched:
            return 0.0, [], "Tidak ada kecocokan metadata yang cukup kuat."

        unique_locations = list(dict.fromkeys(locations))
        location_text = ", ".join(unique_locations[:3])
        matched_text = ", ".join(matched[:5])

        reason = (
            f"Cocok karena kata kunci terkait ({matched_text}) "
            f"ditemukan pada {location_text}."
        )

        return score, matched, reason

    @staticmethod
    def _file_size(sibling: dict[str, Any]) -> Optional[int]:
        lfs = sibling.get("lfs")
        if isinstance(lfs, dict) and lfs.get("size") is not None:
            return HuggingFaceDatasetClient._safe_int(lfs.get("size"))
        if sibling.get("size") is not None:
            return HuggingFaceDatasetClient._safe_int(sibling.get("size"))
        return None

    @staticmethod
    def _safe_int(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _clean_text(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    @classmethod
    def _normalize_text(cls, value: str) -> str:
        value = (value or "").lower().strip()
        value = value.replace("_", " ").replace("-", " ")
        value = re.sub(r"\s+", " ", value)
        return value

    @classmethod
    def _clean_description(cls, value: str) -> str:
        value = cls._clean_text(value)
        if len(value) > 260:
            value = value[:257].rstrip() + "..."
        return value

    def download_file(
        self,
        dataset_id: str,
        filename: str,
        destination_dir: str | Path,
    ) -> Path:
        filename = filename.replace("\\", "/").lstrip("/")
        suffix = Path(filename).suffix.lower()
        if suffix not in self.SUPPORTED_TABULAR:
            raise ValueError(
                "Format file tidak didukung untuk Dataset Search: "
                f"{suffix or 'unknown'}"
            )

        destination = Path(destination_dir)
        destination.mkdir(parents=True, exist_ok=True)

        safe_repo = re.sub(
            r"[^A-Za-z0-9._-]+",
            "_",
            dataset_id.replace("/", "__"),
        )
        safe_name = re.sub(
            r"[^A-Za-z0-9._-]+",
            "_",
            Path(filename).name,
        )
        target = destination / f"{safe_repo}__{safe_name}"

        encoded_path = "/".join(
            quote(part, safe="")
            for part in filename.split("/")
        )
        url = (
            f"{self.DOWNLOAD_ROOT}/"
            f"{quote(dataset_id, safe='/')}/resolve/main/"
            f"{encoded_path}?download=true"
        )

        with httpx.stream(
            "GET",
            url,
            timeout=self.timeout,
            headers=self.headers,
            follow_redirects=True,
        ) as response:
            response.raise_for_status()
            with target.open("wb") as handle:
                for chunk in response.iter_bytes(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        handle.write(chunk)

        return target
