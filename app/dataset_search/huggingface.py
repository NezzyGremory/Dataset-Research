from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote
import re

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

    @property
    def title(self) -> str:
        name = self.dataset_id.split("/", 1)[-1]
        return name.replace("-", " ").replace("_", " ").strip()

    @property
    def url(self) -> str:
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
    """Small public-client wrapper around the Hugging Face Hub dataset APIs."""

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

    def __init__(self, timeout: float = 30.0, max_results: int = 12):
        self.timeout = timeout
        self.max_results = max_results
        self.headers = {
            "User-Agent": "DatasetResearch/2.0 Dataset Search"
        }

    def search(self, query: str, limit: Optional[int] = None) -> list[DatasetSearchResult]:
        query = (query or "").strip()
        if not query:
            return []

        limit = max(1, min(limit or self.max_results, 30))

        params = {
            "search": query,
            "limit": limit,
            "sort": "downloads",
            "direction": -1,
        }

        with httpx.Client(
            timeout=self.timeout,
            headers=self.headers,
            follow_redirects=True,
        ) as client:
            response = client.get(self.API_URL, params=params)
            response.raise_for_status()
            items = response.json()

            if not isinstance(items, list):
                return []

            results: list[DatasetSearchResult] = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                dataset_id = item.get("id")
                if not dataset_id:
                    continue

                # Keep search fast. Detailed file information is requested
                # only for each result so format chips stay useful.
                details = self._get_details(client, dataset_id)
                result = self._parse_result(item, details)
                results.append(result)

            return results

    def _get_details(self, client: httpx.Client, dataset_id: str) -> dict[str, Any]:
        try:
            response = client.get(
                f"{self.API_URL}/{quote(dataset_id, safe='/')}",
                params={"full": "true"},
            )
            if response.status_code >= 400:
                return {}
            data = response.json()
            return data if isinstance(data, dict) else {}
        except (httpx.HTTPError, ValueError):
            return {}

    def _parse_result(self, item: dict[str, Any], details: dict[str, Any]) -> DatasetSearchResult:
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
            files.append(DatasetFile(filename=filename, size_bytes=size_bytes))

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

        return DatasetSearchResult(
            dataset_id=str(item.get("id") or source.get("id") or ""),
            author=str(author),
            description=self._clean_description(str(description)),
            downloads=self._safe_int(source.get("downloads", item.get("downloads", 0))),
            likes=self._safe_int(source.get("likes", item.get("likes", 0))),
            last_modified=str(source.get("lastModified", item.get("lastModified", "") or "")),
            tags=tags,
            files=files,
        )

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
    def _clean_description(value: str) -> str:
        value = re.sub(r"\s+", " ", value).strip()
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
                f"Format file tidak didukung untuk Dataset Search: {suffix or 'unknown'}"
            )

        destination = Path(destination_dir)
        destination.mkdir(parents=True, exist_ok=True)

        safe_repo = re.sub(r"[^A-Za-z0-9._-]+", "_", dataset_id.replace("/", "__"))
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(filename).name)
        target = destination / f"{safe_repo}__{safe_name}"

        encoded_path = "/".join(quote(part, safe="") for part in filename.split("/"))
        url = f"{self.DOWNLOAD_ROOT}/{quote(dataset_id, safe='/')}/resolve/main/{encoded_path}?download=true"

        with httpx.stream(
            "GET",
            url,
            timeout=self.timeout,
            headers=self.headers,
            follow_redirects=True,
        ) as response:
            response.raise_for_status()
            with target.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)

        return target
