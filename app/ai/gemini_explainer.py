from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Iterable

import httpx


class GeminiDatasetExplainer:
    """
    Explain already-computed Dataset Research analysis in Indonesian.

    Gemini is used only to turn local analyzer results into human-readable
    wording. It does not calculate statistics, modify numbers, or choose
    machine-learning methods.
    """

    DEFAULT_MODEL = "gemini-3.5-flash-lite"
    FALLBACK_MODELS = (
        "gemini-3.5-flash-lite",
        "gemini-2.5-flash-lite",
    )

    RETRYABLE_STATUS_CODES = {429, 502, 503, 504}

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv(
            "GEMINI_MODEL",
            self.DEFAULT_MODEL,
        )

        self.timeout = float(
            os.getenv("GEMINI_TIMEOUT", "45")
        )

        # Keep the explanation responsive. A model is retried once before
        # moving to the next available stable Flash fallback.
        self.retries_per_model = 2

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def explain(self, analysis: Dict[str, Any]) -> str:
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY belum diatur."
            )

        compact = self._compact_analysis(analysis)
        prompt = self._build_prompt(compact)

        models = self._models_to_try()
        last_error: str | None = None

        for model_name in models:
            url = (
                "https://generativelanguage.googleapis.com/"
                f"v1beta/models/{model_name}:generateContent"
            )

            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": prompt}
                        ],
                    }
                ],
                "generationConfig": {
                    "thinkingConfig": {
                        "thinkingLevel": "low"
                    },
                    "maxOutputTokens": 1000,
                },
            }

            for attempt in range(1, self.retries_per_model + 1):
                try:
                    response = httpx.post(
                        url,
                        headers={
                            "x-goog-api-key": self.api_key,
                            "Content-Type": "application/json",
                        },
                        json=payload,
                        timeout=self.timeout,
                    )

                    if response.status_code in self.RETRYABLE_STATUS_CODES:
                        last_error = (
                            f"{model_name}: HTTP {response.status_code}"
                        )

                        # A busy model should not block the whole explanation
                        # flow for too long. Retry once, then try fallback.
                        if attempt < self.retries_per_model:
                            time.sleep(1.5 * attempt)
                            continue

                        break

                    response.raise_for_status()

                    data = response.json()
                    text = self._extract_text(data)

                    if text:
                        return text.strip()

                    last_error = (
                        f"{model_name}: respons berhasil tetapi teks kosong"
                    )
                    break

                except httpx.RequestError as exc:
                    last_error = (
                        f"{model_name}: koneksi gagal ({exc})"
                    )

                    if attempt < self.retries_per_model:
                        time.sleep(1.5 * attempt)
                        continue

                    break

                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    last_error = (
                        f"{model_name}: HTTP {status}"
                    )
                    break

        # Do not expose raw server payloads to the UI.
        raise RuntimeError(
            "Gemini sedang tidak dapat membuat penjelasan. "
            "Analisis dataset tetap tersedia seperti biasa. "
            "Silakan coba Run Analysis lagi."
            + (f" ({last_error})" if last_error else "")
        )

    def _models_to_try(self) -> list[str]:
        """Return preferred model followed by stable Flash fallbacks."""
        models: list[str] = []

        for name in (
            self.model,
            *self.FALLBACK_MODELS,
        ):
            normalized = str(name).strip()
            if normalized and normalized not in models:
                models.append(normalized)

        return models

    @staticmethod
    def _build_prompt(
        compact: Dict[str, Any],
    ) -> str:
        analysis_json = json.dumps(
            compact,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )

        return f"""
Kamu adalah penjelas hasil analisis dataset untuk aplikasi Dataset Research.

Tugasmu HANYA menjelaskan fakta yang SUDAH dihitung oleh program lokal.

ATURAN MUTLAK:
- Jangan menghitung ulang statistik.
- Jangan membuat angka baru.
- Jangan mengubah angka yang diberikan.
- Jangan menebak fakta yang tidak tersedia.
- Jangan melakukan analisis statistik tambahan.
- Jangan memberikan rekomendasi machine learning.
- Gunakan hanya DATA ANALYSIS di bawah.
- Jika suatu bagian tidak tersedia, jangan mengarangnya.

Gunakan Bahasa Indonesia yang natural dan mudah dipahami mahasiswa.

Jelaskan secara ringkas tetapi lengkap:
1. Gambaran umum dataset.
2. Kualitas data dan nilai kosong.
3. Duplikat.
4. Outlier jika tersedia.
5. Kolom yang menonjol jika tersedia.
6. Korelasi jika tersedia.
7. Kesimpulan kondisi dataset.

Tulis sekitar 4-6 paragraf pendek.
Jangan membuat tabel.
Jangan menyebut API, JSON, Gemini, model AI, prompt, atau instruksi internal.

DATA ANALYSIS:
{analysis_json}
""".strip()

    @staticmethod
    def _extract_text(
        payload: Dict[str, Any],
    ) -> str:
        candidates = payload.get("candidates") or []

        for candidate in candidates:
            content = candidate.get("content") or {}
            parts = content.get("parts") or []

            texts: list[str] = []

            for part in parts:
                if not isinstance(part, dict):
                    continue

                text = part.get("text")
                if text:
                    texts.append(str(text))

            if texts:
                return "\n".join(texts)

        return ""

    @classmethod
    def _compact_analysis(
        cls,
        analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Keep the request small without calculating new statistics.

        We only select already-computed values and cap long collections so a
        large analyzer result cannot make the explanation request unnecessarily
        heavy.
        """
        if not isinstance(analysis, dict):
            return {}

        return {
            "profile": cls._limit_mapping(
                analysis.get("profile", {}),
                max_items=40,
            ),
            "statistics": cls._limit_collection(
                analysis.get("statistics", []),
                max_items=20,
            ),
            "missing_values": cls._limit_collection(
                analysis.get("missing_values", []),
                max_items=20,
            ),
            "duplicates": cls._limit_mapping(
                analysis.get("duplicates", {}),
                max_items=20,
            ),
            "outliers": cls._limit_collection(
                analysis.get("outliers", []),
                max_items=20,
            ),
            "correlations": cls._limit_correlations(
                analysis.get("correlations")
            ),
        }

    @staticmethod
    def _limit_collection(
        value: Any,
        max_items: int,
    ) -> Any:
        if isinstance(value, list):
            return value[:max_items]

        if isinstance(value, tuple):
            return list(value[:max_items])

        return value

    @staticmethod
    def _limit_mapping(
        value: Any,
        max_items: int,
    ) -> Any:
        if not isinstance(value, dict):
            return value

        items = list(value.items())[:max_items]
        return dict(items)

    @classmethod
    def _limit_correlations(
        cls,
        value: Any,
    ) -> Any:
        if isinstance(value, dict):
            return cls._limit_mapping(value, 30)

        if isinstance(value, list):
            return cls._limit_collection(value, 20)

        return value
