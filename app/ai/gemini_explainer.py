from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

import httpx


class GeminiDatasetExplainer:
    """
    Gemini hanya digunakan untuk menjelaskan hasil analisis
    yang SUDAH dihitung oleh analyzer lokal.

    Gemini tidak menghitung ulang statistik dataset.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

        self.model = model or os.getenv(
            "GEMINI_MODEL",
            "gemini-3.8-flash",
        )

        self.timeout = float(
            os.getenv("GEMINI_TIMEOUT", "60")
        )

        self.max_retries = 3

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def explain(self, analysis: Dict[str, Any]) -> str:
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY belum diatur."
            )

        compact = self._compact_analysis(analysis)

        prompt = """
Kamu adalah penjelas hasil analisis dataset
untuk aplikasi Dataset Research.

Tugasmu HANYA menjelaskan hasil analisis
yang sudah dihitung oleh program.

Jangan melakukan analisis statistik baru.
Jangan menghitung ulang angka.
Jangan mengubah angka.
Jangan mengarang fakta.
Jangan memberikan rekomendasi machine learning.

Gunakan HANYA informasi yang tersedia
di DATA ANALYSIS.

Gunakan Bahasa Indonesia yang natural,
jelas, dan mudah dipahami mahasiswa.

Fokus pada:

1. Gambaran umum dataset.
2. Kondisi kualitas data.
3. Nilai kosong dan duplikat.
4. Outlier jika ada.
5. Kolom yang paling menonjol.
6. Korelasi jika tersedia.
7. Kesimpulan singkat tentang kondisi dataset.

Tulis sekitar 4-6 paragraf pendek.
Setiap paragraf fokus pada satu aspek.
Jangan terlalu panjang.
Jangan membuat tabel.

Pastikan seluruh aspek di atas dibahas
jika informasi tersebut memang tersedia
di DATA ANALYSIS.

Jangan menyebut:
- API
- JSON
- Gemini
- model AI
- prompt
- instruksi internal

Jangan membuat angka baru.
Jangan mengubah atau memperkirakan angka.
Gunakan hanya fakta yang diberikan.

DATA ANALYSIS:
""" + json.dumps(
            compact,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

        url = (
            "https://generativelanguage.googleapis.com/"
            f"v1beta/models/{self.model}:generateContent"
        )

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": prompt
                        }
                    ],
                }
            ],
            "generationConfig": {
                "thinkingConfig": {
                    "thinkingLevel": "low"
                },
                "maxOutputTokens": 1200,
            },
        }

        last_error: Exception | None = None

        for attempt in range(
            1,
            self.max_retries + 1,
        ):
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

                # Error yang biasanya bersifat sementara.
                if response.status_code in (
                    429,
                    502,
                    503,
                    504,
                ):
                    last_error = RuntimeError(
                        "Gemini sementara tidak tersedia "
                        f"(HTTP {response.status_code})."
                    )

                    if attempt < self.max_retries:
                        # 2 detik -> 4 detik -> 8 detik
                        time.sleep(2 ** attempt)
                        continue

                    raise last_error

                response.raise_for_status()

                data = response.json()

                text = self._extract_text(data)

                if not text:
                    raise RuntimeError(
                        "Gemini tidak mengembalikan "
                        "teks penjelasan."
                    )

                return text.strip()

            except httpx.RequestError as exc:
                last_error = RuntimeError(
                    f"Gagal terhubung ke Gemini: {exc}"
                )

                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                    continue

                raise last_error from exc

            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code

                raise RuntimeError(
                    f"Gemini mengembalikan HTTP {status}."
                ) from exc

        if last_error is not None:
            raise last_error

        raise RuntimeError(
            "Gagal mendapatkan penjelasan dari Gemini."
        )

    @staticmethod
    def _extract_text(
        payload: Dict[str, Any],
    ) -> str:
        candidates = payload.get("candidates") or []

        for candidate in candidates:
            content = candidate.get("content") or {}
            parts = content.get("parts") or []

            texts = []

            for part in parts:
                if not isinstance(part, dict):
                    continue

                text = part.get("text")

                if text:
                    texts.append(str(text))

            if texts:
                return "\n".join(texts)

        return ""

    @staticmethod
    def _compact_analysis(
        analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Hanya mengirim hasil analisis yang sudah dihitung
        oleh analyzer lokal.
        """

        if not isinstance(analysis, dict):
            return {}

        return {
            "profile": analysis.get(
                "profile",
                {},
            ),
            "statistics": analysis.get(
                "statistics",
                [],
            ),
            "missing_values": analysis.get(
                "missing_values",
                [],
            ),
            "duplicates": analysis.get(
                "duplicates",
                {},
            ),
            "outliers": analysis.get(
                "outliers",
                [],
            ),
            "correlations": analysis.get(
                "correlations",
            ),
        }