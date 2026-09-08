from __future__ import annotations

from collections import Counter
from typing import Dict, List


class ResearchTrendAnalyzer:
    """
    Menganalisis perkembangan publikasi berdasarkan tahun.

    Semua hasil bersifat HEURISTIC.
    """

    def analyze(
        self,
        papers: List[Dict],
        landscape: Dict | None = None,
    ) -> Dict:

        if not papers:
            return {
                "status": "NO_DATA",
                "publication_trend": [],
                "growth": None,
                "peak_year": None,
                "peak_paper_count": 0,
                "recent_direction": "UNKNOWN",
                "year_range": None,
                "summary": {},
                "status_type": "HEURISTIC",
            }

        yearly = Counter()

        for paper in papers:
            year = paper.get("year")

            if isinstance(year, int):
                yearly[year] += 1

        if not yearly:
            return {
                "status": "NO_YEAR_DATA",
                "publication_trend": [],
                "growth": None,
                "peak_year": None,
                "peak_paper_count": 0,
                "recent_direction": "UNKNOWN",
                "year_range": None,
                "summary": {},
                "status_type": "HEURISTIC",
            }

        years = sorted(yearly.keys())

        trend = [
            {
                "year": year,
                "paper_count": yearly[year],
            }
            for year in years
        ]

        peak_year = max(
            yearly,
            key=yearly.get,
        )

        growth = self._calculate_growth(
            yearly,
            years,
        )

        recent_direction = self._recent_direction(
            yearly,
            years,
        )

        return {
            "status": "SUCCESS",
            "publication_trend": trend,
            "growth": growth,
            "peak_year": peak_year,
            "peak_paper_count": yearly[peak_year],
            "recent_direction": recent_direction,
            "year_range": {
                "start": years[0],
                "end": years[-1],
            },
            "summary": self._build_summary(
                yearly=yearly,
                years=years,
                peak_year=peak_year,
                direction=recent_direction,
            ),
            "status_type": "HEURISTIC",
        }

    @staticmethod
    def _calculate_growth(
        yearly: Counter,
        years: List[int],
    ) -> Dict | None:

        if len(years) < 2:
            return None

        first_year = years[0]
        last_year = years[-1]

        first_count = yearly[first_year]
        last_count = yearly[last_year]

        if first_count == 0:
            percentage = None
        else:
            percentage = round(
                (
                    (last_count - first_count)
                    / first_count
                ) * 100,
                2,
            )

        return {
            "first_year": first_year,
            "first_count": first_count,
            "last_year": last_year,
            "last_count": last_count,
            "percentage": percentage,
        }

    @staticmethod
    def _recent_direction(
        yearly: Counter,
        years: List[int],
    ) -> str:

        # Tidak cukup data untuk membaca tren.
        if len(years) < 2:
            return "INSUFFICIENT_DATA"

        # Gunakan dua tahun terakhir yang tersedia.
        previous_year = years[-2]
        latest_year = years[-1]

        previous_count = yearly[
            previous_year
        ]

        latest_count = yearly[
            latest_year
        ]

        if latest_count > previous_count:
            return "INCREASING"

        if latest_count < previous_count:
            return "DECREASING"

        # Kalau sama, lihat keseluruhan rentang.
        first_count = yearly[
            years[0]
        ]

        last_count = yearly[
            years[-1]
        ]

        if last_count > first_count:
            return "INCREASING"

        if last_count < first_count:
            return "DECREASING"

        return "STABLE"

    @staticmethod
    def _build_summary(
        yearly: Counter,
        years: List[int],
        peak_year: int,
        direction: str,
    ) -> Dict:

        return {
            "total_years": len(years),
            "total_papers": sum(
                yearly.values()
            ),
            "peak_year": peak_year,
            "recent_direction": direction,
        }