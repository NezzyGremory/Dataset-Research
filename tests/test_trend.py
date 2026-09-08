from __future__ import annotations

from app.research.trend_analyzer import ResearchTrendAnalyzer


def test_research_trend():
    papers = [
        {
            "title": "Machine Learning Study 2021",
            "year": 2021,
            "abstract": "Machine learning research paper.",
        },
        {
            "title": "Machine Learning Study 2022 A",
            "year": 2022,
            "abstract": "Machine learning research paper.",
        },
        {
            "title": "Machine Learning Study 2022 B",
            "year": 2022,
            "abstract": "Machine learning research paper.",
        },
        {
            "title": "Machine Learning Study 2023",
            "year": 2023,
            "abstract": "Machine learning research paper.",
        },
        {
            "title": "Machine Learning Study 2024 A",
            "year": 2024,
            "abstract": "Machine learning research paper.",
        },
        {
            "title": "Machine Learning Study 2024 B",
            "year": 2024,
            "abstract": "Machine learning research paper.",
        },
    ]

    analyzer = ResearchTrendAnalyzer()

    result = analyzer.analyze(papers)

    assert isinstance(result, dict), (
        "Result harus berupa dictionary."
    )

    assert result.get("status") == "SUCCESS", (
        "Research Trend gagal.\n"
        f"Result:\n{result}"
    )

    # ==========================================================
    # PUBLICATION TREND
    # ==========================================================

    assert "publication_trend" in result, (
        "Result tidak memiliki publication_trend."
    )

    publication_trend = result["publication_trend"]

    assert isinstance(publication_trend, list), (
        "publication_trend harus berupa list."
    )

    assert len(publication_trend) == 4, (
        "Seharusnya terdapat 4 tahun publikasi."
    )

    expected_trend = [
        {
            "year": 2021,
            "paper_count": 1,
        },
        {
            "year": 2022,
            "paper_count": 2,
        },
        {
            "year": 2023,
            "paper_count": 1,
        },
        {
            "year": 2024,
            "paper_count": 2,
        },
    ]

    assert publication_trend == expected_trend, (
        "Publication trend tidak sesuai.\n"
        f"Expected:\n{expected_trend}\n"
        f"Actual:\n{publication_trend}"
    )

    # ==========================================================
    # GROWTH
    # ==========================================================

    assert "growth" in result, (
        "Result tidak memiliki growth."
    )

    growth = result["growth"]

    assert isinstance(growth, dict), (
        "growth harus berupa dictionary."
    )

    assert growth["first_year"] == 2021
    assert growth["first_count"] == 1

    assert growth["last_year"] == 2024
    assert growth["last_count"] == 2

    assert growth["percentage"] == 100.0, (
        "Growth dari 1 paper menjadi 2 paper "
        "seharusnya 100%."
    )

    # ==========================================================
    # PEAK YEAR
    # ==========================================================

    assert "peak_year" in result, (
        "Result tidak memiliki peak_year."
    )

    assert result["peak_year"] == 2022, (
        "Peak year seharusnya 2022."
    )

    # ==========================================================
    # PEAK PAPER COUNT
    # ==========================================================

    assert "peak_paper_count" in result, (
        "Result tidak memiliki peak_paper_count."
    )

    assert result["peak_paper_count"] == 2, (
        "Peak paper count seharusnya 2."
    )

    # ==========================================================
    # RECENT DIRECTION
    # ==========================================================

    assert "recent_direction" in result, (
        "Result tidak memiliki recent_direction."
    )

    assert result["recent_direction"] == "INCREASING", (
        "Recent direction seharusnya INCREASING.\n"
        f"Actual: {result['recent_direction']}"
    )

    # ==========================================================
    # YEAR RANGE
    # ==========================================================

    assert "year_range" in result, (
        "Result tidak memiliki year_range."
    )

    year_range = result["year_range"]

    assert isinstance(year_range, dict), (
        "year_range harus berupa dictionary."
    )

    assert year_range["start"] == 2021, (
        "Tahun awal seharusnya 2021."
    )

    assert year_range["end"] == 2024, (
        "Tahun akhir seharusnya 2024."
    )

    # ==========================================================
    # SUMMARY
    # ==========================================================

    assert "summary" in result, (
        "Result tidak memiliki summary."
    )

    summary = result["summary"]

    assert isinstance(summary, dict), (
        "summary harus berupa dictionary."
    )

    assert summary["total_years"] == 4, (
        "Total tahun seharusnya 4."
    )

    assert summary["total_papers"] == 6, (
        "Total paper seharusnya 6."
    )

    assert summary["peak_year"] == 2022, (
        "Summary peak year seharusnya 2022."
    )

    assert summary["recent_direction"] == "INCREASING", (
        "Summary recent direction seharusnya INCREASING."
    )

    # ==========================================================
    # OUTPUT
    # ==========================================================

    print("=== RESEARCH TREND ===")
    print()

    print("Publication trend:")

    for item in publication_trend:
        print(
            f"- {item['year']}: "
            f"{item['paper_count']} paper"
        )

    print()

    print("Growth:")
    print(growth)

    print()

    print("Peak year:", result["peak_year"])

    print(
        "Peak paper count:",
        result["peak_paper_count"],
    )

    print()

    print(
        "Recent direction:",
        result["recent_direction"],
    )

    print()

    print("Year range:")
    print(year_range)

    print()

    print("Summary:")
    print(summary)

    print()

    print("Research Trend: PASSED")


if __name__ == "__main__":
    test_research_trend()

    print()
    print("================================")
    print("Semua Research Trend test berhasil.")
    print("================================")