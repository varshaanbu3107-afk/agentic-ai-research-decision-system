from app.research.source_quality import (
    classify_source,
    get_source_quality,
    get_source_metadata,
)


def test_classify_government_source():
    assert classify_source(
        "https://example.gov/research/ai.pdf"
    ) == "government"


def test_classify_academic_source():
    assert classify_source(
        "academic_research_paper.pdf"
    ) == "academic"


def test_classify_industry_report():
    assert classify_source(
        "industry_report.pdf"
    ) == "industry_report"


def test_classify_official_company_source():
    assert classify_source(
        "official_company_report.pdf"
    ) == "official_company"


def test_classify_news_source():
    assert classify_source(
        "news_article.pdf"
    ) == "news"


def test_classify_company_blog():
    assert classify_source(
        "company_blog.pdf"
    ) == "company_blog"


def test_classify_test_document():
    assert classify_source(
        "test_research.pdf"
    ) == "test_document"


def test_unknown_source():
    assert classify_source(
        "random_document.pdf"
    ) == "unknown"


def test_source_quality_scores():
    assert get_source_quality(
        "https://example.gov/research/ai.pdf"
    ) == 1.00

    assert get_source_quality(
        "academic_research_paper.pdf"
    ) == 0.95

    assert get_source_quality(
        "industry_report.pdf"
    ) == 0.80

    assert get_source_quality(
        "company_blog.pdf"
    ) == 0.50

    assert get_source_quality(
        "random_document.pdf"
    ) == 0.30


def test_source_metadata():
    result = get_source_metadata(
        "academic_research_paper.pdf"
    )

    assert result["source"] == "academic_research_paper.pdf"
    assert result["source_type"] == "academic"
    assert result["quality_score"] == 0.95