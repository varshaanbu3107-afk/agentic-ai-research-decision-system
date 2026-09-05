import re


SOURCE_QUALITY_SCORES = {
    "government": 1.00,
    "academic": 0.95,
    "peer_reviewed": 0.95,
    "official_report": 0.90,
    "industry_report": 0.80,
    "official_company": 0.75,
    "news": 0.65,
    "company_blog": 0.50,
    "unknown": 0.30,
    "test_document": 0.10,
}


def classify_source(source: str) -> str:
    """
    Classify a research source into a source-quality category.

    Priority:
        1. Test document
        2. PubMed / NCBI academic sources
        3. WHO / government sources
        4. News
        5. Peer-reviewed sources
        6. Academic sources
        7. Research documents
        8. Official reports
        9. Industry reports
        10. Official company sources
        11. Company blogs
        12. Unknown
    """

    source_lower = str(source).lower().strip()

    # Normalize -, _ and spaces
    normalized = re.sub(
        r"[_\-]+",
        " ",
        source_lower
    )

    # =====================================================
    # 1. TEST DOCUMENT
    # =====================================================

    test_indicators = [
        "test_research",
        "test_document",
        "test_",
        "\\test\\",
        "/test/",
    ]

    if any(
        indicator in source_lower
        for indicator in test_indicators
    ):
        return "test_document"

    # =====================================================
    # 2. PUBMED / NCBI
    # =====================================================
    #
    # PubMed is an academic/medical research source.
    # It must be checked BEFORE government indicators.
    # =====================================================

    academic_medical_indicators = [
        "pubmed",
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi",
        "ncbi.nlm.nih.gov",
        "bmj",
        "thelancet",
        "lancet",
        "plos",
        "nejm",
        "jama",
    ]

    if any(
        indicator in source_lower
        for indicator in academic_medical_indicators
    ):
        return "academic"

    # =====================================================
    # 3. WHO / GOVERNMENT
    # =====================================================

    government_indicators = [
        ".gov",
        ".gov.",
        "government",
        "gov_report",
        "government_report",

        # World Health Organization
        "who.int",
        "world health organization",
        "who ai healthcare",
        "who ai health",
        "who healthcare",
        "who health ethics",
    ]

    if any(
        indicator in source_lower
        for indicator in government_indicators
    ):
        return "government"

    # =====================================================
    # 4. NEWS
    # =====================================================

    news_indicators = [
        "news",
        ".news",
        "news article",
        "news_article",
        "press release",
        "press_release",

        # News organizations
        "reuters",
        "bbc",
        "cnn",
        "associated press",
        "ap news",
        "the guardian",
        "new york times",
        "washington post",
        "times of india",
        "the hindu",
        "indian express",
    ]

    if any(
        indicator in normalized
        for indicator in news_indicators
    ):
        return "news"

    # =====================================================
    # 5. PEER-REVIEWED
    # =====================================================

    peer_reviewed_indicators = [
        "peer-reviewed",
        "peer_reviewed",
        "peer reviewed",
    ]

    if any(
        indicator in source_lower
        for indicator in peer_reviewed_indicators
    ):
        return "peer_reviewed"

    # =====================================================
    # 6. ACADEMIC
    # =====================================================

    academic_indicators = [

        # Academic domains
        ".edu",
        ".ac.",

        # Academic terminology
        "academic",
        "research_paper",
        "research paper",
        "research study",
        "research article",
        "scientific paper",
        "scientific study",
        "scholarly",
        "journal",
        "conference",
        "conference_paper",
        "conference paper",
        "proceedings",
        "systematic_review",
        "systematic review",
        "literature review",
        "meta analysis",
        "meta-analysis",
        "case study",

        # Academic publishers
        "electronics-",
        "mdpi",
        "ieee",
        "acm",
        "springer",
        "elsevier",
        "sciencedirect",
        "nature.com",
        "frontiers",
        "arxiv",
        "researchgate",
        "wiley",
        "tandfonline",
        "sagepub",
        "jstor",

        # Medical academic sources
        "oxford",
        "cambridge",
        "cell",
    ]

    if any(
        indicator in source_lower
        for indicator in academic_indicators
    ):
        return "academic"

    # =====================================================
    # 7. RESEARCH DOCUMENT FILENAMES
    # =====================================================

    research_filename_indicators = [

        "research",
        "study",
        "scientific",

        # Healthcare / AI
        "healthcare ai",
        "ai healthcare",
        "ai in healthcare",
        "artificial intelligence in healthcare",
        "genai in healthcare",
        "clinical ai",
        "llms in clinical",
        "clinical",

        # AI research topics
        "human in the loop",
        "automation bias",
        "deskilling",
        "cognitive risks",
        "ai adoption",
        "ai coding assistants",
        "ai assisted code generation",
        "ai-assisted code generation",
    ]

    if any(
        indicator in normalized
        for indicator in research_filename_indicators
    ):
        return "academic"

    # =====================================================
    # 8. OFFICIAL REPORT
    # =====================================================

    official_report_indicators = [
        "official report",
        "official_report",
        "annual report",
        "annual_report",
        "technical report",
        "technical_report",
        "whitepaper",
        "white paper",
        "white_paper",
        "policy report",
        "policy_report",
        "government report",
        "government_report",
        "guideline",
        "guidelines",
        "framework",
        "recommendation",
        "recommendations",
    ]

    if any(
        indicator in normalized
        for indicator in official_report_indicators
    ):
        return "official_report"

    # =====================================================
    # 9. INDUSTRY REPORT
    # =====================================================

    industry_indicators = [
        "industry report",
        "industry_report",
        "market report",
        "market_report",
        "industry study",
        "industry_study",
        "market study",
        "market_study",
        "industry analysis",
        "industry_analysis",
    ]

    if any(
        indicator in normalized
        for indicator in industry_indicators
    ):
        return "industry_report"

    # =====================================================
    # 10. OFFICIAL COMPANY
    # =====================================================

    official_company_indicators = [
        "official company",
        "official_company",
        "corporate",
        "company report",
        "company_report",
        "technical documentation",
        "technical_documentation",
        "official documentation",
        "official_documentation",
    ]

    if any(
        indicator in normalized
        for indicator in official_company_indicators
    ):
        return "official_company"

    # =====================================================
    # 11. COMPANY BLOG
    # =====================================================

    company_blog_indicators = [
        "company blog",
        "company_blog",
        "blog",
    ]

    if any(
        indicator in normalized
        for indicator in company_blog_indicators
    ):
        return "company_blog"

    # =====================================================
    # 12. UNKNOWN
    # =====================================================

    return "unknown"


def get_source_quality(source: str) -> float:
    """
    Return a source-quality score between 0 and 1.
    """

    source_type = classify_source(source)

    return SOURCE_QUALITY_SCORES[source_type]


def get_source_metadata(source: str) -> dict:
    """
    Return structured metadata about the source.
    """

    source_type = classify_source(source)

    return {
        "source": source,
        "source_type": source_type,
        "quality_score": SOURCE_QUALITY_SCORES[source_type],
    }


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("SOURCE QUALITY TEST")
    print("=" * 60)

    test_sources = [

        # =================================================
        # GOVERNMENT
        # =================================================

        "https://example.gov/research/ai.pdf",

        "WHO AI Healthcare Ethics.pdf",

        "https://www.who.int/publications/ai-health.pdf",

        # =================================================
        # ACADEMIC
        # =================================================

        "academic_research_paper.pdf",

        "electronics-11-01579-1.pdf",

        "https://pubmed.ncbi.nlm.nih.gov/38733640/",

        "AI human in the loop.pdf",

        "Automation bias, deskilling and cognitive risks.pdf",

        "GenAI in Healthcare.pdf",

        "LLMs in Clinical AI.pdf",

        "AI adoption.pdf",

        "AI coding assistants.pdf",

        "AI-assisted code generation.pdf",

        # =================================================
        # REPORTS
        # =================================================

        "official_report.pdf",

        "industry_report.pdf",

        # =================================================
        # COMPANY
        # =================================================

        "official_company_report.pdf",

        "company_blog.pdf",

        # =================================================
        # NEWS
        # =================================================

        "news_article.pdf",

        "Reuters AI healthcare article.pdf",

        # =================================================
        # UNKNOWN
        # =================================================

        "random_document.pdf",

        # =================================================
        # TEST
        # =================================================

        "test_research.pdf",
    ]

    for source in test_sources:

        result = get_source_metadata(source)

        print("\n" + "-" * 60)

        print(
            f"Source: {result['source']}"
        )

        print(
            f"Type: {result['source_type']}"
        )

        print(
            f"Quality score: "
            f"{result['quality_score']:.2f}"
        )