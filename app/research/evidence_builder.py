from app.research.evidence import Evidence
from app.research.source_quality import (
    classify_source,
    get_source_quality,
)


def build_evidence(results):
    """
    Convert vector-search results into structured Evidence objects.

    Each evidence item contains:

    - Content
    - Source
    - Page
    - Similarity score
    - Evidence ID
    - Source type
    - Source quality score
    """

    evidence_items = []

    for index, result in enumerate(
        results,
        start=1
    ):

        document = result["document"]

        source = document.metadata.get(
            "source",
            "Unknown"
        )

        source_type = classify_source(
            source
        )

        source_quality = get_source_quality(
            source
        )

        evidence = Evidence(
            content=document.page_content,

            source=source,

            page=document.metadata.get(
                "page"
            ),

            similarity_score=float(
                result["score"]
            ),

            evidence_id=f"E{index:03d}",

            source_type=source_type,

            source_quality=source_quality
        )

        evidence_items.append(
            evidence
        )

    return evidence_items