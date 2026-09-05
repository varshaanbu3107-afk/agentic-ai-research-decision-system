from types import SimpleNamespace

from app.research.evidence_builder import build_evidence


def create_search_result(
    content="AI can automate customer support.",
    source="research.pdf",
    page=3,
    score=0.82,
):
    document = SimpleNamespace(
        page_content=content,
        metadata={
            "source": source,
            "page": page,
        },
    )

    return {
        "document": document,
        "score": score,
    }


def test_build_evidence_creates_evidence():
    results = [
        create_search_result()
    ]

    evidence_items = build_evidence(results)

    assert len(evidence_items) == 1

    evidence = evidence_items[0]

    assert evidence.content == "AI can automate customer support."
    assert evidence.source == "research.pdf"
    assert evidence.page == 3
    assert evidence.similarity_score == 0.82
    assert evidence.evidence_id == "E001"


def test_build_evidence_creates_multiple_items():
    results = [
        create_search_result(
            content="First evidence",
            source="paper1.pdf",
            page=1,
            score=0.90,
        ),
        create_search_result(
            content="Second evidence",
            source="paper2.pdf",
            page=5,
            score=0.75,
        ),
        create_search_result(
            content="Third evidence",
            source="paper3.pdf",
            page=8,
            score=0.60,
        ),
    ]

    evidence_items = build_evidence(results)

    assert len(evidence_items) == 3

    assert evidence_items[0].evidence_id == "E001"
    assert evidence_items[1].evidence_id == "E002"
    assert evidence_items[2].evidence_id == "E003"


def test_build_evidence_handles_empty_results():
    evidence_items = build_evidence([])

    assert evidence_items == []


def test_build_evidence_uses_unknown_source():
    document = SimpleNamespace(
        page_content="Test content",
        metadata={
            "page": 2,
        },
    )

    results = [
        {
            "document": document,
            "score": 0.70,
        }
    ]

    evidence_items = build_evidence(results)

    assert evidence_items[0].source == "Unknown"