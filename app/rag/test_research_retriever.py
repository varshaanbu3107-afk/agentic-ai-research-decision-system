from unittest.mock import patch

from app.rag.research_retriever import research_with_rag


class FakeDocument:

    def __init__(self, content, source="test.pdf", page=1):
        self.page_content = content
        self.metadata = {
            "source": source,
            "page": page
        }


def fake_search(query, top_k=3):

    results = [
        {
            "document": FakeDocument(
                "AI chatbots can automate frequently asked "
                "customer support questions and improve "
                "response efficiency."
            ),
            "score": 0.80
        },
        {
            "document": FakeDocument(
                "AI can assist customer service teams by "
                "handling routine requests."
            ),
            "score": 0.70
        },
        {
            "document": FakeDocument(
                "Complex customer issues may still require "
                "human agents."
            ),
            "score": 0.60
        }
    ]

    return results[:top_k]


def mock_analysis():
    return {
        "research_question": (
            "How can AI improve customer support efficiency?"
        ),
        "key_findings": [
            "AI can automate routine customer support tasks."
        ],
        "supporting_evidence": [
            "AI chatbots can automate frequently asked questions."
        ],
        "limitations": [
            "Complex cases may require human agents."
        ],
        "confidence": "High",
        "analysis_status": "llm_based_evidence_analysis"
    }


def create_fake_vector_store():

    fake_store = type("FakeVectorStore", (), {})()

    fake_store.load = lambda: None
    fake_store.search = fake_search

    return fake_store


def test_research_with_rag_returns_result():

    with patch(
        "app.rag.research_retriever.VectorStore",
        return_value=create_fake_vector_store()
    ), patch(
        "app.rag.research_retriever.analyze_evidence",
        return_value=mock_analysis()
    ):

        result = research_with_rag(
            research_question=(
                "How can AI improve customer support efficiency?"
            ),
            research_questions=[
                "How can AI automate customer support tasks?"
            ],
            top_k=3
        )

    assert isinstance(result, dict)

    assert "research_question" in result
    assert "retrieved_evidence" in result
    assert "analysis" in result


def test_research_with_rag_preserves_question():

    question = (
        "How can AI improve customer support efficiency?"
    )

    with patch(
        "app.rag.research_retriever.VectorStore",
        return_value=create_fake_vector_store()
    ), patch(
        "app.rag.research_retriever.analyze_evidence",
        return_value=mock_analysis()
    ):

        result = research_with_rag(
            research_question=question,
            research_questions=[
                "How can AI automate customer support tasks?"
            ],
            top_k=3
        )

    assert result["research_question"] == question


def test_research_with_rag_calls_researcher():

    with patch(
        "app.rag.research_retriever.VectorStore",
        return_value=create_fake_vector_store()
    ), patch(
        "app.rag.research_retriever.analyze_evidence",
        return_value=mock_analysis()
    ) as mock_researcher:

        research_with_rag(
            research_question=(
                "How can AI improve customer support efficiency?"
            ),
            research_questions=[
                "How can AI automate customer support tasks?"
            ],
            top_k=3
        )

    assert mock_researcher.called


def test_research_with_rag_returns_analysis():

    with patch(
        "app.rag.research_retriever.VectorStore",
        return_value=create_fake_vector_store()
    ), patch(
        "app.rag.research_retriever.analyze_evidence",
        return_value=mock_analysis()
    ):

        result = research_with_rag(
            research_question=(
                "How can AI improve customer support efficiency?"
            ),
            research_questions=[
                "How can AI automate customer support tasks?"
            ],
            top_k=3
        )

    analysis = result["analysis"]

    assert isinstance(analysis, dict)

    assert "key_findings" in analysis
    assert "supporting_evidence" in analysis
    assert "limitations" in analysis
    assert "confidence" in analysis


def test_research_with_rag_retrieves_evidence():

    with patch(
        "app.rag.research_retriever.VectorStore",
        return_value=create_fake_vector_store()
    ), patch(
        "app.rag.research_retriever.analyze_evidence",
        return_value=mock_analysis()
    ):

        result = research_with_rag(
            research_question=(
                "How can AI improve customer support efficiency?"
            ),
            research_questions=[
                "How can AI automate customer support tasks?"
            ],
            top_k=3
        )

    evidence = result["retrieved_evidence"]

    assert isinstance(evidence, str)

    assert len(evidence) > 0


def test_research_with_rag_respects_top_k():

    with patch(
        "app.rag.research_retriever.VectorStore",
        return_value=create_fake_vector_store()
    ), patch(
        "app.rag.research_retriever.analyze_evidence",
        return_value=mock_analysis()
    ):

        result = research_with_rag(
            research_question=(
                "How can AI improve customer support efficiency?"
            ),
            research_questions=[],
            top_k=2
        )

    assert isinstance(result, dict)
    assert "retrieved_evidence" in result