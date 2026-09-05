from unittest.mock import patch

import pytest

from app.agents.researcher import analyze_evidence


def test_researcher_returns_valid_analysis():
    mock_response = """
    {
        "research_question": "How can AI improve customer support efficiency?",
        "key_findings": [
            "AI can automate routine customer support tasks."
        ],
        "supporting_evidence": [
            "AI agents can automate frequently asked questions."
        ],
        "limitations": [
            "Human agents may still be required for complex cases."
        ],
        "confidence": "Medium"
    }
    """

    with patch(
        "app.agents.researcher.generate_content",
        return_value=mock_response
    ):

        result = analyze_evidence(
            "How can AI improve customer support efficiency?",
            "AI agents can automate frequently asked questions."
        )

    assert isinstance(result, dict)

    assert "research_question" in result
    assert "key_findings" in result
    assert "supporting_evidence" in result
    assert "limitations" in result
    assert "confidence" in result
    assert "analysis_status" in result

    assert result["analysis_status"] == (
        "llm_based_evidence_analysis"
    )


def test_researcher_handles_empty_evidence():

    result = analyze_evidence(
        "How can AI improve customer support efficiency?",
        ""
    )

    assert result["key_findings"] == []

    assert result["confidence"] == "Low"

    assert result["analysis_status"] == "no_evidence"

    assert len(result["limitations"]) > 0


def test_researcher_removes_markdown_json():

    mock_response = """
    ```json
    {
        "research_question": "What are the benefits of AI?",
        "key_findings": [
            "AI can automate repetitive tasks."
        ],
        "supporting_evidence": [
            "The evidence describes task automation."
        ],
        "limitations": [
            "The evidence does not provide quantitative results."
        ],
        "confidence": "Medium"
    }
    ```
    """

    with patch(
        "app.agents.researcher.generate_content",
        return_value=mock_response
    ):

        result = analyze_evidence(
            "What are the benefits of AI?",
            "AI can automate repetitive tasks."
        )

    assert result["key_findings"] == [
        "AI can automate repetitive tasks."
    ]


def test_researcher_rejects_invalid_json():

    with patch(
        "app.agents.researcher.generate_content",
        return_value="This is not valid JSON"
    ):

        with pytest.raises(ValueError):

            analyze_evidence(
                "What are the benefits of AI?",
                "AI can automate repetitive tasks."
            )


def test_researcher_sends_question_and_evidence_to_model():

    mock_response = """
    {
        "research_question": "What are the benefits of AI?",
        "key_findings": [
            "AI can automate repetitive tasks."
        ],
        "supporting_evidence": [
            "AI can automate repetitive tasks."
        ],
        "limitations": [],
        "confidence": "Medium"
    }
    """

    with patch(
        "app.agents.researcher.generate_content",
        return_value=mock_response
    ) as mock_generate:

        analyze_evidence(
            "What are the benefits of AI?",
            "AI can automate repetitive tasks."
        )

    prompt = mock_generate.call_args[0][0]

    assert "What are the benefits of AI?" in prompt

    assert "AI can automate repetitive tasks." in prompt