from unittest.mock import patch

import pytest

from app.agents.planner import create_research_plan


def test_planner_returns_valid_plan():

    mock_response = """
    {
        "objective": "Evaluate how AI can improve customer support efficiency.",
        "research_questions": [
            "How can AI automate customer support tasks?",
            "How does AI affect response time?"
        ],
        "important_factors": [
            "Automation",
            "Response time",
            "Customer satisfaction"
        ],
        "evidence_required": [
            "Research studies",
            "Case studies"
        ],
        "risks_and_opposing_viewpoints": [
            "Incorrect AI responses",
            "Privacy concerns"
        ]
    }
    """

    with patch(
        "app.agents.planner.generate_content",
        return_value=mock_response
    ):

        result = create_research_plan(
            "How can AI improve customer support efficiency?"
        )

    assert isinstance(result, dict)

    assert "objective" in result
    assert "research_questions" in result
    assert "important_factors" in result
    assert "evidence_required" in result
    assert "risks_and_opposing_viewpoints" in result


def test_planner_rejects_invalid_json():

    with patch(
        "app.agents.planner.generate_content",
        return_value="This is not valid JSON"
    ):

        with pytest.raises(ValueError):

            create_research_plan(
                "How can AI improve customer support efficiency?"
            )


def test_planner_sends_research_question_to_model():

    mock_response = """
    {
        "objective": "Evaluate AI.",
        "research_questions": [
            "How can AI improve efficiency?"
        ],
        "important_factors": [
            "Automation"
        ],
        "evidence_required": [
            "Research studies"
        ],
        "risks_and_opposing_viewpoints": [
            "Implementation risks"
        ]
    }
    """

    with patch(
        "app.agents.planner.generate_content",
        return_value=mock_response
    ) as mock_generate:

        create_research_plan(
            "How can AI improve customer support efficiency?"
        )

    prompt = mock_generate.call_args[0][0]

    assert (
        "How can AI improve customer support efficiency?"
        in prompt
    )


def test_planner_rejects_missing_required_field():

    mock_response = """
    {
        "objective": "Evaluate AI.",
        "research_questions": [
            "How can AI improve efficiency?"
        ]
    }
    """

    with patch(
        "app.agents.planner.generate_content",
        return_value=mock_response
    ):

        with pytest.raises(ValueError):

            create_research_plan(
                "How can AI improve customer support efficiency?"
            )