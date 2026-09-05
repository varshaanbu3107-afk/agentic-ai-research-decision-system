from unittest.mock import patch

from app.core.orchestrator import run_research_system


def mock_plan():
    return {
        "objective": "Evaluate how AI can improve customer support efficiency.",
        "research_questions": [
            "How can AI automate customer support tasks?"
        ],
        "important_factors": [
            "Automation",
            "Response time"
        ],
        "evidence_required": [
            "Evidence about AI automation"
        ],
        "risks_and_opposing_viewpoints": [
            "AI may struggle with complex cases"
        ]
    }


def mock_research():
    return {
        "research_question": (
            "How can AI improve customer support efficiency?"
        ),
        "retrieved_evidence": (
            "AI chatbots can automate routine customer support tasks."
        ),
        "analysis": {
            "research_question": (
                "How can AI improve customer support efficiency?"
            ),
            "key_findings": [
                "AI can automate routine customer support tasks."
            ],
            "supporting_evidence": [
                "AI chatbots can automate routine customer support tasks."
            ],
            "limitations": [
                "Complex cases may require human agents."
            ],
            "confidence": "High",
            "analysis_status": "llm_based_evidence_analysis"
        }
    }


def mock_verification():
    return {
        "research_question": (
            "How can AI improve customer support efficiency?"
        ),
        "evidence_assessment": [
            {
                "evidence_id": "EVIDENCE 1",
                "relationship": "Supports",
                "claim_supported": (
                    "AI can automate routine customer support tasks."
                ),
                "reasoning": (
                    "The evidence directly supports the finding."
                )
            },
            {
                "evidence_id": "EVIDENCE 2",
                "relationship": "Neutral",
                "claim_supported": None,
                "reasoning": "Not directly relevant."
            }
        ],
        "overall_relevance": "High",
        "verification_confidence": "High",
        "limitations": []
    }


def mock_decision():
    return {
        "research_question": (
            "How can AI improve customer support efficiency?"
        ),
        "decision": (
            "The available evidence supports the research question."
        ),
        "recommendation": (
            "Use AI for routine customer support tasks."
        ),
        "confidence": "High",
        "reasons": [
            "AI can automate routine support tasks."
        ],
        "risks": [
            "Complex cases may require human assistance."
        ],
        "alternatives": [
            "Collect additional evidence."
        ],
        "decision_status": "supported_evidence"
    }


def test_orchestrator_runs_complete_pipeline():

    with patch(
        "app.core.orchestrator.create_research_plan",
        return_value=mock_plan()
    ), patch(
        "app.core.orchestrator.research_with_rag",
        return_value=mock_research()
    ), patch(
        "app.core.orchestrator.verify_evidence",
        return_value=mock_verification()
    ), patch(
        "app.core.orchestrator.make_decision",
        return_value=mock_decision()
    ):

        result = run_research_system(
            "How can AI improve customer support efficiency?"
        )

    assert isinstance(result, dict)

    assert "research_question" in result
    assert "research_plan" in result
    assert "research_analysis" in result
    assert "verification" in result
    assert "decision" in result


def test_orchestrator_preserves_research_question():

    question = "How can AI improve customer support efficiency?"

    with patch(
        "app.core.orchestrator.create_research_plan",
        return_value=mock_plan()
    ), patch(
        "app.core.orchestrator.research_with_rag",
        return_value=mock_research()
    ), patch(
        "app.core.orchestrator.verify_evidence",
        return_value=mock_verification()
    ), patch(
        "app.core.orchestrator.make_decision",
        return_value=mock_decision()
    ):

        result = run_research_system(question)

    assert result["research_question"] == question


def test_orchestrator_passes_all_verified_evidence():

    verification = mock_verification()

    with patch(
        "app.core.orchestrator.create_research_plan",
        return_value=mock_plan()
    ), patch(
        "app.core.orchestrator.research_with_rag",
        return_value=mock_research()
    ), patch(
        "app.core.orchestrator.verify_evidence",
        return_value=verification
    ), patch(
        "app.core.orchestrator.make_decision",
        return_value=mock_decision()
    ) as mock_decision_agent:

        run_research_system(
            "How can AI improve customer support efficiency?"
        )

    decision_call = mock_decision_agent.call_args.kwargs

    verified = decision_call["verification"]["evidence_assessment"]

    # Both verified evidence assessments must be passed
    # to the Decision Agent.
    assert len(verified) == 2

    # Supporting evidence must be preserved.
    assert verified[0]["relationship"] == "Supports"

    # Neutral evidence must also be preserved.
    assert verified[1]["relationship"] == "Neutral"


def test_orchestrator_returns_decision():

    with patch(
        "app.core.orchestrator.create_research_plan",
        return_value=mock_plan()
    ), patch(
        "app.core.orchestrator.research_with_rag",
        return_value=mock_research()
    ), patch(
        "app.core.orchestrator.verify_evidence",
        return_value=mock_verification()
    ), patch(
        "app.core.orchestrator.make_decision",
        return_value=mock_decision()
    ):

        result = run_research_system(
            "How can AI improve customer support efficiency?"
        )

    assert result["decision"]["decision_status"] == (
        "supported_evidence"
    )