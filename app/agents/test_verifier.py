from app.agents.verifier import verify_evidence, verify_with_gemini


RESEARCH_QUESTION = (
    "How can AI improve customer support efficiency?"
)


EVIDENCE_ITEMS = [
    {
        "evidence_id": "E001",
        "content": (
            "AI agents can automate routine customer support "
            "tasks and improve response efficiency."
        ),
        "source": "test_source.pdf",
        "page": 1,
        "similarity_score": 0.85,
        "source_type": "academic",
        "source_quality": 0.95,
    }
]


def test_verify_evidence_returns_valid_json(monkeypatch):

    fake_result = {
        "evidence_assessment": [
            {
                "evidence_id": "E001",
                "relationship": "Supports",
                "claim_supported": (
                    "AI agents can automate routine customer "
                    "support tasks."
                ),
                "reasoning": (
                    "The evidence directly describes "
                    "automation of routine support tasks."
                ),
            }
        ]
    }

    monkeypatch.setattr(
        "app.agents.verifier.verify_with_gemini",
        lambda research_question, evidence_items: fake_result,
    )

    result = verify_evidence(
        RESEARCH_QUESTION,
        EVIDENCE_ITEMS,
    )

    assert isinstance(result, dict)
    assert "evidence_assessment" in result
    assert len(result["evidence_assessment"]) == 1
    assert result["evidence_assessment"][0]["evidence_id"] == "E001"
    assert result["evidence_assessment"][0]["relationship"] == "Supports"


def test_verify_evidence_handles_markdown_json(monkeypatch):

    fake_result = {
        "evidence_assessment": [
            {
                "evidence_id": "E001",
                "relationship": "Partially Supports",
                "claim_supported": (
                    "AI may improve customer support."
                ),
                "reasoning": (
                    "The evidence is relevant but "
                    "does not fully establish efficiency."
                ),
            }
        ]
    }

    monkeypatch.setattr(
        "app.agents.verifier.verify_with_gemini",
        lambda research_question, evidence_items: fake_result,
    )

    result = verify_evidence(
        RESEARCH_QUESTION,
        EVIDENCE_ITEMS,
    )

    assert isinstance(result, dict)
    assert (
        result["evidence_assessment"][0]["relationship"]
        == "Partially Supports"
    )


def test_verify_evidence_uses_local_fallback_when_gemini_fails(
    monkeypatch
):

    def fake_verify_with_gemini(
        research_question,
        evidence_items
    ):
        return None

    monkeypatch.setattr(
        "app.agents.verifier.verify_with_gemini",
        fake_verify_with_gemini,
    )

    result = verify_evidence(
        RESEARCH_QUESTION,
        EVIDENCE_ITEMS,
    )

    assert isinstance(result, dict)
    assert "evidence_assessment" in result
    assert "verification_status" in result
    assert "local_fallback" in result["verification_status"]
    assert result["evidence_assessment"]


def test_verify_evidence_sends_question_and_evidence_to_model(
    monkeypatch
):

    captured = {}

    def fake_verify_with_gemini(
        research_question,
        evidence_items
    ):
        captured["research_question"] = research_question
        captured["evidence_items"] = evidence_items

        return {
            "evidence_assessment": [
                {
                    "evidence_id": "E001",
                    "relationship": "Supports",
                    "claim_supported": "Test claim",
                    "reasoning": "Test reasoning",
                }
            ]
        }

    monkeypatch.setattr(
        "app.agents.verifier.verify_with_gemini",
        fake_verify_with_gemini,
    )

    verify_evidence(
        RESEARCH_QUESTION,
        EVIDENCE_ITEMS,
    )

    assert captured["research_question"] == RESEARCH_QUESTION
    assert captured["evidence_items"] == EVIDENCE_ITEMS


# =================================================================
# REGRESSION TESTS: exercise verify_with_gemini itself
#
# These mock generate_content (the real Gemini boundary) instead
# of mocking verify_with_gemini. That distinction matters: a
# broken import or a broken response-parsing step inside
# verify_with_gemini would be invisible to every test above,
# since those tests replace the whole function. These tests
# would have caught the previous bug where verify_with_gemini
# imported from a module that didn't exist and silently
# returned None on every call.
# =================================================================

def test_verify_with_gemini_calls_real_client_and_parses_json(
    monkeypatch
):

    fake_response_text = """```json
{
    "evidence_assessment": [
        {
            "evidence_id": "E001",
            "relationship": "Supports",
            "claim_supported": "AI can automate support tasks.",
            "reasoning": "Directly stated in the evidence.",
            "confidence": 0.9
        }
    ]
}
```"""

    monkeypatch.setattr(
        "app.utils.gemini_client.generate_content",
        lambda prompt: fake_response_text,
    )

    result = verify_with_gemini(
        RESEARCH_QUESTION,
        EVIDENCE_ITEMS,
    )

    assert isinstance(result, dict)
    assert "evidence_assessment" in result
    assert result["evidence_assessment"][0]["evidence_id"] == "E001"
    assert result["evidence_assessment"][0]["relationship"] == "Supports"


def test_verify_with_gemini_returns_none_on_invalid_json(
    monkeypatch
):

    monkeypatch.setattr(
        "app.utils.gemini_client.generate_content",
        lambda prompt: "not valid json at all",
    )

    result = verify_with_gemini(
        RESEARCH_QUESTION,
        EVIDENCE_ITEMS,
    )

    assert result is None


def test_verify_with_gemini_returns_none_on_client_failure(
    monkeypatch
):

    def raise_error(prompt):
        raise RuntimeError("Gemini API daily quota has been exceeded.")

    monkeypatch.setattr(
        "app.utils.gemini_client.generate_content",
        raise_error,
    )

    result = verify_with_gemini(
        RESEARCH_QUESTION,
        EVIDENCE_ITEMS,
    )

    assert result is None


def test_verify_with_gemini_imports_from_the_real_module():
    """
    Locks in the fix: verify_with_gemini must import
    generate_content from app.utils.gemini_client (the module
    that actually exists), not a non-existent module.
    """

    import app.agents.verifier as verifier_module
    import inspect

    source = inspect.getsource(
        verifier_module.verify_with_gemini
    )

    assert "app.utils.gemini_client" in source
    assert "app.agents.gemini_client" not in source