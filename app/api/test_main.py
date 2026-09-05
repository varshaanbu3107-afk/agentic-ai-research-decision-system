from fastapi.testclient import TestClient

from app.api.main import app


client = TestClient(app)


def test_health_endpoint():

    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "healthy"
    assert "service" in body


def test_research_rejects_empty_question():

    response = client.post(
        "/research",
        json={"question": "   "},
    )

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_research_returns_pipeline_result(monkeypatch):

    fake_result = {
        "research_question": "How can AI improve customer support?",
        "decision": "Supported",
        "confidence": "High",
    }

    def fake_run_research_system(question):
        return fake_result

    monkeypatch.setattr(
        "app.core.orchestrator.run_research_system",
        fake_run_research_system,
    )

    response = client.post(
        "/research",
        json={"question": "How can AI improve customer support?"},
    )

    assert response.status_code == 200
    assert response.json() == fake_result


def test_research_returns_500_on_pipeline_error(monkeypatch):

    def fake_run_research_system(question):
        raise RuntimeError("Vector store not found.")

    monkeypatch.setattr(
        "app.core.orchestrator.run_research_system",
        fake_run_research_system,
    )

    response = client.post(
        "/research",
        json={"question": "How can AI improve customer support?"},
    )

    assert response.status_code == 500
    assert "vector store" in response.json()["detail"].lower()


def test_research_strips_whitespace_before_calling_pipeline(
    monkeypatch
):

    captured = {}

    def fake_run_research_system(question):
        captured["question"] = question
        return {"ok": True}

    monkeypatch.setattr(
        "app.core.orchestrator.run_research_system",
        fake_run_research_system,
    )

    client.post(
        "/research",
        json={"question": "  How can AI help?  "},
    )

    assert captured["question"] == "How can AI help?"
