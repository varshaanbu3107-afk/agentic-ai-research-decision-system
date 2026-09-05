from unittest.mock import MagicMock, patch

import pytest

from app.utils.gemini_client import generate_content


def _fake_client(response=None, side_effect=None):
    """
    Build a fake Gemini client whose
    `.models.generate_content(...)` behaves like the real
    google-genai client for test purposes.
    """

    client = MagicMock()

    if side_effect is not None:
        client.models.generate_content.side_effect = side_effect
    else:
        client.models.generate_content.return_value = response

    return client


def test_generate_content_returns_response():

    mock_response = type(
        "Response",
        (),
        {
            "text": "Gemini response"
        }
    )()

    with patch(
        "app.utils.gemini_client._get_client",
        return_value=_fake_client(response=mock_response),
    ):

        result = generate_content(
            "Test prompt"
        )

    assert result == "Gemini response"


def test_generate_content_strips_response():

    mock_response = type(
        "Response",
        (),
        {
            "text": "   Gemini response   "
        }
    )()

    with patch(
        "app.utils.gemini_client._get_client",
        return_value=_fake_client(response=mock_response),
    ):

        result = generate_content(
            "Test prompt"
        )

    assert result == "Gemini response"


def test_generate_content_rejects_empty_response():

    mock_response = type(
        "Response",
        (),
        {
            "text": ""
        }
    )()

    with patch(
        "app.utils.gemini_client._get_client",
        return_value=_fake_client(response=mock_response),
    ):

        with pytest.raises(RuntimeError):

            generate_content(
                "Test prompt"
            )


def test_generate_content_handles_daily_quota():

    error = Exception(
        "429 RESOURCE_EXHAUSTED "
        "GenerateRequestsPerDayPerProject-FreeTier"
    )

    with patch(
        "app.utils.gemini_client._get_client",
        return_value=_fake_client(side_effect=error),
    ):

        with pytest.raises(RuntimeError) as exc_info:

            generate_content(
                "Test prompt"
            )

    assert "daily quota" in str(
        exc_info.value
    ).lower()


def test_generate_content_handles_rate_limit():

    error = Exception(
        "429 RESOURCE_EXHAUSTED temporary rate limit"
    )

    with patch(
        "app.utils.gemini_client._get_client",
        return_value=_fake_client(side_effect=error),
    ), patch(
        "app.utils.gemini_client.time.sleep"
    ):

        with pytest.raises(RuntimeError):

            generate_content(
                "Test prompt",
                max_retries=2
            )


def test_generate_content_handles_server_error():

    error = Exception(
        "503 UNAVAILABLE"
    )

    with patch(
        "app.utils.gemini_client._get_client",
        return_value=_fake_client(side_effect=error),
    ), patch(
        "app.utils.gemini_client.time.sleep"
    ):

        with pytest.raises(RuntimeError):

            generate_content(
                "Test prompt",
                max_retries=2
            )


def test_generate_content_handles_connection_error():

    error = Exception(
        "Connection reset by peer"
    )

    with patch(
        "app.utils.gemini_client._get_client",
        return_value=_fake_client(side_effect=error),
    ), patch(
        "app.utils.gemini_client.time.sleep"
    ):

        with pytest.raises(RuntimeError):

            generate_content(
                "Test prompt",
                max_retries=2
            )


# =================================================================
# NEW: lazy-client / missing-key behavior
#
# These lock in the fix where the module no longer crashes at
# import time when GEMINI_API_KEY is unset. Instead, the error
# is raised lazily on first real use, which is what lets every
# agent's local-fallback logic actually run for a clone with no
# API key configured.
# =================================================================

def test_module_imports_without_api_key(monkeypatch):

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # Reload to prove import itself doesn't touch the network
    # or require a key.
    import importlib
    import app.utils.gemini_client as gemini_client_module

    importlib.reload(gemini_client_module)

    assert gemini_client_module._client is None


def test_generate_content_raises_runtime_error_without_api_key(
    monkeypatch
):

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    import app.utils.gemini_client as gemini_client_module
    gemini_client_module._client = None

    with pytest.raises(RuntimeError) as exc_info:

        gemini_client_module.generate_content(
            "Test prompt",
            max_retries=1,
        )

    assert "gemini api" in str(exc_info.value).lower()
