import os
import time

from dotenv import load_dotenv
from google import genai


# =============================================================
# LOAD ENVIRONMENT
# =============================================================

load_dotenv()


# =============================================================
# GEMINI CLIENT (lazy)
#
# The client is created on first use rather than at import
# time. This module is imported by the planner, researcher,
# verifier, and decision agents, all of which are designed to
# fall back to local deterministic logic when Gemini is
# unavailable. Raising at import time defeats that design:
# it would crash the whole process (and the test suite) before
# any fallback logic ever gets a chance to run, even for a
# clone that never intends to use Gemini at all.
# =============================================================

_client = None


def _get_client():

    global _client

    if _client is not None:
        return _client

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "Gemini API key is not set (GEMINI_API_KEY). "
            "Add it to a .env file in the project root, or "
            "the pipeline will use local fallback logic."
        )

    _client = genai.Client(
        api_key=api_key
    )

    return _client


# =============================================================
# GENERATE CONTENT
# =============================================================

def generate_content(
    prompt: str,
    model: str = "gemini-2.5-flash",
    max_retries: int = 3
) -> str:
    """
    Generate content using the Gemini API.

    Handles:
    - Rate limits
    - Daily quota exhaustion
    - Temporary server errors
    - Connection failures
    - Empty responses

    Daily quota exhaustion is NOT retried because
    retrying cannot restore the daily quota.
    """

    for attempt in range(1, max_retries + 1):

        try:

            print(
                f"\nGemini API request "
                f"(attempt {attempt}/{max_retries})..."
            )

            response = _get_client().models.generate_content(
                model=model,
                contents=prompt
            )

            # -------------------------------------------------
            # VALIDATE RESPONSE
            # -------------------------------------------------

            if not response.text:
                raise RuntimeError(
                    "Gemini returned an empty response."
                )

            return response.text.strip()

        except Exception as error:

            error_text = str(error)
            error_lower = error_text.lower()

            # =================================================
            # QUOTA / RATE LIMIT
            # =================================================

            if (
                "429" in error_text
                or "resource_exhausted" in error_lower
            ):

                # -------------------------------------------------
                # DAILY QUOTA
                # -------------------------------------------------

                if (
                    "perday" in error_lower
                    or "per day" in error_lower
                    or "daily" in error_lower
                    or "quota exceeded" in error_lower
                ):

                    print(
                        "\nGemini daily quota exhausted."
                    )

                    raise RuntimeError(
                        "Gemini API daily quota has been exceeded. "
                        "Wait for the quota to reset or enable billing "
                        "for higher limits."
                    ) from error

                # -------------------------------------------------
                # TEMPORARY RATE LIMIT
                # -------------------------------------------------

                if attempt < max_retries:

                    wait_time = 10 * attempt

                    print(
                        "Gemini rate limit detected. "
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                    continue

                raise RuntimeError(
                    "Gemini API rate limit exceeded "
                    "after multiple attempts."
                ) from error

            # =================================================
            # TEMPORARY SERVER ERRORS
            # =================================================

            if (
                "503" in error_text
                or "unavailable" in error_lower
                or "500" in error_text
                or "internal" in error_lower
                or "502" in error_text
                or "504" in error_text
            ):

                if attempt < max_retries:

                    wait_time = 5 * attempt

                    print(
                        "Gemini temporary server error detected. "
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                    continue

                raise RuntimeError(
                    "Gemini service is temporarily unavailable "
                    "after multiple attempts. "
                    "Please try again later."
                ) from error

            # =================================================
            # CONNECTION ERRORS
            # =================================================

            if (
                "readerror" in error_lower
                or "winerror 10054" in error_lower
                or "connection" in error_lower
                or "timeout" in error_lower
            ):

                if attempt < max_retries:

                    wait_time = 5 * attempt

                    print(
                        "Gemini connection failed. "
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                    continue

                raise RuntimeError(
                    "Unable to connect to the Gemini API "
                    "after multiple attempts. "
                    "Please check your internet connection "
                    "and try again later."
                ) from error

            # =================================================
            # UNKNOWN ERROR
            # =================================================

            raise


# =============================================================
# DIRECT TEST
# =============================================================

if __name__ == "__main__":

    test_prompt = (
        "Explain in one sentence how AI can improve "
        "customer support efficiency."
    )

    try:

        result = generate_content(
            test_prompt
        )

        print("\n" + "=" * 60)
        print("GEMINI TEST RESULT")
        print("=" * 60)

        print(result)

    except Exception as error:

        print("\n" + "=" * 60)
        print("GEMINI TEST FAILED")
        print("=" * 60)

        print(error)

