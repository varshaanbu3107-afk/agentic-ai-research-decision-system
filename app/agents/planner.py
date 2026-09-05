import json

from app.utils.gemini_client import generate_content


def create_research_plan(
    question: str
) -> dict:
    """
    Create and validate a structured research plan.

    Gemini is used when available.

    If Gemini API/quota is unavailable, a deterministic
    local fallback plan is generated.

    Validation errors are NOT converted into fallback results.
    This preserves the original test behavior.
    """

    # ---------------------------------------------------------
    # 1. VALIDATE INPUT
    # ---------------------------------------------------------

    if not question or not question.strip():

        raise ValueError(
            "Research question cannot be empty."
        )

    question = question.strip()

    # ---------------------------------------------------------
    # 2. BUILD PLANNER PROMPT
    # ---------------------------------------------------------

    prompt = f"""
You are a Research Planning Agent in an advanced
Agentic AI Research and Decision-Making System.

Your job is to break a complex research question into
structured research tasks.

User question:
{question}

Return ONLY valid JSON.

Use exactly this structure:

{{
    "objective": "string",
    "research_questions": [
        "string"
    ],
    "important_factors": [
        "string"
    ],
    "evidence_required": [
        "string"
    ],
    "risks_and_opposing_viewpoints": [
        "string"
    ]
}}

Requirements:

- objective must describe what the research should determine.
- research_questions must contain specific questions that researchers
  need to investigate.
- important_factors must identify major decision factors.
- evidence_required must identify evidence that should be collected.
- risks_and_opposing_viewpoints must identify potential risks,
  limitations, and arguments against the decision.
- Do not provide an answer to the original question.
- Return JSON only.
"""

    # ---------------------------------------------------------
    # 3. CALL GEMINI
    # ---------------------------------------------------------

    try:

        text = generate_content(prompt)

    except RuntimeError as error:

        error_text = str(error).lower()

        # -----------------------------------------------------
        # ONLY API AVAILABILITY PROBLEMS USE FALLBACK
        # -----------------------------------------------------

        api_failure_keywords = [
            "quota",
            "rate limit",
            "resource exhausted",
            "unable to connect",
            "temporarily unavailable",
            "gemini api",
            "connection",
            "timeout",
            "service unavailable"
        ]

        if not any(
            keyword in error_text
            for keyword in api_failure_keywords
        ):

            raise

        print(
            "\nGemini Planner unavailable."
        )

        print(
            "Using local fallback research planner."
        )

        print(
            f"Reason: {error}"
        )

        return _create_local_fallback_plan(
            question
        )

    # ---------------------------------------------------------
    # 4. VALIDATE GEMINI RESPONSE
    # ---------------------------------------------------------

    if not text or not text.strip():

        raise ValueError(
            "Planner returned an empty response."
        )

    text = text.strip()

    # ---------------------------------------------------------
    # 5. REMOVE MARKDOWN CODE FENCES
    # ---------------------------------------------------------

    if text.startswith("```json"):

        text = text[7:]

    elif text.startswith("```"):

        text = text[3:]

    if text.endswith("```"):

        text = text[:-3]

    text = text.strip()

    # ---------------------------------------------------------
    # 6. PARSE JSON
    # ---------------------------------------------------------

    try:

        result = json.loads(text)

    except json.JSONDecodeError as error:

        raise ValueError(
            f"Planner returned invalid JSON:\n{text}"
        ) from error

    # ---------------------------------------------------------
    # 7. VALIDATE RESULT TYPE
    # ---------------------------------------------------------

    if not isinstance(result, dict):

        raise ValueError(
            "Planner response must be a JSON object."
        )

    # ---------------------------------------------------------
    # 8. VALIDATE REQUIRED FIELDS
    # ---------------------------------------------------------

    required_fields = [
        "objective",
        "research_questions",
        "important_factors",
        "evidence_required",
        "risks_and_opposing_viewpoints"
    ]

    for field in required_fields:

        if field not in result:

            raise ValueError(
                f"Planner response is missing "
                f"required field: {field}"
            )

    # ---------------------------------------------------------
    # 9. VALIDATE OBJECTIVE
    # ---------------------------------------------------------

    if not isinstance(
        result["objective"],
        str
    ):

        raise ValueError(
            "Planner 'objective' must be a string."
        )

    # ---------------------------------------------------------
    # 10. VALIDATE LIST FIELDS
    # ---------------------------------------------------------

    list_fields = [
        "research_questions",
        "important_factors",
        "evidence_required",
        "risks_and_opposing_viewpoints"
    ]

    for field in list_fields:

        if not isinstance(
            result[field],
            list
        ):

            raise ValueError(
                f"Planner '{field}' must be a list."
            )

        for item in result[field]:

            if not isinstance(item, str):

                raise ValueError(
                    f"Planner '{field}' must contain "
                    "only strings."
                )

    # ---------------------------------------------------------
    # 11. MARK GEMINI PLAN
    # ---------------------------------------------------------

    result["planning_status"] = "llm_based"

    return result


# =============================================================
# LOCAL FALLBACK PLANNER
# =============================================================

def _create_local_fallback_plan(
    question: str
) -> dict:
    """
    Create a deterministic research plan without Gemini.

    This allows the pipeline to continue when the Gemini API
    quota is exhausted or temporarily unavailable.
    """

    return {

        "objective": (
            f"Determine how the topic described by the "
            f"research question can be evaluated using "
            f"available evidence: {question}"
        ),

        "research_questions": [

            question,

            (
                f"What specific technologies, methods, "
                f"or approaches are relevant to: {question}"
            ),

            (
                f"How do these approaches affect efficiency, "
                f"performance, cost, or outcomes related to: "
                f"{question}"
            ),

            (
                f"What measurable benefits can be identified "
                f"from the approaches related to: {question}"
            ),

            (
                f"What limitations, risks, or challenges "
                f"should be considered for: {question}"
            ),

            (
                f"What best practices can improve successful "
                f"implementation of solutions related to: "
                f"{question}"
            ),

            (
                f"What data, infrastructure, and human "
                f"requirements are necessary for: {question}"
            )
        ],

        "important_factors": [

            "Effectiveness",

            "Efficiency",

            "Implementation requirements",

            "Cost and operational impact",

            "Customer or user experience",

            "Scalability",

            "Risks and limitations"
        ],

        "evidence_required": [

            "Peer-reviewed research",

            "Measured performance outcomes",

            "Implementation evidence",

            "Cost or efficiency measurements",

            "Documented limitations",

            "Comparative evidence",

            "Operational requirements"
        ],

        "risks_and_opposing_viewpoints": [

            "AI systems may produce incorrect or unreliable results.",

            "Implementation may require significant infrastructure and data.",

            "Human oversight may still be required for complex cases.",

            "Poor implementation may reduce user satisfaction.",

            "Benefits may vary depending on organization, industry, "
            "and task complexity."
        ],

        "planning_status": "local_fallback"
    }


# =============================================================
# DIRECT TEST
# =============================================================

if __name__ == "__main__":

    question = input(
        "\nEnter your research question: "
    )

    plan = create_research_plan(
        question
    )

    print("\n" + "=" * 60)
    print("STRUCTURED RESEARCH PLAN")
    print("=" * 60)

    print(
        json.dumps(
            plan,
            indent=4
        )
    )