import json

from app.agents.planner import create_research_plan
from app.agents.verifier import verify_evidence
from app.agents.decision import make_decision
from app.rag.research_retriever import research_with_rag


# =============================================================
# DECISION REASONS
# =============================================================

def _build_clean_decision_reasons(
    verification: dict,
    evidence_items: list,
    average_quality: float = 0.0
) -> list:
    """
    Build concise, structured decision reasons.

    Raw evidence chunks are not copied into the final
    decision reasons. Instead, the verified evidence
    structure is summarized.
    """

    assessments = verification.get(
        "evidence_assessment",
        []
    )

    if not isinstance(assessments, list):
        assessments = []

    supports = sum(
        1
        for item in assessments
        if isinstance(item, dict)
        and item.get("relationship") == "Supports"
    )

    partial = sum(
        1
        for item in assessments
        if isinstance(item, dict)
        and item.get("relationship") == "Partially Supports"
    )

    contradicts = sum(
        1
        for item in assessments
        if isinstance(item, dict)
        and item.get("relationship") == "Contradicts"
    )

    neutral = sum(
        1
        for item in assessments
        if isinstance(item, dict)
        and item.get("relationship") == "Neutral"
    )

    total = len(assessments)

    reasons = []

    # ---------------------------------------------------------
    # SUPPORTING EVIDENCE
    # ---------------------------------------------------------

    if supports > 0:

        reasons.append(
            f"{supports} of {total} verified evidence item(s) "
            "directly support the research question."
        )

    # ---------------------------------------------------------
    # PARTIAL EVIDENCE
    # ---------------------------------------------------------

    if partial > 0:

        reasons.append(
            f"{partial} evidence item(s) provide partial or "
            "indirect support rather than strong direct evidence."
        )

    # ---------------------------------------------------------
    # CONTRADICTING EVIDENCE
    # ---------------------------------------------------------

    if contradicts > 0:

        reasons.append(
            f"{contradicts} evidence item(s) contradict the "
            "research question and reduce confidence in the conclusion."
        )

    else:

        reasons.append(
            "No retrieved evidence was classified as contradicting "
            "the research question."
        )

    # ---------------------------------------------------------
    # NEUTRAL EVIDENCE
    # ---------------------------------------------------------

    if neutral > 0:

        reasons.append(
            f"{neutral} evidence item(s) were classified as neutral "
            "because they did not provide sufficient direct support."
        )

    # ---------------------------------------------------------
    # SOURCE QUALITY
    # ---------------------------------------------------------

    if average_quality > 0:

        reasons.append(
            "The selected evidence has an average source/evidence "
            f"quality score of {average_quality:.2f}%."
        )

    # ---------------------------------------------------------
    # RELEVANCE
    # ---------------------------------------------------------

    overall_relevance = verification.get(
        "overall_relevance",
        "Low"
    )

    reasons.append(
        f"Overall evidence relevance was assessed as "
        f"{overall_relevance}."
    )

    return reasons


# =============================================================
# CLEAN LIMITATIONS
# =============================================================

def _build_clean_limitations(
    verification: dict,
    analysis: dict
) -> list:
    """
    Build a clean, non-duplicated limitations list.
    """

    limitations = []

    # ---------------------------------------------------------
    # GEMINI / LOCAL FALLBACK
    # ---------------------------------------------------------

    verification_status = str(
        verification.get(
            "verification_status",
            ""
        )
    ).lower()

    analysis_limitations = analysis.get(
        "limitations",
        []
    )

    if not isinstance(
        analysis_limitations,
        list
    ):
        analysis_limitations = []

    analysis_limitations_text = " ".join(
        str(item).lower()
        for item in analysis_limitations
    )

    used_local_fallback = (
        "local_fallback" in verification_status
        or "gemini was unavailable" in analysis_limitations_text
        or "deterministic local" in analysis_limitations_text
    )

    if used_local_fallback:

        limitations.append(
            "Gemini was unavailable, so deterministic local "
            "reasoning and verification were used."
        )

        limitations.append(
            "The local reasoning and verification use rule-based "
            "concept and context matching rather than full semantic "
            "LLM reasoning."
        )

    # ---------------------------------------------------------
    # PARTIAL / CONTRADICTORY EVIDENCE
    # ---------------------------------------------------------

    assessments = verification.get(
        "evidence_assessment",
        []
    )

    if not isinstance(
        assessments,
        list
    ):
        assessments = []

    partial = sum(
        1
        for item in assessments
        if isinstance(item, dict)
        and item.get(
            "relationship"
        ) == "Partially Supports"
    )

    contradicts = sum(
        1
        for item in assessments
        if isinstance(item, dict)
        and item.get(
            "relationship"
        ) == "Contradicts"
    )

    if partial > 0:

        limitations.append(
            f"{partial} evidence item(s) provided only "
            "partial or indirect support."
        )

    if contradicts > 0:

        limitations.append(
            f"{contradicts} evidence item(s) contradicted "
            "the research question."
        )

    # ---------------------------------------------------------
    # EVIDENCE SCOPE
    # ---------------------------------------------------------

    limitations.append(
        "The findings are limited to the evidence retrieved "
        "by the research pipeline."
    )

    # ---------------------------------------------------------
    # REMOVE DUPLICATES
    # ---------------------------------------------------------

    unique_limitations = []

    for limitation in limitations:

        normalized = limitation.strip().lower()

        if not any(
            normalized == existing.strip().lower()
            for existing in unique_limitations
        ):

            unique_limitations.append(
                limitation
            )

    return unique_limitations


# =============================================================
# MAIN RESEARCH SYSTEM
# =============================================================

def run_research_system(
    research_question: str,
    top_k: int = 3
) -> dict:
    """
    Run the complete Agentic AI Research
    and Decision pipeline.

    Pipeline:

        Research Question
              ↓
           Planner
              ↓
        RAG Retrieval
              ↓
        Evidence Builder
              ↓
        Source Quality
              ↓
        Research Agent
              ↓
           Verifier
              ↓
       Verified Evidence
              ↓
        Decision Agent
              ↓
         Final Result
    """

    print(
        "\n" + "=" * 60
    )

    print(
        "AGENTIC AI RESEARCH DECISION SYSTEM"
    )

    print(
        "=" * 60
    )

    # =========================================================
    # STEP 1: PLANNER
    # =========================================================

    print(
        "\n[1/5] Creating research plan..."
    )

    plan = create_research_plan(
        research_question
    )

    if not isinstance(
        plan,
        dict
    ):
        plan = {}

    print(
        "Research plan created."
    )

    # =========================================================
    # STEP 2: RAG + RESEARCH AGENT
    # =========================================================

    print(
        "\n[2/5] Retrieving evidence and analyzing research..."
    )

    research_questions = plan.get(
        "research_questions",
        []
    )

    if not isinstance(
        research_questions,
        list
    ):
        research_questions = []

    research_result = research_with_rag(
        research_question=research_question,
        research_questions=research_questions,
        top_k=top_k
    )

    if not isinstance(
        research_result,
        dict
    ):
        research_result = {}

    analysis = research_result.get(
        "analysis",
        {}
    )

    if not isinstance(
        analysis,
        dict
    ):
        analysis = {}

    # =========================================================
    # GET STRUCTURED EVIDENCE
    # =========================================================

    evidence_items = research_result.get(
        "evidence_items",
        []
    )

    # Compatibility fallback for older versions
    # of research_retriever.py.
    if not evidence_items:

        evidence_items = research_result.get(
            "retrieved_evidence",
            []
        )

    if not isinstance(
        evidence_items,
        list
    ):
        evidence_items = []

    print(
        "Research analysis completed."
    )

    print(
        f"Structured evidence available for verification: "
        f"{len(evidence_items)}"
    )

    # =========================================================
    # STEP 3: VERIFIER
    # =========================================================

    print(
        "\n[3/5] Verifying research findings..."
    )

    verification = verify_evidence(
        research_question=research_question,
        evidence_items=evidence_items,
        use_gemini=True
    )

    if not isinstance(
        verification,
        dict
    ):
        verification = {}

    print(
        "Verification completed."
    )

    # =========================================================
    # DISPLAY VERIFICATION SUMMARY
    # =========================================================

    evidence_assessments = verification.get(
        "evidence_assessment",
        []
    )

    if not isinstance(
        evidence_assessments,
        list
    ):
        evidence_assessments = []

    supporting_count = sum(
        1
        for item in evidence_assessments
        if isinstance(item, dict)
        and item.get(
            "relationship"
        ) == "Supports"
    )

    partial_count = sum(
        1
        for item in evidence_assessments
        if isinstance(item, dict)
        and item.get(
            "relationship"
        ) == "Partially Supports"
    )

    contradicting_count = sum(
        1
        for item in evidence_assessments
        if isinstance(item, dict)
        and item.get(
            "relationship"
        ) == "Contradicts"
    )

    neutral_count = sum(
        1
        for item in evidence_assessments
        if isinstance(item, dict)
        and item.get(
            "relationship"
        ) == "Neutral"
    )

    print(
        "\nVerification Summary:"
    )

    print(
        f"  Supporting evidence: "
        f"{supporting_count}"
    )

    print(
        f"  Partially supporting evidence: "
        f"{partial_count}"
    )

    print(
        f"  Contradicting evidence: "
        f"{contradicting_count}"
    )

    print(
        f"  Neutral evidence: "
        f"{neutral_count}"
    )

    print(
        f"  Overall relevance: "
        f"{verification.get(
            'overall_relevance',
            'Unknown'
        )}"
    )

    print(
        f"  Verification confidence: "
        f"{verification.get(
            'verification_confidence',
            'Unknown'
        )}"
    )

    # =========================================================
    # PASS VERIFIED EVIDENCE TO DECISION AGENT
    # =========================================================

    verified_evidence = evidence_assessments

    print(
        f"\nVerified evidence passed to Decision Agent: "
        f"{len(verified_evidence)}"
    )

    # =========================================================
    # STEP 4: DECISION AGENT
    # =========================================================

    print(
        "\n[4/5] Making research decision..."
    )

    decision = make_decision(
        research_question=research_question,

        research_analysis={
            "key_findings": analysis.get(
                "key_findings",
                []
            ),

            "limitations": analysis.get(
                "limitations",
                []
            )
        },

        verification={
            "evidence_assessment": verified_evidence,

            "overall_relevance": verification.get(
                "overall_relevance",
                "Low"
            ),

            "verification_confidence": verification.get(
                "verification_confidence",
                "Low"
            ),

            "verification_status": verification.get(
                "verification_status",
                "unknown"
            ),

            "relevance_score": verification.get(
                "relevance_score",
                0.0
            ),

            "average_confidence": verification.get(
                "average_confidence",
                0.0
            ),

            "supporting_evidence": verification.get(
                "supporting_evidence",
                0
            ),

            "partially_supporting_evidence": verification.get(
                "partially_supporting_evidence",
                0
            ),

            "contradicting_evidence": verification.get(
                "contradicting_evidence",
                0
            ),

            "neutral_evidence": verification.get(
                "neutral_evidence",
                0
            ),

            "total_evidence": verification.get(
                "total_evidence",
                0
            ),

            "limitations": verification.get(
                "limitations",
                []
            )
        }
    )

    if not isinstance(
        decision,
        dict
    ):
        decision = {}

    print(
        "Decision completed."
    )

    # =========================================================
    # AUTHORITATIVE EVIDENCE QUALITY
    # =========================================================
    #
    # IMPORTANT:
    #
    # Evidence quality is calculated by decision.py.
    # The orchestrator must NOT recalculate it using a
    # different formula.
    #
    # This prevents conflicting values such as:
    #
    # decision.py       -> 82.45%
    # orchestrator.py   -> 95.00%
    #
    # The Decision Agent is the single authority.
    # =========================================================

    average_quality = decision.get(
        "average_evidence_quality",
        0.0
    )

    try:

        average_quality = float(
            average_quality
        )

    except (
        TypeError,
        ValueError
    ):

        average_quality = 0.0

    average_quality = max(
        0.0,
        min(
            100.0,
            average_quality
        )
    )

    # =========================================================
    # BUILD CLEAN DECISION REASONS
    # =========================================================

    decision_reasons = _build_clean_decision_reasons(
        verification=verification,

        evidence_items=evidence_items,

        average_quality=average_quality
    )

    # =========================================================
    # BUILD CLEAN LIMITATIONS
    # =========================================================

    limitations = _build_clean_limitations(
        verification=verification,

        analysis=analysis
    )

    # =========================================================
    # ADD CLEAN VALUES TO DECISION
    # =========================================================

    # ---------------------------------------------------------
    # SINGLE AUTHORITATIVE EVIDENCE QUALITY
    # ---------------------------------------------------------

    decision["average_evidence_quality"] = (
        average_quality
    )

    # ---------------------------------------------------------
    # KEEP NESTED EVIDENCE METRICS CONSISTENT
    # ---------------------------------------------------------

    if isinstance(
        decision.get(
            "evidence_metrics"
        ),
        dict
    ):

        decision["evidence_metrics"][
            "average_evidence_quality"
        ] = average_quality

    # ---------------------------------------------------------
    # CLEAN DECISION REASONS
    # ---------------------------------------------------------

    decision["decision_reasons"] = (
        decision_reasons
    )

    # Keep compatibility with the older
    # "reasons" field.

    decision["reasons"] = (
        decision_reasons
    )

    # ---------------------------------------------------------
    # CLEAN LIMITATIONS
    # ---------------------------------------------------------

    decision["limitations"] = (
        limitations
    )

    # =========================================================
    # STEP 5: FINAL RESULT
    # =========================================================

    print(
        "\n[5/5] Building final research result..."
    )

    result = {

        "research_question":
            research_question,

        "research_plan":
            plan,

        "retrieval": {

            "retrieval_questions":
                research_result.get(
                    "retrieval_questions",
                    []
                ),

            "evidence_items":
                evidence_items
        },

        "research_analysis":
            analysis,

        "verification":
            verification,

        "decision":
            decision,

        "report_quality": {

            "average_evidence_quality":
                average_quality,

            "supporting_evidence":
                supporting_count,

            "partially_supporting_evidence":
                partial_count,

            "contradicting_evidence":
                contradicting_count,

            "neutral_evidence":
                neutral_count,

            "total_verified_evidence":
                len(
                    evidence_assessments
                )
        },

        "final_report": {

            "decision_reasons":
                decision_reasons,

            "limitations":
                limitations
        }
    }

    print(
        "Research pipeline completed."
    )

    return result


# =============================================================
# MAIN EXECUTION
# =============================================================

if __name__ == "__main__":

    question = input(
        "\nEnter your research question: "
    )

    result = run_research_system(
        research_question=question,
        top_k=3
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "FINAL RESEARCH RESULT"
    )

    print(
        "=" * 60
    )

    print(
        json.dumps(
            result,
            indent=4
        )
    )