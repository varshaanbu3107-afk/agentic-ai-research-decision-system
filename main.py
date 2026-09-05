from app.core.orchestrator import run_research_system


# =============================================================
# DISPLAY HELPERS
# =============================================================

def print_header(title):
    """Print a major report header."""
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def print_section(title):
    """Print a report section."""
    print("\n" + "-" * 64)
    print(title)
    print("-" * 64)


def print_list(items):
    """Print a numbered list."""
    if not items:
        print("No information available.")
        return

    for index, item in enumerate(items, start=1):
        print(f"{index}. {item}")


def print_metric(label, value, width=28):
    """Print a formatted metric."""
    print(f"{label:<{width}}: {value}")


# =============================================================
# EVIDENCE DISPLAY
# =============================================================

def display_evidence_summary(result):
    """Display detailed evidence verification results."""

    verification = result.get(
        "verification",
        {}
    )

    evidence_assessment = verification.get(
        "evidence_assessment",
        []
    )

    if not evidence_assessment:
        return

    print_section(
        "EVIDENCE VERIFICATION"
    )

    print(
        f"{'Evidence ID':<14}"
        f"{'Relationship':<24}"
        f"Claim / Reasoning"
    )

    print("-" * 64)

    for item in evidence_assessment:

        evidence_id = str(
            item.get(
                "evidence_id",
                "Unknown"
            )
        )

        relationship = str(
            item.get(
                "relationship",
                "Unknown"
            )
        )

        claim = item.get(
            "claim_supported"
        )

        reasoning = item.get(
            "reasoning",
            ""
        )

        if claim:
            description = claim
        else:
            description = reasoning

        # Keep terminal output readable.
        description = str(
            description
        ).replace("\n", " ")

        if len(description) > 80:
            description = (
                description[:77]
                + "..."
            )

        print(
            f"{evidence_id:<14}"
            f"{relationship:<24}"
            f"{description}"
        )


# =============================================================
# FINAL REPORT
# =============================================================

def display_final_report(result):

    print_header(
        "FINAL RESEARCH REPORT"
    )

    # =========================================================
    # RESEARCH QUESTION
    # =========================================================

    research_question = result.get(
        "research_question",
        "Not available"
    )

    print_section(
        "RESEARCH QUESTION"
    )

    print(
        research_question
    )

    # =========================================================
    # DECISION
    # =========================================================

    decision = result.get(
        "decision",
        {}
    )

    print_section(
        "DECISION"
    )

    decision_text = decision.get(
        "decision",
        "Not available"
    )

    confidence = decision.get(
        "confidence",
        "Not available"
    )

    # Current Decision Agent uses "status".
    decision_status = decision.get(
        "status",
        "Not available"
    )

    print(
        f"Decision    : {decision_text}"
    )

    print(
        f"Confidence  : {confidence}"
    )

    print(
        f"Status      : {decision_status}"
    )

    # =========================================================
    # EVIDENCE METRICS
    # =========================================================

    evidence_metrics = decision.get(
        "evidence_metrics",
        {}
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    #
    # Current Decision Agent stores these three scores at the
    # TOP LEVEL of the decision object.
    # ---------------------------------------------------------

    evidence_strength = decision.get(
        "evidence_strength_score",
        0
    )

    evidence_relevance = decision.get(
        "evidence_relevance_score",
        0
    )

    average_evidence_quality = decision.get(
        "average_evidence_quality",
        0
    )

    # ---------------------------------------------------------
    # Evidence counts are stored inside evidence_metrics.
    # ---------------------------------------------------------

    total_evidence = evidence_metrics.get(
        "total_evidence",
        0
    )

    supporting_evidence = evidence_metrics.get(
        "supporting_evidence",
        0
    )

    partial_evidence = evidence_metrics.get(
        "partially_supporting_evidence",
        0
    )

    contradicting_evidence = evidence_metrics.get(
        "contradicting_evidence",
        0
    )

    neutral_evidence = evidence_metrics.get(
        "neutral_evidence",
        0
    )

    # ---------------------------------------------------------
    # Relevant evidence:
    #
    # Supports + Partially Supports
    # ---------------------------------------------------------

    relevant_evidence = (
        supporting_evidence
        +
        partial_evidence
    )

    print_section(
        "EVIDENCE QUALITY METRICS"
    )

    print_metric(
        "Evidence Strength",
        f"{evidence_strength}/100"
    )

    print_metric(
        "Evidence Relevance",
        f"{evidence_relevance}%"
    )

    print_metric(
        "Average Evidence Quality",
        f"{average_evidence_quality}%"
    )

    print_metric(
        "Relevant Evidence",
        f"{relevant_evidence}/{total_evidence}"
    )

    # =========================================================
    # RESEARCH ANALYSIS
    # =========================================================

    analysis = result.get(
        "research_analysis",
        {}
    )

    # =========================================================
    # KEY FINDINGS
    # =========================================================

    print_section(
        "KEY FINDINGS"
    )

    print_list(
        analysis.get(
            "key_findings",
            []
        )
    )

    # =========================================================
    # EVIDENCE SUMMARY
    # =========================================================

    print_section(
        "EVIDENCE SUMMARY"
    )

    print_metric(
        "Total Evidence",
        total_evidence
    )

    print_metric(
        "Relevant Evidence",
        relevant_evidence
    )

    print_metric(
        "Supporting Evidence",
        supporting_evidence
    )

    print_metric(
        "Partially Supporting",
        partial_evidence
    )

    print_metric(
        "Contradicting Evidence",
        contradicting_evidence
    )

    print_metric(
        "Neutral Evidence",
        neutral_evidence
    )

    print_metric(
        "Evidence Strength Score",
        f"{evidence_strength}/100"
    )

    print_metric(
        "Evidence Relevance Score",
        f"{evidence_relevance}%"
    )

    print_metric(
        "Average Evidence Quality",
        f"{average_evidence_quality}%"
    )

    # =========================================================
    # VERIFICATION SUMMARY
    # =========================================================

    verification = result.get(
        "verification",
        {}
    )

    if verification:

        print_section(
            "VERIFICATION SUMMARY"
        )

        print_metric(
            "Overall Relevance",
            verification.get(
                "overall_relevance",
                "Not available"
            )
        )

        print_metric(
            "Verification Confidence",
            verification.get(
                "verification_confidence",
                "Not available"
            )
        )

        print_metric(
            "Verification Status",
            verification.get(
                "verification_status",
                "Not available"
            )
        )

        print_metric(
            "Supporting Evidence",
            len([
                item
                for item in verification.get(
                    "evidence_assessment",
                    []
                )
                if item.get(
                    "relationship"
                ) == "Supports"
            ])
        )

        print_metric(
            "Partially Supporting",
            len([
                item
                for item in verification.get(
                    "evidence_assessment",
                    []
                )
                if item.get(
                    "relationship"
                ) == "Partially Supports"
            ])
        )

        print_metric(
            "Contradicting Evidence",
            len([
                item
                for item in verification.get(
                    "evidence_assessment",
                    []
                )
                if item.get(
                    "relationship"
                ) == "Contradicts"
            ])
        )

        print_metric(
            "Neutral Evidence",
            len([
                item
                for item in verification.get(
                    "evidence_assessment",
                    []
                )
                if item.get(
                    "relationship"
                ) == "Neutral"
            ])
        )

    # =========================================================
    # DETAILED EVIDENCE
    # =========================================================

    display_evidence_summary(
        result
    )

    # =========================================================
    # LIMITATIONS
    # =========================================================

    print_section(
        "LIMITATIONS"
    )

    limitations = analysis.get(
        "limitations",
        []
    )

    verification_limitations = verification.get(
        "limitations",
        []
    )

    combined_limitations = []

    for limitation in (
        limitations
        + verification_limitations
    ):

        if limitation not in combined_limitations:

            combined_limitations.append(
                limitation
            )

    # ---------------------------------------------------------
    # Also include Decision Agent limitations.
    # ---------------------------------------------------------

    decision_limitations = decision.get(
        "limitations",
        []
    )

    for limitation in decision_limitations:

        if limitation not in combined_limitations:

            combined_limitations.append(
                limitation
            )

    print_list(
        combined_limitations
    )

    # =========================================================
    # RECOMMENDATION
    # =========================================================

    print_section(
        "RECOMMENDATION"
    )

    print(
        decision.get(
            "recommendation",
            "No recommendation available."
        )
    )

    # =========================================================
    # DECISION REASONS
    # =========================================================

    # Current Decision Agent uses "decision_reasons".
    decision_reasons = decision.get(
        "decision_reasons",
        []
    )

    if decision_reasons:

        print_section(
            "DECISION REASONS"
        )

        print_list(
            decision_reasons
        )

    # =========================================================
    # RISKS
    # =========================================================

    print_section(
        "RISKS"
    )

    print_list(
        decision.get(
            "risks",
            []
        )
    )

    # =========================================================
    # ALTERNATIVES
    # =========================================================

    print_section(
        "ALTERNATIVES"
    )

    print_list(
        decision.get(
            "alternatives",
            []
        )
    )

    # =========================================================
    # FINAL STATUS
    # =========================================================

    print_header(
        "RESEARCH PIPELINE COMPLETED"
    )


# =============================================================
# MAIN APPLICATION
# =============================================================

def main():

    print_header(
        "AGENTIC AI RESEARCH & DECISION SYSTEM"
    )

    research_question = input(
        "\nEnter your research question: "
    ).strip()

    # =========================================================
    # INPUT VALIDATION
    # =========================================================

    if not research_question:

        print(
            "\nResearch question cannot be empty."
        )

        return

    # =========================================================
    # RUN RESEARCH PIPELINE
    # =========================================================

    try:

        result = run_research_system(
            research_question=research_question,
            top_k=3
        )

        # =====================================================
        # DISPLAY FINAL RESULT
        # =====================================================

        display_final_report(
            result
        )

    except RuntimeError as error:

        print_header(
            "SYSTEM ERROR"
        )

        print(
            f"\n{error}"
        )

        print(
            "\nThe research system could not "
            "complete the request."
        )

        print(
            "Check the Gemini API configuration, "
            "quota, network connection, or try again later."
        )

    except Exception as error:

        print_header(
            "UNEXPECTED SYSTEM ERROR"
        )

        print(
            f"\nError: {error}"
        )

        print(
            "\nThe application encountered an "
            "unexpected error."
        )


# =============================================================
# APPLICATION ENTRY POINT
# =============================================================

if __name__ == "__main__":

    main()

