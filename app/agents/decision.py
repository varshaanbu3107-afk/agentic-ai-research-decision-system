import json

from app.utils.gemini_client import generate_content


# =============================================================
# HELPER FUNCTIONS
# =============================================================

def safe_float(value):
    """Safely convert a value to float."""

    if value is None:
        return None

    try:
        return float(value)

    except (TypeError, ValueError):
        return None


def normalize_evidence_id(evidence_id):
    """
    Normalize evidence IDs.

    E1   -> E001
    E01  -> E001
    E001 -> E001
    E12  -> E012
    """

    if not evidence_id:
        return evidence_id

    evidence_id = str(evidence_id).strip().upper()

    if evidence_id.startswith("E"):

        number = evidence_id[1:]

        if number.isdigit():
            return f"E{int(number):03d}"

    return evidence_id


# =============================================================
# NORMALIZE VERIFICATION ASSESSMENTS
# =============================================================

def normalize_assessments(verification):
    """Normalize verifier evidence assessments."""

    assessments = verification.get(
        "evidence_assessment",
        []
    )

    normalized = []

    valid_relationships = {
        "Supports",
        "Partially Supports",
        "Neutral",
        "Contradicts"
    }

    for item in assessments:

        if not isinstance(item, dict):
            continue

        evidence_id = normalize_evidence_id(
            item.get("evidence_id")
        )

        relationship = item.get(
            "relationship",
            "Neutral"
        )

        if relationship not in valid_relationships:
            relationship = "Neutral"

        normalized.append({
            "evidence_id": evidence_id,

            "relationship": relationship,

            "claim_supported": item.get(
                "claim_supported"
            ),

            "reasoning": item.get(
                "reasoning",
                ""
            ),

            "evidence_quality": item.get(
                "evidence_quality",
                item.get("source_quality_score")
            ),

            "similarity": item.get(
                "similarity",
                item.get("similarity_score")
            ),

            "domain_relevance": item.get(
                "domain_relevance"
            )
        })

    return normalized


# =============================================================
# CALCULATE EVIDENCE COUNTS
# =============================================================

def calculate_evidence_counts(assessments):

    supporting_count = sum(
        1
        for item in assessments
        if item["relationship"] == "Supports"
    )

    partial_count = sum(
        1
        for item in assessments
        if item["relationship"] == "Partially Supports"
    )

    neutral_count = sum(
        1
        for item in assessments
        if item["relationship"] == "Neutral"
    )

    contradicting_count = sum(
        1
        for item in assessments
        if item["relationship"] == "Contradicts"
    )

    return (
        supporting_count,
        partial_count,
        neutral_count,
        contradicting_count
    )


# =============================================================
# CALCULATE RELEVANCE SCORE
# =============================================================

def calculate_relevance_score(assessments):

    if not assessments:
        return 0.0

    total = len(assessments)

    relevant = sum(
        1
        for item in assessments
        if item.get("relationship")
        in {
            "Supports",
            "Partially Supports"
        }
    )

    return round(
        (relevant / total) * 100,
        2
    )


# =============================================================
# CALCULATE INDIVIDUAL EVIDENCE QUALITY
# =============================================================

def calculate_evidence_quality(assessment):

    quality = safe_float(
        assessment.get(
            "evidence_quality"
        )
    )

    similarity = safe_float(
        assessment.get(
            "similarity"
        )
    )

    domain_relevance = safe_float(
        assessment.get(
            "domain_relevance"
        )
    )

    # ---------------------------------------------------------
    # Defaults
    # ---------------------------------------------------------

    if quality is None:
        quality = 0.75

    if similarity is None:
        similarity = 0.75

    if domain_relevance is None:
        domain_relevance = 0.75

    # ---------------------------------------------------------
    # Convert percentages to 0-1
    # ---------------------------------------------------------

    if quality > 1:
        quality /= 100

    if similarity > 1:
        similarity /= 100

    if domain_relevance > 1:
        domain_relevance /= 100

    # ---------------------------------------------------------
    # Clamp
    # ---------------------------------------------------------

    quality = max(
        0.0,
        min(1.0, quality)
    )

    similarity = max(
        0.0,
        min(1.0, similarity)
    )

    domain_relevance = max(
        0.0,
        min(1.0, domain_relevance)
    )

    # ---------------------------------------------------------
    # Weighted quality
    # ---------------------------------------------------------

    return (
        quality * 0.40
        +
        similarity * 0.30
        +
        domain_relevance * 0.30
    )


# =============================================================
# RELATIONSHIP WEIGHT
# =============================================================

def get_relationship_weight(relationship):

    if relationship == "Supports":
        return 1.0

    if relationship == "Partially Supports":
        return 0.5

    if relationship == "Contradicts":
        return -1.0

    return 0.0


# =============================================================
# CALCULATE OVERALL EVIDENCE STRENGTH
# =============================================================

def calculate_evidence_strength(assessments):

    if not assessments:
        return 0.0

    weighted_total = 0.0
    quality_total = 0.0

    for assessment in assessments:

        quality_score = calculate_evidence_quality(
            assessment
        )

        relationship_weight = get_relationship_weight(
            assessment["relationship"]
        )

        weighted_total += (
            relationship_weight
            *
            quality_score
        )

        quality_total += quality_score

    if quality_total == 0:
        return 0.0

    score = (
        weighted_total
        /
        quality_total
    ) * 100

    score = max(
        0.0,
        min(100.0, score)
    )

    return round(
        score,
        2
    )


# =============================================================
# CALCULATE AVERAGE EVIDENCE QUALITY
# =============================================================

def calculate_average_quality(assessments):

    if not assessments:
        return 0.0

    total_quality = sum(
        calculate_evidence_quality(item)
        for item in assessments
    )

    return round(
        (
            total_quality
            /
            len(assessments)
        ) * 100,
        2
    )


# =============================================================
# DETERMINE DECISION CONFIDENCE
# =============================================================

def determine_decision_confidence(
    supporting_count,
    partial_count,
    neutral_count,
    contradicting_count,
    relevance_score,
    evidence_strength_score,
    verification_confidence
):
    """
    Determine confidence using deterministic evidence metrics.

    Rules:

    1. No evidence -> Low
    2. Contradicting evidence -> Low
    3. One strong supporting item with medium/high verification
       confidence -> High
    4. Multiple strong supporting items -> High
    5. Moderate support -> Medium
    6. Partial/weak support -> Low
    """

    total_evidence = (
        supporting_count
        +
        partial_count
        +
        neutral_count
        +
        contradicting_count
    )

    if total_evidence == 0:
        return "Low"

    # ---------------------------------------------------------
    # Contradictions always reduce confidence.
    # ---------------------------------------------------------

    if contradicting_count > 0:
        return "Low"

    # ---------------------------------------------------------
    # SINGLE / STRONG DIRECT SUPPORT
    #
    # IMPORTANT:
    # Medium verification confidence is allowed here because
    # the deterministic evidence metrics are authoritative.
    # ---------------------------------------------------------

    if (
        supporting_count >= 1
        and partial_count == 0
        and neutral_count == 0
        and relevance_score == 100
        and evidence_strength_score >= 75
        and verification_confidence in {
            "High",
            "Medium"
        }
    ):
        return "High"

    # ---------------------------------------------------------
    # MULTIPLE STRONG SUPPORT
    # ---------------------------------------------------------

    if (
        supporting_count >= 2
        and relevance_score >= 75
        and evidence_strength_score >= 75
        and verification_confidence in {
            "High",
            "Medium"
        }
    ):
        return "High"

    # ---------------------------------------------------------
    # MODERATE SUPPORT
    # ---------------------------------------------------------

    if (
        supporting_count >= 1
        and evidence_strength_score >= 40
    ):
        return "Medium"

    return "Low"


# =============================================================
# BUILD DECISION
# =============================================================

def build_decision(
    research_question,
    assessments,
    verification,
    research_analysis
):
    """
    Build the final deterministic decision.

    Deterministic logic controls:

    - evidence counts
    - relevance
    - evidence strength
    - evidence quality
    - confidence
    - status
    - decision
    - recommendation
    """

    (
        supporting_count,
        partial_count,
        neutral_count,
        contradicting_count
    ) = calculate_evidence_counts(
        assessments
    )

    total_evidence = len(
        assessments
    )

    # =========================================================
    # METRICS
    # =========================================================

    evidence_relevance_score = (
        calculate_relevance_score(
            assessments
        )
    )

    evidence_strength_score = (
        calculate_evidence_strength(
            assessments
        )
    )

    average_evidence_quality = (
        calculate_average_quality(
            assessments
        )
    )

    verification_confidence = verification.get(
        "verification_confidence",
        "Low"
    )

    # =========================================================
    # CONFIDENCE
    # =========================================================

    decision_confidence = determine_decision_confidence(
        supporting_count=supporting_count,
        partial_count=partial_count,
        neutral_count=neutral_count,
        contradicting_count=contradicting_count,
        relevance_score=evidence_relevance_score,
        evidence_strength_score=evidence_strength_score,
        verification_confidence=verification_confidence
    )

    # =========================================================
    # DETERMINE DECISION
    # =========================================================

    # ---------------------------------------------------------
    # NO EVIDENCE
    # ---------------------------------------------------------

    if total_evidence == 0:

        decision = (
            "The available evidence is insufficient to "
            "establish a conclusion for the research "
            f"question: '{research_question}'."
        )

        recommendation = (
            "Collect relevant evidence before making "
            "a decision."
        )

        decision_status = (
            "insufficient_evidence"
        )

    # ---------------------------------------------------------
    # CONTRADICTORY EVIDENCE
    # ---------------------------------------------------------

    elif contradicting_count > 0:

        if supporting_count > 0:

            decision = (
                "The available evidence contains "
                "conflicting findings for the research "
                f"question: '{research_question}'."
            )

            decision_status = (
                "conflicted_evidence"
            )

        else:

            decision = (
                "The available evidence is insufficient "
                "to support the research question and "
                "contains contradictory findings: "
                f"'{research_question}'."
            )

            decision_status = (
                "insufficient_evidence"
            )

        recommendation = (
            "Investigate the conflicting evidence and "
            "collect additional high-quality evidence "
            "before making a high-confidence decision."
        )

    # ---------------------------------------------------------
    # STRONG DIRECT SUPPORT
    # ---------------------------------------------------------

    elif (
        supporting_count >= 1
        and partial_count == 0
        and neutral_count == 0
        and evidence_relevance_score == 100
        and evidence_strength_score >= 75
    ):

        decision = (
            "The available evidence supports the "
            "research question: "
            f"'{research_question}'."
        )

        recommendation = (
            "The evidence provides direct support for "
            "the research question. The findings can be "
            "used as evidence-based guidance while "
            "considering the limitations."
        )

        decision_status = (
            "supported_evidence"
        )

    # ---------------------------------------------------------
    # MODERATE SUPPORT
    # ---------------------------------------------------------

    elif (
        supporting_count >= 2
        and evidence_relevance_score >= 50
        and evidence_strength_score >= 50
    ):

        decision = (
            "The available evidence supports the "
            "research question: "
            f"'{research_question}', but the evidence "
            "strength is moderate."
        )

        recommendation = (
            "Use the findings as evidence-based guidance "
            "and collect additional high-quality evidence "
            "to increase confidence."
        )

        decision_status = (
            "moderately_supported_evidence"
        )

    # ---------------------------------------------------------
    # PARTIAL SUPPORT
    # ---------------------------------------------------------

    elif (
        partial_count > 0
        or (
            supporting_count > 0
            and evidence_strength_score >= 25
        )
    ):

        decision = (
            "The available evidence partially supports "
            "the research question: "
            f"'{research_question}', but the evidence "
            "is not strong enough for a definitive "
            "conclusion."
        )

        recommendation = (
            "Use the findings cautiously and collect "
            "additional relevant evidence to improve "
            "confidence."
        )

        decision_status = (
            "partially_supported_evidence"
        )

    # ---------------------------------------------------------
    # INSUFFICIENT
    # ---------------------------------------------------------

    else:

        decision = (
            "The available evidence is insufficient "
            "to support the research question: "
            f"'{research_question}'."
        )

        recommendation = (
            "Collect additional evidence that directly "
            "addresses the research question."
        )

        decision_status = (
            "insufficient_evidence"
        )

    # =========================================================
    # KEY FINDINGS
    # =========================================================

    key_findings = research_analysis.get(
        "key_findings",
        []
    )

    if not isinstance(
        key_findings,
        list
    ):
        key_findings = []

    unique_findings = []

    for finding in key_findings:

        if not finding:
            continue

        finding = str(
            finding
        ).strip()

        if (
            finding
            and
            finding not in unique_findings
        ):
            unique_findings.append(
                finding
            )

    key_findings = unique_findings[:5]

    # =========================================================
    # LIMITATIONS
    # =========================================================

    limitations = []

    research_limitations = research_analysis.get(
        "limitations",
        []
    )

    verification_limitations = verification.get(
        "limitations",
        []
    )

    if isinstance(
        research_limitations,
        list
    ):
        limitations.extend(
            research_limitations
        )

    if isinstance(
        verification_limitations,
        list
    ):
        limitations.extend(
            verification_limitations
        )

    if neutral_count > 0:

        limitations.append(
            f"{neutral_count} evidence item(s) were "
            "classified as neutral because they did not "
            "provide sufficient direct support."
        )

    if partial_count > 0:

        limitations.append(
            f"{partial_count} evidence item(s) provided "
            "only partial or indirect support."
        )

    if contradicting_count > 0:

        limitations.append(
            f"{contradicting_count} evidence item(s) "
            "contained contradictory information."
        )

    unique_limitations = []

    for limitation in limitations:

        if not limitation:
            continue

        limitation = str(
            limitation
        ).strip()

        if (
            limitation
            and
            limitation not in unique_limitations
        ):
            unique_limitations.append(
                limitation
            )

    limitations = unique_limitations

    # =========================================================
    # DECISION REASONS
    # =========================================================

    decision_reasons = []

    if supporting_count > 0:

        decision_reasons.append(
            f"{supporting_count} of {total_evidence} "
            "verified evidence item(s) directly support "
            "the research question."
        )

    if partial_count > 0:

        decision_reasons.append(
            f"{partial_count} of {total_evidence} "
            "evidence item(s) provide partial or "
            "indirect support."
        )

    if contradicting_count > 0:

        decision_reasons.append(
            f"{contradicting_count} of {total_evidence} "
            "evidence item(s) contradict the relevant "
            "claim."
        )

    else:

        decision_reasons.append(
            "No retrieved evidence was classified as "
            "contradicting the research question."
        )

    if neutral_count > 0:

        decision_reasons.append(
            f"{neutral_count} evidence item(s) were "
            "classified as neutral because they did "
            "not provide sufficient direct support."
        )

    decision_reasons.append(
        "The selected evidence has an average "
        "source/evidence quality score of "
        f"{average_evidence_quality:.2f}%."
    )

    if evidence_relevance_score >= 75:

        relevance_label = "High"

    elif evidence_relevance_score >= 50:

        relevance_label = "Medium"

    else:

        relevance_label = "Low"

    decision_reasons.append(
        "Overall evidence relevance was assessed "
        f"as {relevance_label}."
    )

    # =========================================================
    # RISKS
    # =========================================================

    risks = []

    if contradicting_count > 0:

        risks.append(
            "Contradictory evidence may reduce "
            "confidence in the final conclusion."
        )

    if neutral_count > 0:

        risks.append(
            "Some retrieved evidence does not directly "
            "support the research question."
        )

    if partial_count > 0:

        risks.append(
            "Some evidence provides only partial or "
            "indirect support."
        )

    if verification.get(
        "verification_status"
    ) in {
        "local_fallback",
        "verified_local_fallback"
    }:

        risks.append(
            "The evidence verifier used deterministic "
            "rule-based verification rather than full "
            "LLM semantic reasoning."
        )

    if research_analysis.get(
        "analysis_status"
    ) == "local_fallback":

        risks.append(
            "The research analysis used deterministic "
            "local evidence extraction because Gemini "
            "was unavailable."
        )

    if not risks:

        risks.append(
            "The findings are limited to the evidence "
            "retrieved by the research pipeline."
        )

    unique_risks = []

    for risk in risks:

        if risk not in unique_risks:
            unique_risks.append(
                risk
            )

    risks = unique_risks

    # =========================================================
    # ALTERNATIVES
    # =========================================================

    alternatives = [
        "Collect additional evidence before acting.",
        "Use the findings as preliminary guidance "
        "rather than a definitive conclusion.",
        "Review the evidence manually before making "
        "a final decision."
    ]

    # =========================================================
    # EVIDENCE METRICS
    # =========================================================

    evidence_metrics = {

        "total_evidence":
            total_evidence,

        "relevant_evidence":
            supporting_count
            +
            partial_count,

        "supporting_evidence":
            supporting_count,

        "partially_supporting_evidence":
            partial_count,

        "neutral_evidence":
            neutral_count,

        "contradicting_evidence":
            contradicting_count,

        "evidence_strength_score":
            evidence_strength_score,

        "evidence_relevance_score":
            evidence_relevance_score,

        "average_evidence_quality":
            average_evidence_quality
    }

    # =========================================================
    # RETURN
    # =========================================================

    return {

        "research_question":
            research_question,

        "decision":
            decision,

        "confidence":
            decision_confidence,

        "status":
            decision_status,

        "decision_status":
            decision_status,

        "evidence_strength_score":
            evidence_strength_score,

        "evidence_relevance_score":
            evidence_relevance_score,

        "average_evidence_quality":
            average_evidence_quality,

        "evidence_metrics":
            evidence_metrics,

        "key_findings":
            key_findings,

        "recommendation":
            recommendation,

        "decision_reasons":
            decision_reasons,

        "reasons":
            decision_reasons,

        "risks":
            risks,

        "alternatives":
            alternatives,

        "limitations":
            limitations
    }


# =============================================================
# GEMINI DECISION AGENT
# =============================================================

def make_decision(
    research_question,
    research_analysis,
    verification
):
    """
    Final Decision Agent.

    Deterministic local metrics are authoritative.

    Gemini can provide qualitative reasoning only.
    """

    assessments = normalize_assessments(
        verification
    )

    # =========================================================
    # BUILD DETERMINISTIC DECISION FIRST
    # =========================================================

    local_decision = build_decision(
        research_question,
        assessments,
        verification,
        research_analysis
    )

    # =========================================================
    # GEMINI PROMPT
    # =========================================================

    prompt = f"""
You are a Decision Agent in an Agentic AI
Research and Decision-Making System.

Research Question:
{research_question}

Research Analysis:
{json.dumps(research_analysis, indent=2)}

Verification:
{json.dumps(verification, indent=2)}

Deterministic Evidence Decision:
{json.dumps(local_decision, indent=2)}

The deterministic decision is authoritative.

Your task is ONLY to provide qualitative reasoning.

Do NOT change:

- evidence counts
- evidence strength score
- evidence relevance score
- average evidence quality
- confidence
- status
- decision status
- recommendation

Do NOT invent facts.

Return ONLY valid JSON.

Use this structure:

{{
    "research_question": "{research_question}",
    "decision_reasoning": [],
    "qualitative_summary": "string"
}}

Rules:

1. Consider all verified evidence.
2. Supports = positive evidence.
3. Partially Supports = partial positive evidence.
4. Neutral = zero support.
5. Contradicts = negative evidence.
6. Do not invent information.
7. Do not contradict the deterministic decision.
8. Keep reasoning concise.
9. Return only valid JSON.
"""

    # =========================================================
    # TRY GEMINI
    # =========================================================

    try:

        print(
            "\nGenerating qualitative decision reasoning..."
        )

        text = generate_content(
            prompt
        )

    except Exception as error:

        print(
            "\nGemini Decision Agent unavailable."
        )

        print(
            "Using deterministic local decision engine."
        )

        print(
            f"Reason: {error}"
        )

        return local_decision

    # =========================================================
    # EMPTY RESPONSE
    # =========================================================

    if not text or not text.strip():

        print(
            "\nGemini Decision Agent returned "
            "an empty response."
        )

        print(
            "Using deterministic local decision engine."
        )

        return local_decision

    # =========================================================
    # PARSE GEMINI RESPONSE
    # =========================================================

    try:

        text = text.strip()

        if text.startswith(
            "```json"
        ):

            text = text[7:]

        elif text.startswith(
            "```"
        ):

            text = text[3:]

        if text.endswith(
            "```"
        ):

            text = text[:-3]

        text = text.strip()

        start = text.find(
            "{"
        )

        end = text.rfind(
            "}"
        )

        if (
            start == -1
            or
            end == -1
            or
            end <= start
        ):

            raise ValueError(
                "No valid JSON object found."
            )

        json_text = text[
            start:end + 1
        ]

        result = json.loads(
            json_text
        )

        if not isinstance(
            result,
            dict
        ):

            raise ValueError(
                "Gemini response is not a JSON object."
            )

        qualitative_reasoning = result.get(
            "decision_reasoning",
            []
        )

        if not isinstance(
            qualitative_reasoning,
            list
        ):

            qualitative_reasoning = []

        qualitative_reasoning = [
            str(item).strip()
            for item in qualitative_reasoning
            if str(item).strip()
        ]

        qualitative_summary = result.get(
            "qualitative_summary",
            ""
        )

        if not isinstance(
            qualitative_summary,
            str
        ):

            qualitative_summary = ""

        qualitative_summary = (
            qualitative_summary.strip()
        )

        # =====================================================
        # MERGE QUALITATIVE GEMINI OUTPUT
        # WITH DETERMINISTIC RESULT
        # =====================================================

        final_result = dict(
            local_decision
        )

        final_result[
            "qualitative_reasoning"
        ] = qualitative_reasoning

        if qualitative_summary:

            final_result[
                "qualitative_summary"
            ] = qualitative_summary

        else:

            final_result[
                "qualitative_summary"
            ] = local_decision[
                "decision"
            ]

        # =====================================================
        # DETERMINISTIC VALUES REMAIN AUTHORITATIVE
        # =====================================================

        final_result[
            "decision_engine"
        ] = "deterministic_local_metrics"

        final_result[
            "decision_status"
        ] = local_decision[
            "decision_status"
        ]

        final_result[
            "status"
        ] = local_decision[
            "status"
        ]

        final_result[
            "confidence"
        ] = local_decision[
            "confidence"
        ]

        final_result[
            "decision"
        ] = local_decision[
            "decision"
        ]

        final_result[
            "recommendation"
        ] = local_decision[
            "recommendation"
        ]

        final_result[
            "evidence_strength_score"
        ] = local_decision[
            "evidence_strength_score"
        ]

        final_result[
            "evidence_relevance_score"
        ] = local_decision[
            "evidence_relevance_score"
        ]

        final_result[
            "average_evidence_quality"
        ] = local_decision[
            "average_evidence_quality"
        ]

        final_result[
            "evidence_metrics"
        ] = local_decision[
            "evidence_metrics"
        ]

        final_result[
            "decision_reasons"
        ] = local_decision[
            "decision_reasons"
        ]

        final_result[
            "reasons"
        ] = local_decision[
            "decision_reasons"
        ]

        final_result[
            "risks"
        ] = local_decision[
            "risks"
        ]

        final_result[
            "alternatives"
        ] = local_decision[
            "alternatives"
        ]

        final_result[
            "limitations"
        ] = local_decision[
            "limitations"
        ]

        return final_result

    except Exception as error:

        print(
            "\nGemini Decision Agent returned "
            "invalid or incomplete JSON."
        )

        print(
            f"Reason: {error}"
        )

        print(
            "Using deterministic local decision engine."
        )

        return local_decision


# =============================================================
# MANUAL TEST
# =============================================================

if __name__ == "__main__":

    question = (
        "How can AI improve customer support efficiency?"
    )

    research_analysis = {

        "key_findings": [

            "AI chatbots can automate routine "
            "customer support tasks.",

            "AI can provide faster responses "
            "to customer requests."
        ],

        "limitations": [

            "The evidence set is limited."
        ]
    }

    verification = {

        "evidence_assessment": [

            {
                "evidence_id": "E001",

                "relationship": "Supports",

                "claim_supported": (
                    "AI chatbots can automate routine "
                    "customer support tasks and provide "
                    "faster responses."
                ),

                "reasoning": (
                    "The evidence directly supports "
                    "AI-enabled efficiency."
                ),

                "evidence_quality": 0.95,

                "similarity": 0.82,

                "domain_relevance": 0.90
            },

            {
                "evidence_id": "E002",

                "relationship": "Neutral",

                "claim_supported": None,

                "reasoning": (
                    "The evidence does not directly "
                    "address efficiency."
                ),

                "evidence_quality": 0.90,

                "similarity": 0.40,

                "domain_relevance": 0.40
            },

            {
                "evidence_id": "E003",

                "relationship": "Partially Supports",

                "claim_supported": (
                    "Human agents are required for "
                    "complex customer service situations."
                ),

                "reasoning": (
                    "The evidence provides partial "
                    "context about AI and customer service."
                ),

                "evidence_quality": 0.85,

                "similarity": 0.55,

                "domain_relevance": 0.70
            }
        ],

        "overall_relevance": "Medium",

        "verification_confidence": "Medium",

        "verification_status": "local_fallback",

        "limitations": []
    }

    result = make_decision(
        research_question=question,
        research_analysis=research_analysis,
        verification=verification
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "DECISION AGENT RESULT"
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

