"""
Evidence Verification Agent

This module verifies whether retrieved evidence:
    - Supports
    - Partially Supports
    - Contradicts
    - Is Neutral

The verifier supports:
1. Gemini semantic verification when available.
2. Deterministic local verification when Gemini is unavailable.

The local verifier is designed to be:
    - domain-aware
    - research-question-aware
    - evidence-grounded
    - reusable across different research domains

Examples:
    How can AI improve healthcare?
    How can AI improve customer support?
    How can AI improve education?
    What are the benefits of AI in finance?
"""

import json
import re
from typing import Any, Dict, List, Optional


# =============================================================
# DOMAIN / CONCEPT KEYWORDS
# =============================================================

AI_TERMS = {
    "artificial intelligence",
    "ai",
    "machine learning",
    "ml",
    "deep learning",
    "chatbot",
    "chatbots",
    "conversational ai",
    "natural language processing",
    "nlp",
    "generative ai",
    "llm",
    "large language model",
    "intelligent system",
    "intelligent systems",
    "automation",
    "automated",
    "algorithm",
    "algorithms",
}


HEALTHCARE_TERMS = {
    "healthcare",
    "health care",
    "health-care",
    "patient",
    "patients",
    "medical",
    "medicine",
    "clinical",
    "clinician",
    "clinicians",
    "doctor",
    "doctors",
    "physician",
    "physicians",
    "hospital",
    "hospitals",
    "health system",
    "health systems",
    "diagnosis",
    "diagnoses",
    "diagnostic",
    "treatment",
    "treatments",
    "therapy",
    "therapies",
    "disease",
    "diseases",
    "public health",
    "health policy",
    "health policies",
    "health worker",
    "health workers",
    "health-care workers",
    "medical professional",
    "medical professionals",
    "patient care",
    "community health",
}


CUSTOMER_SUPPORT_TERMS = {
    "customer service",
    "customer services",
    "customer support",
    "customer",
    "customers",
    "client",
    "clients",
    "helpdesk",
    "help desk",
    "call center",
    "call centre",
    "contact center",
    "contact centre",
    "service",
    "support",
    "user experience",
    "customer experience",
    "customer satisfaction",
}


EDUCATION_TERMS = {
    "education",
    "educational",
    "student",
    "students",
    "teacher",
    "teachers",
    "school",
    "schools",
    "university",
    "universities",
    "learning",
    "teaching",
    "classroom",
    "academic",
    "curriculum",
    "assessment",
}


FINANCE_TERMS = {
    "finance",
    "financial",
    "bank",
    "banking",
    "banks",
    "investment",
    "investments",
    "investor",
    "investors",
    "loan",
    "loans",
    "credit",
    "fraud",
    "financial services",
    "insurance",
    "risk management",
}


MANUFACTURING_TERMS = {
    "manufacturing",
    "factory",
    "factories",
    "production",
    "industrial",
    "industry",
    "machine",
    "machines",
    "equipment",
    "maintenance",
    "predictive maintenance",
    "quality control",
    "supply chain",
}


CYBERSECURITY_TERMS = {
    "cybersecurity",
    "cyber security",
    "security",
    "network security",
    "information security",
    "threat",
    "threats",
    "attack",
    "attacks",
    "malware",
    "phishing",
    "vulnerability",
    "vulnerabilities",
    "intrusion",
    "data protection",
}


EFFICIENCY_TERMS = {
    "efficiency",
    "efficient",
    "productivity",
    "performance",
    "time",
    "effort",
    "cost",
    "costs",
    "saving",
    "savings",
    "faster",
    "speed",
    "quick",
    "quickly",
    "automation",
    "automated",
    "scalability",
    "scale",
    "workload",
    "resolution",
    "resolve",
    "improve",
    "improves",
    "improvement",
    "improvements",
    "benefit",
    "benefits",
    "outcome",
    "outcomes",
    "quality",
    "access",
    "accuracy",
    "accurate",
}


NEGATIVE_TERMS = {
    "inefficient",
    "inefficiency",
    "slow",
    "slower",
    "increase cost",
    "increased cost",
    "higher cost",
    "expensive",
    "failure",
    "failures",
    "error",
    "errors",
    "risk",
    "risks",
    "limitation",
    "limitations",
    "negative",
    "harm",
    "harms",
    "worse",
    "reduced satisfaction",
    "decreased satisfaction",
    "uncertainty",
    "uncertain",
    "challenge",
    "challenges",
    "problem",
    "problems",
    "barrier",
    "barriers",
    "bias",
    "biased",
    "discrimination",
    "privacy",
    "privacy risk",
    "security risk",
}


DIRECT_OUTCOME_TERMS = {
    "improve",
    "improves",
    "improved",
    "improvement",
    "improvements",
    "increase",
    "increased",
    "increases",
    "effective",
    "effectiveness",
    "efficient",
    "efficiency",
    "reduce",
    "reduced",
    "reduces",
    "reduction",
    "saving",
    "savings",
    "faster",
    "quickly",
    "performance",
    "productivity",
    "time and effort",
    "resolve",
    "resolution",
    "automation",
    "automated",
    "benefit",
    "benefits",
    "support",
    "augment",
    "enable",
    "enables",
    "provide",
    "provides",
    "optimize",
    "optimize treatment",
    "accuracy",
    "accurate",
    "access",
    "bridge gaps",
}


# =============================================================
# DOMAIN GROUPS
# =============================================================

DOMAIN_GROUPS = {
    "healthcare": HEALTHCARE_TERMS,
    "customer_support": CUSTOMER_SUPPORT_TERMS,
    "education": EDUCATION_TERMS,
    "finance": FINANCE_TERMS,
    "manufacturing": MANUFACTURING_TERMS,
    "cybersecurity": CYBERSECURITY_TERMS,
}


# =============================================================
# TEXT HELPERS
# =============================================================

def normalize_text(text: Any) -> str:
    """
    Normalize text for deterministic analysis.
    """

    if text is None:
        return ""

    text = str(text).lower()

    # Fix common PDF ligatures.
    text = text.replace("\ufb01", "fi")
    text = text.replace("\ufb02", "fl")
    text = text.replace("\ufb00", "ff")
    text = text.replace("\ufb03", "ffi")
    text = text.replace("\ufb04", "ffl")

    # Normalize whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def contains_any(
    text: str,
    terms: set
) -> List[str]:
    """
    Return all terms found in text.
    """

    found = []

    for term in terms:

        if term in text:

            found.append(term)

    return found


def calculate_keyword_coverage(
    text: str,
    term_groups: List[set]
) -> float:
    """
    Calculate how many conceptual groups
    are represented in the evidence.

    Returns:
        float from 0.0 to 1.0
    """

    if not term_groups:

        return 0.0

    matched_groups = 0

    for group in term_groups:

        if contains_any(
            text,
            group
        ):

            matched_groups += 1

    return (
        matched_groups
        / len(term_groups)
    )


# =============================================================
# QUESTION ANALYSIS
# =============================================================

def analyze_research_question(
    research_question: str
) -> Dict[str, Any]:
    """
    Extract major concepts from the research question.

    Unlike the previous version, this function dynamically
    identifies the domain instead of assuming customer support.
    """

    question = normalize_text(
        research_question
    )

    ai_matches = contains_any(
        question,
        AI_TERMS
    )

    efficiency_matches = contains_any(
        question,
        EFFICIENCY_TERMS
    )

    detected_domains = {}

    for domain_name, domain_terms in DOMAIN_GROUPS.items():

        matches = contains_any(
            question,
            domain_terms
        )

        if matches:

            detected_domains[
                domain_name
            ] = matches

    # ---------------------------------------------------------
    # Generic research terms
    # ---------------------------------------------------------

    research_intent_terms = {
        "improve",
        "improvement",
        "benefit",
        "benefits",
        "impact",
        "effect",
        "effects",
        "advantage",
        "advantages",
        "potential",
        "gain",
        "gains",
        "outcome",
        "outcomes",
        "efficiency",
        "performance",
        "cost",
        "risk",
        "risks",
        "challenge",
        "challenges",
        "implementation",
        "application",
        "applications",
    }

    intent_matches = contains_any(
        question,
        research_intent_terms
    )

    return {
        "normalized_question": question,

        "ai_terms": ai_matches,

        "detected_domains": detected_domains,

        "efficiency_terms": efficiency_matches,

        "intent_terms": intent_matches,

        "has_ai_concept": bool(
            ai_matches
        ),

        "has_domain_concept": bool(
            detected_domains
        ),

        "has_efficiency_concept": bool(
            efficiency_matches
        ),
    }


# =============================================================
# QUESTION DOMAIN TERMS
# =============================================================

def get_question_domain_terms(
    question_info: Dict[str, Any]
) -> set:
    """
    Return the domain vocabulary relevant to the question.
    """

    detected_domains = question_info.get(
        "detected_domains",
        {}
    )

    domain_terms = set()

    for domain_name in detected_domains:

        domain_terms.update(
            DOMAIN_GROUPS.get(
                domain_name,
                set()
            )
        )

    return domain_terms


# =============================================================
# EVIDENCE SCORING
# =============================================================

def score_evidence(
    research_question: str,
    evidence_content: str
) -> Dict[str, Any]:
    """
    Score one evidence item against the research question.

    The scoring is question-aware rather than hard-coded
    to customer support.
    """

    question_info = analyze_research_question(
        research_question
    )

    evidence = normalize_text(
        evidence_content
    )

    # ---------------------------------------------------------
    # Question concepts
    # ---------------------------------------------------------

    question_ai = question_info.get(
        "has_ai_concept",
        False
    )

    question_domains = (
        question_info.get(
            "detected_domains",
            {}
        )
    )

    question_efficiency = (
        question_info.get(
            "has_efficiency_concept",
            False
        )
    )

    domain_terms = get_question_domain_terms(
        question_info
    )

    # ---------------------------------------------------------
    # Evidence concept matches
    # ---------------------------------------------------------

    ai_matches = contains_any(
        evidence,
        AI_TERMS
    )

    domain_matches = contains_any(
        evidence,
        domain_terms
    )

    efficiency_matches = contains_any(
        evidence,
        EFFICIENCY_TERMS
    )

    negative_matches = contains_any(
        evidence,
        NEGATIVE_TERMS
    )

    direct_matches = contains_any(
        evidence,
        DIRECT_OUTCOME_TERMS
    )

    # ---------------------------------------------------------
    # Domain-specific matching
    # ---------------------------------------------------------

    matched_domains = {}

    for domain_name in question_domains:

        matches = contains_any(
            evidence,
            DOMAIN_GROUPS.get(
                domain_name,
                set()
            )
        )

        if matches:

            matched_domains[
                domain_name
            ] = matches

    # ---------------------------------------------------------
    # AI score
    # ---------------------------------------------------------

    if question_ai:

        ai_score = (
            1.0
            if ai_matches
            else 0.0
        )

    else:

        # If the question is not explicitly about AI,
        # AI presence should not dominate the result.
        ai_score = 0.0

    # ---------------------------------------------------------
    # Domain score
    # ---------------------------------------------------------

    if question_domains:

        domain_score = min(
            len(domain_matches) / 2.0,
            1.0
        )

    else:

        # No known domain.
        domain_score = 0.0

    # ---------------------------------------------------------
    # Efficiency / outcome score
    # ---------------------------------------------------------

    if question_efficiency:

        efficiency_score = (
            1.0
            if efficiency_matches
            else 0.0
        )

    else:

        # For questions such as "How can AI improve healthcare?",
        # outcome language is still relevant.
        efficiency_score = min(
            len(direct_matches) / 3.0,
            1.0
        )

    # ---------------------------------------------------------
    # Direct relationship score
    # ---------------------------------------------------------

    direct_score = min(
        len(direct_matches) / 3.0,
        1.0
    )

    # ---------------------------------------------------------
    # Concept coverage
    # ---------------------------------------------------------

    coverage_groups = []

    if question_ai:

        coverage_groups.append(
            AI_TERMS
        )

    if question_domains:

        coverage_groups.append(
            domain_terms
        )

    if question_efficiency:

        coverage_groups.append(
            EFFICIENCY_TERMS
        )

    if not coverage_groups:

        coverage_groups.append(
            DIRECT_OUTCOME_TERMS
        )

    concept_coverage = (
        calculate_keyword_coverage(
            evidence,
            coverage_groups
        )
    )

    # ---------------------------------------------------------
    # Relationship score
    # ---------------------------------------------------------

    if question_domains and question_ai:

        positive_score = (
            0.40 * ai_score
            + 0.35 * domain_score
            + 0.15 * direct_score
            + 0.10 * efficiency_score
        )

    elif question_ai:

        positive_score = (
            0.50 * ai_score
            + 0.30 * direct_score
            + 0.20 * efficiency_score
        )

    elif question_domains:

        positive_score = (
            0.50 * domain_score
            + 0.30 * direct_score
            + 0.20 * efficiency_score
        )

    else:

        positive_score = (
            0.60 * direct_score
            + 0.40 * efficiency_score
        )

    # ---------------------------------------------------------
    # Negative score
    # ---------------------------------------------------------

    negative_score = min(
        len(negative_matches) / 4.0,
        1.0
    )

    # ---------------------------------------------------------
    # Relationship classification
    # ---------------------------------------------------------

    # Strong direct evidence.
    if (
        positive_score >= 0.65
        and concept_coverage >= 0.50
    ):

        relationship = "Supports"

    # Useful but incomplete evidence.
    elif (
        positive_score >= 0.35
        and concept_coverage >= 0.33
    ):

        relationship = "Partially Supports"

    # Strong negative evidence.
    elif (
        negative_score >= 0.75
        and positive_score < 0.35
    ):

        relationship = "Contradicts"

    else:

        relationship = "Neutral"

    # ---------------------------------------------------------
    # Confidence
    # ---------------------------------------------------------

    confidence = round(
        max(
            positive_score,
            negative_score
        ),
        3
    )

    # ---------------------------------------------------------
    # Reasoning
    # ---------------------------------------------------------

    reasoning_parts = []

    if ai_matches and question_ai:

        reasoning_parts.append(
            "AI-related concepts are present."
        )

    if matched_domains:

        domain_names = ", ".join(
            matched_domains.keys()
        )

        reasoning_parts.append(
            f"Question-relevant domain concepts "
            f"are present ({domain_names})."
        )

    if efficiency_matches:

        reasoning_parts.append(
            "Efficiency or outcome-related concepts "
            "are present."
        )

    if direct_matches:

        reasoning_parts.append(
            "Direct outcome language was detected."
        )

    if negative_matches:

        reasoning_parts.append(
            "The evidence also contains limitation, "
            "risk, or negative-outcome language."
        )

    if not reasoning_parts:

        reasoning_parts.append(
            "The evidence does not contain enough "
            "question-relevant concepts to establish "
            "a meaningful relationship."
        )

    reasoning = " ".join(
        reasoning_parts
    )

    return {
        "relationship": relationship,

        "confidence": confidence,

        "positive_score": round(
            positive_score,
            3
        ),

        "negative_score": round(
            negative_score,
            3
        ),

        "concept_coverage": round(
            concept_coverage,
            3
        ),

        "ai_matches": ai_matches,

        "domain_matches": domain_matches,

        "matched_domains": matched_domains,

        "efficiency_matches": efficiency_matches,

        "negative_matches": negative_matches,

        "direct_matches": direct_matches,

        "reasoning": reasoning,

        "question_analysis": question_info,
    }


# =============================================================
# EVIDENCE EXTRACTION
# =============================================================

def extract_evidence_content(
    evidence: Dict[str, Any]
) -> str:
    """
    Extract evidence text from supported
    structured evidence formats.
    """

    if not isinstance(
        evidence,
        dict
    ):

        return ""

    for key in (
        "content",
        "text",
        "claim",
        "claim_supported",
        "evidence",
    ):

        value = evidence.get(
            key
        )

        if value:

            return str(value)

    return ""


# =============================================================
# LOCAL VERIFICATION
# =============================================================

def local_verify_evidence(
    research_question: str,
    evidence_items: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Deterministically verify all evidence items.
    """

    assessments = []

    for index, evidence in enumerate(
        evidence_items,
        start=1
    ):

        if not isinstance(
            evidence,
            dict
        ):

            continue

        evidence_id = evidence.get(
            "evidence_id",
            f"E{index:03d}"
        )

        content = extract_evidence_content(
            evidence
        )

        if not content.strip():

            continue

        score = score_evidence(
            research_question,
            content
        )

        assessment = {
            "evidence_id": evidence_id,

            "relationship": score[
                "relationship"
            ],

            "claim_supported": content[:500],

            "reasoning": score[
                "reasoning"
            ],

            "confidence": score[
                "confidence"
            ],

            "positive_score": score[
                "positive_score"
            ],

            "negative_score": score[
                "negative_score"
            ],

            "concept_coverage": score[
                "concept_coverage"
            ],

            "ai_matches": score[
                "ai_matches"
            ],

            "domain_matches": score[
                "domain_matches"
            ],

            "matched_domains": score[
                "matched_domains"
            ],

            "efficiency_matches": score[
                "efficiency_matches"
            ],

            "negative_matches": score[
                "negative_matches"
            ],

            "direct_matches": score[
                "direct_matches"
            ],

            # Preserve metadata.
            "source": evidence.get(
                "source",
                evidence.get(
                    "source_path",
                    ""
                )
            ),

            "page": evidence.get(
                "page",
                evidence.get(
                    "page_number",
                    None
                )
            ),

            "similarity_score": evidence.get(
                "similarity_score",
                evidence.get(
                    "similarity",
                    0.0
                )
            ),

            "source_type": evidence.get(
                "source_type",
                ""
            ),

            "source_quality_score": evidence.get(
                "source_quality",
                evidence.get(
                    "source_quality_score",
                    evidence.get(
                        "quality_score",
                        0.0
                    )
                )
            ),
        }

        assessments.append(
            assessment
        )

    return build_verification_result(
        research_question,
        assessments
    )


# =============================================================
# VERIFICATION SUMMARY
# =============================================================

def build_verification_result(
    research_question: str,
    assessments: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Build a consistent verification result.
    """

    total = len(
        assessments
    )

    supporting = sum(
        1
        for item in assessments
        if item.get(
            "relationship"
        ) == "Supports"
    )

    partial = sum(
        1
        for item in assessments
        if item.get(
            "relationship"
        ) == "Partially Supports"
    )

    contradicting = sum(
        1
        for item in assessments
        if item.get(
            "relationship"
        ) == "Contradicts"
    )

    neutral = sum(
        1
        for item in assessments
        if item.get(
            "relationship"
        ) == "Neutral"
    )

    relevant = (
        supporting
        + partial
    )

    # ---------------------------------------------------------
    # Relevance score
    # ---------------------------------------------------------

    if total:

        relevance_score = (
            relevant
            / total
        ) * 100

    else:

        relevance_score = 0.0

    # ---------------------------------------------------------
    # Average confidence
    # ---------------------------------------------------------

    confidence_values = []

    for item in assessments:

        try:

            confidence_values.append(
                float(
                    item.get(
                        "confidence",
                        0
                    )
                )
            )

        except (
            TypeError,
            ValueError
        ):

            pass

    if confidence_values:

        average_confidence = (
            sum(confidence_values)
            / len(confidence_values)
        )

    else:

        average_confidence = 0.0

    # ---------------------------------------------------------
    # Overall relevance
    # ---------------------------------------------------------

    if total == 0:

        overall_relevance = "Insufficient"

    elif relevance_score >= 75:

        overall_relevance = "High"

    elif relevance_score >= 50:

        overall_relevance = "Medium"

    elif relevance_score > 0:

        overall_relevance = "Low"

    else:

        overall_relevance = "Insufficient"

    # ---------------------------------------------------------
    # Verification confidence
    # ---------------------------------------------------------

    if total == 0:

        verification_confidence = "Low"

    elif average_confidence >= 0.70:

        verification_confidence = "High"

    elif average_confidence >= 0.45:

        verification_confidence = "Medium"

    else:

        verification_confidence = "Low"

    # ---------------------------------------------------------
    # Verification status
    # ---------------------------------------------------------

    if total < 2:

        verification_status = (
            "insufficient_evidence"
        )

    elif supporting == 0 and partial == 0:

        verification_status = (
            "no_supporting_evidence"
        )

    elif contradicting > supporting:

        verification_status = (
            "contradicting_evidence"
        )

    elif supporting == 0 and partial > 0:

        verification_status = (
            "partial_support_only"
        )

    else:

        verification_status = (
            "verified"
        )

    # ---------------------------------------------------------
    # Limitations
    # ---------------------------------------------------------

    limitations = []

    if total == 0:

        limitations.append(
            "No evidence items were available "
            "for verification."
        )

    if total < 3:

        limitations.append(
            "The number of evidence items is small, "
            "so the conclusion should be treated cautiously."
        )

    if partial > 0:

        limitations.append(
            f"{partial} evidence item(s) provided only "
            "partial or indirect support."
        )

    if contradicting > 0:

        limitations.append(
            f"{contradicting} evidence item(s) contained "
            "evidence that may contradict the research question."
        )

    if neutral > 0:

        limitations.append(
            f"{neutral} evidence item(s) were not sufficiently "
            "relevant to establish support."
        )

    # ---------------------------------------------------------
    # Recommendation
    # ---------------------------------------------------------

    if verification_status == "insufficient_evidence":

        recommendation = (
            "Insufficient evidence is available. "
            "Collect additional relevant evidence before "
            "making a strong conclusion."
        )

    elif verification_status == "no_supporting_evidence":

        recommendation = (
            "The retrieved evidence does not directly support "
            "the research question. Additional evidence is required."
        )

    elif verification_status == "contradicting_evidence":

        recommendation = (
            "The evidence contains substantial contradictory "
            "signals. Review the evidence manually before acting."
        )

    elif verification_status == "partial_support_only":

        recommendation = (
            "The evidence provides only partial support. "
            "Additional direct evidence should be collected."
        )

    elif supporting >= 3 and contradicting == 0:

        recommendation = (
            "The evidence provides strong support for the "
            "research question, subject to the limitations identified."
        )

    else:

        recommendation = (
            "The evidence provides useful support, but additional "
            "evidence should be considered before making a definitive decision."
        )

    return {
        "research_question": research_question,

        "evidence_assessment": assessments,

        "overall_relevance": overall_relevance,

        "verification_confidence": verification_confidence,

        "verification_status": verification_status,

        "relevance_score": round(
            relevance_score,
            2
        ),

        "average_confidence": round(
            average_confidence,
            3
        ),

        "supporting_evidence": supporting,

        "partially_supporting_evidence": partial,

        "contradicting_evidence": contradicting,

        "neutral_evidence": neutral,

        "total_evidence": total,

        "limitations": limitations,

        "recommendation": recommendation,
    }


# =============================================================
# GEMINI VERIFICATION
# =============================================================

def verify_with_gemini(
    research_question: str,
    evidence_items: List[Dict[str, Any]]
):
    """
    Gemini semantic verification.

    Returns parsed Gemini output (a dict containing
    "evidence_assessment") when available.
    Returns None when Gemini is unavailable or the
    response cannot be parsed.
    """

    try:

        from app.utils.gemini_client import (
            generate_content
        )

    except ImportError:

        return None

    evidence_text = []

    for index, evidence in enumerate(
        evidence_items,
        start=1
    ):

        if not isinstance(
            evidence,
            dict
        ):

            continue

        evidence_id = evidence.get(
            "evidence_id",
            f"E{index:03d}"
        )

        content = extract_evidence_content(
            evidence
        )

        evidence_text.append(
            f"""
Evidence ID: {evidence_id}

{content}
"""
        )

    prompt = f"""
You are an evidence verification agent in an
Agentic AI Research and Decision System.

Research question:
{research_question}

Your task is to determine whether each retrieved
evidence item supports the research question.

For each evidence item classify the relationship
as exactly one of:

Supports
Partially Supports
Contradicts
Neutral

Definitions:

Supports:
The evidence directly provides information that
helps answer or support the research question.

Partially Supports:
The evidence is relevant but incomplete,
indirect, conditional, or only addresses part
of the research question.

Contradicts:
The evidence provides information that conflicts
with the research question.

Neutral:
The evidence does not provide sufficient
information to determine support.

STRICT RULES:

1. Use ONLY the supplied evidence.
2. Do not use outside knowledge.
3. Do not assume relationships that are not present.
4. Do not classify evidence as relevant merely because
   it mentions the same general topic.
5. Consider the complete research question.
6. Evidence must actually help answer the question.
7. Do not invent statistics or conclusions.
8. Return valid JSON only.

Required format:

{{
    "evidence_assessment": [
        {{
            "evidence_id": "E001",
            "relationship": "Supports",
            "claim_supported": "short evidence-grounded claim",
            "reasoning": "short explanation",
            "confidence": 0.8
        }}
    ]
}}

Evidence:

{"".join(evidence_text)}
"""

    try:

        print(
            "\nVerifying evidence with Gemini Verification Agent..."
        )

        text = generate_content(
            prompt
        )

        if not text or not text.strip():

            print(
                "\nGemini Verification Agent returned "
                "an empty response."
            )

            return None

        text = text.strip()

        # -------------------------------------------------
        # Remove Markdown code fences
        # -------------------------------------------------

        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"\s*```$",
            "",
            text
        )

        text = text.strip()

        # -------------------------------------------------
        # Extract JSON object
        # -------------------------------------------------

        if not text.startswith("{"):

            json_start = text.find("{")
            json_end = text.rfind("}")

            if (
                json_start != -1
                and json_end != -1
                and json_end > json_start
            ):

                text = text[
                    json_start:json_end + 1
                ]

        try:

            result = json.loads(text)

        except json.JSONDecodeError:

            print(
                "\nGemini Verification Agent returned "
                "invalid JSON. Using local fallback."
            )

            return None

        if not isinstance(result, dict):

            return None

        return result

    except Exception as error:

        print(
            "\nGemini Verification Agent unavailable."
        )

        print(
            f"Reason: {error}"
        )

        print(
            "Using deterministic local verification."
        )

        return None


# =============================================================
# GEMINI RESULT NORMALIZATION
# =============================================================

def normalize_gemini_assessments(
    assessments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Normalize Gemini assessment objects.
    """

    normalized = []

    valid_relationships = {
        "Supports",
        "Partially Supports",
        "Contradicts",
        "Neutral",
    }

    for index, item in enumerate(
        assessments,
        start=1
    ):

        if not isinstance(
            item,
            dict
        ):

            continue

        relationship = item.get(
            "relationship",
            "Neutral"
        )

        if relationship not in valid_relationships:

            relationship = "Neutral"

        try:

            confidence = float(
                item.get(
                    "confidence",
                    0.0
                )
            )

        except (
            TypeError,
            ValueError
        ):

            confidence = 0.0

        normalized.append(
            {
                "evidence_id": item.get(
                    "evidence_id",
                    f"E{index:03d}"
                ),

                "relationship": relationship,

                "claim_supported": item.get(
                    "claim_supported",
                    ""
                ),

                "reasoning": item.get(
                    "reasoning",
                    ""
                ),

                "confidence": confidence,
            }
        )

    return normalized


# =============================================================
# MAIN VERIFIER
# =============================================================

def verify_evidence(
    research_question: str,
    evidence: Optional[
        List[Dict[str, Any]]
    ] = None,
    evidence_items: Optional[
        List[Dict[str, Any]]
    ] = None,
    use_gemini: bool = True
) -> Dict[str, Any]:
    """
    Main verification entry point.

    Supports:

        verify_evidence(
            research_question=question,
            evidence=evidence
        )

    and:

        verify_evidence(
            research_question=question,
            evidence_items=evidence
        )
    """

    # ---------------------------------------------------------
    # Resolve evidence input
    # ---------------------------------------------------------

    if evidence is not None:

        evidence_list = evidence

    elif evidence_items is not None:

        evidence_list = evidence_items

    else:

        evidence_list = []

    # ---------------------------------------------------------
    # Safety normalization
    # ---------------------------------------------------------

    if not isinstance(
        evidence_list,
        list
    ):

        evidence_list = []

    # ---------------------------------------------------------
    # Remove invalid items
    # ---------------------------------------------------------

    valid_evidence = []

    for item in evidence_list:

        if not isinstance(
            item,
            dict
        ):

            continue

        content = extract_evidence_content(
            item
        )

        if content.strip():

            valid_evidence.append(
                item
            )

    evidence_list = valid_evidence

    # ---------------------------------------------------------
    # Empty evidence
    # ---------------------------------------------------------

    if not evidence_list:

        return build_verification_result(
            research_question,
            []
        )

    # ---------------------------------------------------------
    # Try Gemini first
    # ---------------------------------------------------------

    if use_gemini:

        gemini_result = verify_with_gemini(
            research_question,
            evidence_list
        )

        if isinstance(
            gemini_result,
            dict
        ):

            assessments = gemini_result.get(
                "evidence_assessment",
                []
            )

            if (
                isinstance(
                    assessments,
                    list
                )
                and assessments
            ):

                normalized_assessments = (
                    normalize_gemini_assessments(
                        assessments
                    )
                )

                if normalized_assessments:

                    return build_verification_result(
                        research_question,
                        normalized_assessments
                    )

    # ---------------------------------------------------------
    # Local deterministic fallback
    # ---------------------------------------------------------

    result = local_verify_evidence(
        research_question,
        evidence_list
    )

    result.setdefault(
        "limitations",
        []
    )

    result["limitations"].insert(
        0,
        "Gemini verification was unavailable, so deterministic "
        "local evidence verification was used."
    )

    result["verification_status"] = (
        result.get(
            "verification_status",
            "verified"
        )
        + "_local_fallback"
    )

    return result


# =============================================================
# DIRECT TEST
# =============================================================

if __name__ == "__main__":

    question = (
        "How can AI improve healthcare?"
    )

    test_evidence = [

        {
            "evidence_id": "E001",
            "source": "WHO AI Healthcare Ethics.pdf",
            "page": 3,
            "source_type": "government",
            "source_quality": 1.0,
            "similarity_score": 0.7610,
            "content": """
            AI can enable resource-poor countries,
            where patients often have restricted access
            to health-care workers or medical professionals,
            to bridge gaps in access to health care.
            """
        },

        {
            "evidence_id": "E002",
            "source": "WHO AI Healthcare Ethics.pdf",
            "page": 3,
            "source_type": "government",
            "source_quality": 1.0,
            "similarity_score": 0.7298,
            "content": """
            AI can augment the ability of health-care
            providers to improve patient care, provide
            accurate diagnoses, optimize treatment plans,
            support pandemic preparedness and response,
            inform the decisions of health policy-makers
            or allocate resources within health systems.
            """
        },

        {
            "evidence_id": "E003",
            "source": "WHO AI Healthcare Ethics.pdf",
            "page": 7,
            "source_type": "government",
            "source_quality": 1.0,
            "similarity_score": 0.6622,
            "content": """
            Ethical, transparent design of AI technologies,
            mechanisms for engagement with providers and
            patients, impact assessment, and research for
            ethical use of AI are important for healthcare.
            """
        },
    ]

    print(
        "\n" + "=" * 60
    )

    print(
        "LOCAL VERIFIER TEST"
    )

    print(
        "=" * 60
    )

    result = verify_evidence(
        research_question=question,
        evidence=test_evidence,
        use_gemini=False
    )

    print(
        "\nVerification Summary:"
    )

    print(
        "Supporting:",
        result[
            "supporting_evidence"
        ]
    )

    print(
        "Partially Supporting:",
        result[
            "partially_supporting_evidence"
        ]
    )

    print(
        "Contradicting:",
        result[
            "contradicting_evidence"
        ]
    )

    print(
        "Neutral:",
        result[
            "neutral_evidence"
        ]
    )

    print(
        "Overall Relevance:",
        result[
            "overall_relevance"
        ]
    )

    print(
        "Relevance Score:",
        result[
            "relevance_score"
        ]
    )

    print(
        "\nDetailed Assessment:"
    )

    for item in result[
        "evidence_assessment"
    ]:

        print(
            f"\n{item['evidence_id']}"
        )

        print(
            "Relationship:",
            item[
                "relationship"
            ]
        )

        print(
            "Confidence:",
            item[
                "confidence"
            ]
        )

        print(
            "Reasoning:",
            item[
                "reasoning"
            ]
        )