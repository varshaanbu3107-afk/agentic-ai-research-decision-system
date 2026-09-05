import json
import re

from app.utils.gemini_client import generate_content


# =============================================================
# RESEARCH AGENT
# =============================================================

def analyze_evidence(
    research_question: str,
    evidence: str
) -> dict:
    """
    Research Agent.

    Uses Gemini when available.

    If Gemini is unavailable, uses a deterministic local
    evidence-analysis fallback.

    The fallback:
    - removes PDF formatting noise
    - removes incomplete sentence fragments
    - extracts complete evidence-grounded sentences
    - uses the research question to rank relevance
    - prioritizes direct answer/action sentences
    - avoids copying entire PDF chunks
    - preserves Evidence IDs
    - does not use outside knowledge
    """

    # =========================================================
    # 1. EMPTY EVIDENCE
    # =========================================================

    if not evidence or not evidence.strip():

        return {
            "research_question": research_question,
            "key_findings": [],
            "supporting_evidence": [],
            "limitations": [
                "No evidence was retrieved from the document collection."
            ],
            "confidence": "Low",
            "analysis_status": "no_evidence"
        }

    # =========================================================
    # 2. EXTRACT EVIDENCE IDS
    # =========================================================

    evidence_ids = sorted(
        set(
            re.findall(
                r"\bE\d{3}\b",
                evidence
            )
        )
    )

    evidence_id_text = ", ".join(evidence_ids)

    # =========================================================
    # 3. GEMINI PROMPT
    # =========================================================

    prompt = f"""
You are the Researcher Agent in an Agentic AI
Research and Decision System.

Your responsibility is to analyze ONLY the retrieved
evidence and produce concise evidence-grounded findings.

Research Question:
{research_question}

Retrieved Evidence:
{evidence}

Valid Evidence IDs:
{evidence_id_text}

Return ONLY valid JSON using exactly this structure:

{{
    "research_question": "string",
    "key_findings": [
        "string"
    ],
    "supporting_evidence": [
        "string"
    ],
    "limitations": [
        "string"
    ],
    "confidence": "High | Medium | Low"
}}

STRICT RULES:

1. Use ONLY information explicitly present in the evidence.

2. Do NOT use outside knowledge.

3. Do NOT invent statistics, percentages, costs,
   benefits, risks, technologies, or conclusions.

4. Every key finding must be directly supported
   by retrieved evidence.

5. Keep findings concise and complete.

6. Do NOT copy entire evidence chunks.

7. Do NOT return incomplete sentence fragments.

8. Every supporting_evidence item must contain
   the correct Evidence ID.

9. Do not use an Evidence ID that does not exist.

10. Do not force unrelated evidence into the answer.

11. The research_question must exactly match the
    supplied research question.

12. Return only valid JSON.
"""

    # =========================================================
    # 4. TRY GEMINI
    # =========================================================

    try:

        print(
            "\nAnalyzing evidence with Gemini Research Agent..."
        )

        text = generate_content(prompt)

        if not text or not text.strip():

            raise ValueError(
                "Researcher Agent returned an empty response."
            )

        text = text.strip()

        # -----------------------------------------------------
        # Remove Markdown code fences
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Extract JSON object
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Parse JSON
        # -----------------------------------------------------

        try:

            result = json.loads(text)

        except json.JSONDecodeError as error:

            raise ValueError(
                "Researcher Agent returned invalid JSON:\n"
                f"{text}"
            ) from error

        # -----------------------------------------------------
        # Validate object
        # -----------------------------------------------------

        if not isinstance(result, dict):

            raise ValueError(
                "Researcher Agent response must be a JSON object."
            )

        # -----------------------------------------------------
        # Required fields
        # -----------------------------------------------------

        required_fields = [
            "research_question",
            "key_findings",
            "supporting_evidence",
            "limitations",
            "confidence"
        ]

        for field in required_fields:

            if field not in result:

                raise ValueError(
                    "Researcher Agent response is missing "
                    f"required field: {field}"
                )

        # -----------------------------------------------------
        # Validate list fields
        # -----------------------------------------------------

        list_fields = [
            "key_findings",
            "supporting_evidence",
            "limitations"
        ]

        for field in list_fields:

            if not isinstance(
                result[field],
                list
            ):

                raise ValueError(
                    f"Researcher Agent '{field}' must be a list."
                )

            for item in result[field]:

                if not isinstance(item, str):

                    raise ValueError(
                        f"Researcher Agent '{field}' "
                        "must contain only strings."
                    )

        # -----------------------------------------------------
        # Validate confidence
        # -----------------------------------------------------

        if result["confidence"] not in {
            "High",
            "Medium",
            "Low"
        }:

            raise ValueError(
                "Researcher Agent 'confidence' must be "
                "High, Medium, or Low."
            )

        # -----------------------------------------------------
        # Preserve exact research question
        # -----------------------------------------------------

        result["research_question"] = research_question

        # -----------------------------------------------------
        # Validate Evidence IDs
        # -----------------------------------------------------

        valid_ids = set(evidence_ids)

        cleaned_supporting_evidence = []

        for item in result["supporting_evidence"]:

            referenced_ids = set(
                re.findall(
                    r"\bE\d{3}\b",
                    item
                )
            )

            invalid_ids = (
                referenced_ids - valid_ids
            )

            if invalid_ids:

                raise ValueError(
                    "Researcher Agent referenced invalid "
                    f"Evidence ID(s): {invalid_ids}"
                )

            item = item.strip()

            if item:
                cleaned_supporting_evidence.append(
                    item
                )

        result["supporting_evidence"] = (
            cleaned_supporting_evidence
        )

        # -----------------------------------------------------
        # Remove empty strings
        # -----------------------------------------------------

        for field in list_fields:

            result[field] = [
                item.strip()
                for item in result[field]
                if item.strip()
            ]

        result["analysis_status"] = (
            "llm_based_evidence_analysis"
        )

        return result

    # =========================================================
    # 5. GEMINI FAILURE
    # =========================================================

    except RuntimeError as error:

        error_text = str(error).lower()

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
            "\nGemini Research Agent unavailable."
        )

        print(
            "Using local fallback evidence analysis."
        )

        print(
            f"Reason: {error}"
        )

        return _create_local_fallback_analysis(
            research_question,
            evidence
        )


# =============================================================
# TEXT NORMALIZATION
# =============================================================

def _normalize_text(text: str) -> str:
    """
    Clean PDF text without changing its meaning.
    """

    text = str(text)

    # Fix common PDF ligatures.
    replacements = {
        "ï¬": "fi",
        "ï¬‚": "fl",
        "ï¬€": "ff",
        "ï¬ƒ": "ffi",
        "ï¬„": "ffl"
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )

    # Join words split by PDF hyphenation.
    text = re.sub(
        r"(\w)-\s*\n\s*(\w)",
        r"\1\2",
        text
    )

    # Convert line breaks into spaces.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =============================================================
# EVIDENCE BLOCK EXTRACTION
# =============================================================

def _extract_evidence_blocks(evidence: str) -> list:
    """
    Extract:

        E001 -> content
        E002 -> content

    from the formatted evidence supplied by the RAG pipeline.
    """

    pattern = re.compile(
        r"EVIDENCE\s+(\d+)(.*?)(?=EVIDENCE\s+\d+|$)",
        flags=re.IGNORECASE | re.DOTALL
    )

    blocks = []

    for match in pattern.finditer(evidence):

        number = int(
            match.group(1)
        )

        evidence_id = (
            f"E{number:03d}"
        )

        block = match.group(2).strip()

        content_match = re.search(
            r"Content\s*:\s*(.*)",
            block,
            flags=re.IGNORECASE | re.DOTALL
        )

        if content_match:

            content = (
                content_match.group(1)
                .strip()
            )

        else:

            content = block

        content = _normalize_text(
            content
        )

        if content:

            blocks.append({
                "evidence_id": evidence_id,
                "content": content
            })

    return blocks


# =============================================================
# QUESTION TERMS
# =============================================================

STOP_WORDS = {
    "how",
    "can",
    "could",
    "would",
    "should",
    "what",
    "why",
    "when",
    "where",
    "which",
    "does",
    "do",
    "is",
    "are",
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "by",
    "from",
    "about",
    "related",
    "this",
    "that",
    "these",
    "those",
    "be",
    "as"
}


def _get_question_terms(
    research_question: str
) -> set:
    """
    Extract meaningful words from the research question.
    """

    normalized = _normalize_text(
        research_question
    ).lower()

    words = re.findall(
        r"[a-z0-9]+",
        normalized
    )

    return {
        word
        for word in words
        if len(word) >= 3
        and word not in STOP_WORDS
    }


# =============================================================
# DOMAIN CONCEPTS
# =============================================================

DOMAIN_GROUPS = {

    "ai": {
        "ai",
        "artificial",
        "intelligence",
        "machine",
        "learning",
        "ml",
        "automation",
        "automated",
        "automate",
        "chatbot",
        "chatbots",
        "algorithm",
        "algorithms",
        "technology",
        "technologies",
        "agent",
        "agents"
    },

    "healthcare": {
        "healthcare",
        "health",
        "health-care",
        "medical",
        "medicine",
        "patient",
        "patients",
        "clinical",
        "clinician",
        "clinicians",
        "hospital",
        "hospitals",
        "diagnosis",
        "diagnoses",
        "treatment",
        "treatments",
        "pandemic",
        "care",
        "health-system",
        "health-systems"
    },

    "customer_support": {
        "customer",
        "customers",
        "support",
        "service",
        "services",
        "chatbot",
        "chatbots",
        "request",
        "requests",
        "complaint",
        "complaints",
        "agent",
        "agents",
        "client",
        "clients"
    },

    "efficiency": {
        "efficiency",
        "efficient",
        "improve",
        "improved",
        "improves",
        "improvement",
        "improvements",
        "faster",
        "fast",
        "quick",
        "quickly",
        "rapid",
        "performance",
        "optimize",
        "optimization",
        "productivity",
        "saving",
        "savings",
        "reduce",
        "reduced",
        "reduction",
        "time",
        "effort",
        "outcomes",
        "benefit",
        "benefits"
    },

    "ethics": {
        "ethical",
        "ethically",
        "ethics",
        "rights",
        "privacy",
        "dignity",
        "autonomy",
        "trust",
        "trustworthiness",
        "transparent",
        "transparency",
        "governance",
        "liability",
        "legal",
        "laws",
        "policies",
        "policy",
        "risk",
        "risks"
    }
}


# =============================================================
# CONCEPT DETECTION
# =============================================================

def _detect_question_concepts(
    question_terms: set
) -> set:
    """
    Identify broad concepts present in the question.
    """

    concepts = set()

    for concept, terms in DOMAIN_GROUPS.items():

        if any(
            term in question_terms
            for term in terms
        ):

            concepts.add(
                concept
            )

    return concepts


def _detect_content_concepts(
    content: str
) -> set:
    """
    Identify concepts present in evidence.
    """

    normalized = _normalize_text(
        content
    ).lower()

    words = set(
        re.findall(
            r"[a-z0-9-]+",
            normalized
        )
    )

    concepts = set()

    for concept, terms in DOMAIN_GROUPS.items():

        if any(
            term in words
            or term in normalized
            for term in terms
        ):

            concepts.add(
                concept
            )

    return concepts


# =============================================================
# SENTENCE EXTRACTION
# =============================================================

def _extract_complete_sentences(
    content: str
) -> list:
    """
    Extract complete sentences.

    Important:
    PDF retrieval chunks often begin or end in the middle
    of a sentence. Those fragments must NOT become findings.
    """

    content = _normalize_text(
        content
    )

    if not content:
        return []

    # Split at actual sentence-ending punctuation.
    raw_sentences = re.split(
        r"(?<=[.!?])\s+",
        content
    )

    sentences = []

    for sentence in raw_sentences:

        sentence = sentence.strip()

        if not sentence:
            continue

        # -----------------------------------------------------
        # Reject obvious leading fragments.
        # -----------------------------------------------------

        fragment_starters = {
            "and",
            "or",
            "but",
            "because",
            "although",
            "however",
            "therefore",
            "they",
            "them",
            "which",
            "that",
            "these",
            "those",
            "it",
            "this"
        }

        first_word_match = re.match(
            r"([A-Za-z]+)",
            sentence
        )

        if first_word_match:

            first_word = (
                first_word_match.group(1)
                .lower()
            )

            if (
                first_word in fragment_starters
                and len(sentence) < 180
            ):
                continue

        # -----------------------------------------------------
        # Reject very short fragments.
        # -----------------------------------------------------

        if len(sentence) < 45:
            continue

        # -----------------------------------------------------
        # Reject obvious metadata.
        # -----------------------------------------------------

        lower = sentence.lower()

        metadata_terms = [
            "creative commons",
            "copyright",
            "issn",
            "isbn",
            "doi:",
            "correspondence:",
            "department of",
            "university of",
            "received ",
            "accepted ",
            "figure ",
            "table "
        ]

        if any(
            term in lower
            for term in metadata_terms
        ):
            continue

        # -----------------------------------------------------
        # Clean extra whitespace.
        # -----------------------------------------------------

        sentence = re.sub(
            r"\s+",
            " ",
            sentence
        ).strip()

        # -----------------------------------------------------
        # Sentence must end properly.
        # -----------------------------------------------------

        if not sentence.endswith(
            (".", "!", "?")
        ):
            continue

        sentences.append(
            sentence
        )

    return sentences


# =============================================================
# SENTENCE SCORING
# =============================================================

def _score_sentence(
    sentence: str,
    question_terms: set,
    question_concepts: set
) -> float:
    """
    Score how strongly a sentence answers the research question.

    Direct answer/action sentences receive a strong bonus.
    This is especially useful for questions such as:

        How can AI improve healthcare?

because sentences such as:

        AI can augment...
        AI can enable...
        AI can improve...

are more useful findings than general ethical statements.
    """

    normalized = _normalize_text(
        sentence
    ).lower()

    words = set(
        re.findall(
            r"[a-z0-9-]+",
            normalized
        )
    )

    # ---------------------------------------------------------
    # Question keyword overlap
    # ---------------------------------------------------------

    keyword_matches = (
        question_terms & words
    )

    score = (
        len(keyword_matches) * 2.0
    )

    # ---------------------------------------------------------
    # Concept overlap
    # ---------------------------------------------------------

    content_concepts = _detect_content_concepts(
        normalized
    )

    concept_matches = (
        question_concepts &
        content_concepts
    )

    score += (
        len(concept_matches) * 5.0
    )

    # ---------------------------------------------------------
    # Direct answer/action indicators
    # ---------------------------------------------------------

    benefit_terms = {
        "improve",
        "improves",
        "improved",
        "improvement",
        "improvements",
        "augment",
        "supports",
        "support",
        "provide",
        "provides",
        "accurate",
        "optimize",
        "optimizing",
        "optimization",
        "enable",
        "enables",
        "bridge",
        "performance",
        "outcomes",
        "benefit",
        "benefits",
        "automate",
        "automation",
        "reduce",
        "reduces",
        "reduction",
        "saving",
        "savings"
    }

    benefit_hits = sum(
        1
        for term in benefit_terms
        if term in words
    )

    score += (
        benefit_hits * 2.5
    )

    # ---------------------------------------------------------
    # Strong bonus for sentences that directly describe
    # what AI can do.
    # ---------------------------------------------------------

    direct_action_patterns = [
        r"\bai can\b",
        r"\bai enables\b",
        r"\bai can enable\b",
        r"\bai can augment\b",
        r"\bai can improve\b",
        r"\bai can support\b",
        r"\bai can provide\b",
        r"\bai can optimize\b",
        r"\bai can help\b"
    ]

    for pattern in direct_action_patterns:

        if re.search(
            pattern,
            normalized,
            flags=re.IGNORECASE
        ):
            score += 8.0
            break

    # ---------------------------------------------------------
    # Risk/ethics terms are useful, but should not dominate
    # benefit-oriented questions.
    # ---------------------------------------------------------

    ethics_terms = {
        "ethical",
        "ethics",
        "rights",
        "privacy",
        "dignity",
        "autonomy",
        "trust",
        "transparent",
        "transparency",
        "liability",
        "legal",
        "governance",
        "risk",
        "risks"
    }

    ethics_hits = sum(
        1
        for term in ethics_terms
        if term in words
    )

    if "ethics" not in question_concepts:

        score -= (
            ethics_hits * 1.5
        )

    # ---------------------------------------------------------
    # Very long sentences are less useful as findings.
    # ---------------------------------------------------------

    if len(sentence) > 450:

        score -= 1.0

    return score


# =============================================================
# CLEAN FINDING
# =============================================================

def _clean_finding(
    sentence: str
) -> str:
    """
    Convert a complete evidence sentence into a clean
    report-ready finding without changing its meaning.
    """

    sentence = _normalize_text(
        sentence
    )

    # Remove leading continuation markers.
    sentence = re.sub(
        r"^(and|or|but|however|therefore)\s+",
        "",
        sentence,
        flags=re.IGNORECASE
    )

    sentence = sentence.strip()

    if not sentence:
        return ""

    # Capitalize first character if necessary.
    sentence = (
        sentence[0].upper()
        +
        sentence[1:]
    )

    return sentence


# =============================================================
# DEDUPLICATION
# =============================================================

def _deduplicate(
    sentences: list
) -> list:
    """
    Remove duplicate or near-duplicate findings.
    """

    result = []
    seen = set()

    for sentence in sentences:

        normalized = re.sub(
            r"[^a-z0-9]+",
            " ",
            sentence.lower()
        ).strip()

        if not normalized:
            continue

        if normalized in seen:
            continue

        # Basic containment duplicate check.
        duplicate = False

        for previous in result:

            previous_normalized = re.sub(
                r"[^a-z0-9]+",
                " ",
                previous.lower()
            ).strip()

            if (
                normalized in previous_normalized
                or
                previous_normalized in normalized
            ):
                duplicate = True
                break

        if duplicate:
            continue

        seen.add(
            normalized
        )

        result.append(
            sentence
        )

    return result


# =============================================================
# LOCAL FALLBACK ANALYSIS
# =============================================================

def _create_local_fallback_analysis(
    research_question: str,
    evidence: str
) -> dict:
    """
    Deterministic local fallback.

    This function intentionally does NOT copy complete PDF
    chunks into key_findings.

    It extracts complete sentences, ranks them against the
    research question, and returns only the strongest
    evidence-grounded findings.
    """

    # =========================================================
    # 1. EXTRACT EVIDENCE BLOCKS
    # =========================================================

    evidence_blocks = _extract_evidence_blocks(
        evidence
    )

    if not evidence_blocks:

        return {
            "research_question": research_question,
            "key_findings": [],
            "supporting_evidence": [],
            "limitations": [
                "No structured evidence items were available "
                "for local analysis."
            ],
            "confidence": "Low",
            "analysis_status": "local_fallback"
        }

    # =========================================================
    # 2. QUESTION ANALYSIS
    # =========================================================

    question_terms = _get_question_terms(
        research_question
    )

    question_concepts = _detect_question_concepts(
        question_terms
    )

    # =========================================================
    # 3. EXTRACT COMPLETE SENTENCES
    # =========================================================

    candidates = []

    for block in evidence_blocks:

        evidence_id = block[
            "evidence_id"
        ]

        content = block[
            "content"
        ]

        sentences = _extract_complete_sentences(
            content
        )

        for sentence in sentences:

            clean_sentence = _clean_finding(
                sentence
            )

            if not clean_sentence:
                continue

            score = _score_sentence(
                clean_sentence,
                question_terms,
                question_concepts
            )

            candidates.append({
                "evidence_id": evidence_id,
                "sentence": clean_sentence,
                "score": score
            })

    # =========================================================
    # 4. SORT BY RELEVANCE
    # =========================================================

    candidates.sort(
        key=lambda item: (
            item["score"],
            len(item["sentence"])
        ),
        reverse=True
    )

    # =========================================================
    # 5. REMOVE DUPLICATES
    # =========================================================

    selected = []

    seen_sentences = set()

    for candidate in candidates:

        sentence = candidate[
            "sentence"
        ]

        normalized = re.sub(
            r"[^a-z0-9]+",
            " ",
            sentence.lower()
        ).strip()

        if normalized in seen_sentences:
            continue

        seen_sentences.add(
            normalized
        )

        selected.append(
            candidate
        )

    # =========================================================
    # 6. SELECT STRONG FINDINGS
    # =========================================================

    strong_candidates = [
        item
        for item in selected
        if item["score"] >= 5
    ]

    # If strict threshold leaves nothing, use the best
    # evidence-grounded candidates.
    if not strong_candidates:

        strong_candidates = [
            item
            for item in selected
            if item["score"] >= 3
        ]

    strong_candidates = strong_candidates[:5]

    # =========================================================
    # 7. BUILD KEY FINDINGS
    # =========================================================

    key_findings = [
        item["sentence"]
        for item in strong_candidates
    ]

    key_findings = _deduplicate(
        key_findings
    )

    key_findings = key_findings[:5]

    # =========================================================
    # 8. BUILD SUPPORTING EVIDENCE
    # =========================================================

    supporting_evidence = []

    used_supporting = set()

    for item in strong_candidates:

        evidence_id = item[
            "evidence_id"
        ]

        sentence = item[
            "sentence"
        ]

        support_text = (
            f"{sentence} ({evidence_id})"
        )

        normalized = support_text.lower()

        if normalized in used_supporting:
            continue

        used_supporting.add(
            normalized
        )

        supporting_evidence.append(
            support_text
        )

    # =========================================================
    # 9. CONFIDENCE
    # =========================================================

    evidence_sources = {
        item["evidence_id"]
        for item in strong_candidates
    }

    if (
        len(key_findings) >= 3
        and len(evidence_sources) >= 2
    ):

        confidence = "Medium"

    elif (
        len(key_findings) >= 2
    ):

        confidence = "Medium"

    else:

        confidence = "Low"

    # =========================================================
    # 10. LIMITATIONS
    # =========================================================

    limitations = [
        "Gemini was unavailable, so deterministic local "
        "evidence extraction was used.",

        "The local fallback uses question-aware lexical "
        "and concept matching rather than full semantic "
        "LLM reasoning.",

        "The findings are limited to the evidence retrieved "
        "by the RAG pipeline."
    ]

    if not key_findings:

        limitations.append(
            "The retrieved evidence did not contain "
            "sufficient complete sentences that directly "
            "answered the research question."
        )

    # =========================================================
    # 11. RETURN
    # =========================================================

    return {
        "research_question": research_question,

        "key_findings": key_findings,

        "supporting_evidence": supporting_evidence,

        "limitations": limitations,

        "confidence": confidence,

        "analysis_status": "local_fallback"
    }


# =============================================================
# DIRECT TEST
# =============================================================

if __name__ == "__main__":

    question = (
        "How can AI improve healthcare?"
    )

    evidence = """
EVIDENCE 1

Evidence ID: E001

Source: WHO AI Healthcare Ethics.pdf

Page: 3

Content:
health care and better understand their evolving needs.
To achieve this, patients and communities require assurance
that their rights and interests will not be subordinated to
the powerful commercial interests of technology companies.
AI can enable resource-poor countries, where patients often
have restricted access to health-care workers or medical
professionals, to bridge gaps in access to health care.

EVIDENCE 2

Evidence ID: E002

Source: WHO AI Healthcare Ethics.pdf

Page: 3

Content:
AI, although AI itself presents a number of novel concerns.
Whether AI can advance the interests of patients and
communities depends on a collective effort to design and
implement ethically defensible laws and policies.
AI can augment the ability of health-care providers to
improve patient care, provide accurate diagnoses, optimize
treatment plans, support pandemic preparedness and response,
inform the decisions of health policy-makers or allocate
resources within health care.

EVIDENCE 3

Evidence ID: E003

Source: WHO AI Healthcare Ethics.pdf

Page: 7

Content:
programmes and measures to anticipate or meet ethical norms
and legal obligations. They include: ethical, transparent
design of AI technologies; mechanisms for the engagement
and role of the public and demonstrating trustworthiness
with providers and patients; impact assessment; and a
research agenda for ethical use of AI for health care.
"""

    result = analyze_evidence(
        question,
        evidence
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "RESEARCH EVIDENCE ANALYSIS"
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

