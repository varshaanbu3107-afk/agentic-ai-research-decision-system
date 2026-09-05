import json
import re

from app.rag.vector_store import VectorStore
from app.agents.researcher import analyze_evidence

from app.research.evidence_builder import build_evidence
from app.research.source_quality import get_source_metadata


# =========================================================
# CONFIGURATION
# =========================================================

MIN_SIMILARITY_SCORE = 0.30

MAX_EVIDENCE_CHUNKS = 8

MIN_CONTENT_LENGTH = 80

MAX_METADATA_HITS = 3

MIN_DOMAIN_SCORE = 0.15

MAX_CHUNKS_PER_SOURCE = 3

# Minimum number of meaningful topic matches required
# for unknown/general domains.
MIN_TOPIC_MATCHES = 1

# For multi-word concepts, require the main concept or
# enough related concept terms before accepting evidence.
MIN_GENERAL_TOPIC_SCORE = 0.20


# =========================================================
# STOPWORDS
# =========================================================

STOPWORDS = {
    "what",
    "are",
    "is",
    "the",
    "a",
    "an",
    "of",
    "and",
    "or",
    "to",
    "for",
    "in",
    "on",
    "with",
    "how",
    "why",
    "can",
    "could",
    "would",
    "should",
    "does",
    "do",
    "from",
    "by",
    "be",
    "this",
    "that",
    "these",
    "those",
    "related",
    "specific",
    "different",
    "various",
    "following",
    "benefits",
    "benefit",
    "advantages",
    "advantage",
    "risks",
    "risk",
    "limitations",
    "limitation",
    "challenges",
    "challenge",
    "impact",
    "effects",
    "effect",
    "ways",
    "way",
    "methods",
    "method",
    "approaches",
    "approach",
    "measurable",
    "positive",
    "outcomes",
    "outcome",
    "associated",
    "considered",
    "successful",
    "implementation",
    "improve",
    "improvement",
    "improvements",
    "necessary",
    "requirements",
    "required",
}


# =========================================================
# DOMAIN KEYWORDS
# =========================================================

DOMAIN_KEYWORDS = {

    "healthcare": [
        "health",
        "healthcare",
        "medical",
        "medicine",
        "clinical",
        "hospital",
        "patient",
        "patients",
        "doctor",
        "doctors",
        "physician",
        "physicians",
        "nurse",
        "nurses",
        "diagnosis",
        "diagnostic",
        "disease",
        "diseases",
        "treatment",
        "treatments",
        "therapy",
        "therapeutic",
        "drug",
        "drugs",
        "medication",
        "medications",
        "pharmaceutical",
        "radiology",
        "pathology",
        "surgery",
        "surgical",
        "clinical trial",
        "electronic health record",
        "ehr",
        "medical record",
        "medical records",
        "healthcare system",
        "health system",
        "public health",
        "mental health",
        "telemedicine",
        "telehealth",
        "biomedical",
        "biomarker",
        "genomics",
        "genomic",
        "medical imaging",
        "patient care",
        "patient safety",
        "health outcome",
        "health outcomes",
        "mortality",
        "morbidity",
    ],

    "customer_support": [
        "customer support",
        "customer service",
        "customer experience",
        "customer satisfaction",
        "customer",
        "support agent",
        "support agents",
        "service agent",
        "service agents",
        "chatbot",
        "chatbots",
        "conversational agent",
        "conversational agents",
        "virtual assistant",
        "natural language processing",
        "nlp",
        "response time",
        "problem resolution",
        "issue resolution",
        "query resolution",
        "help desk",
        "contact center",
        "call center",
        "first contact resolution",
        "average handle time",
        "aht",
        "fcr",
        "self-service",
        "customer interaction",
        "customer interactions",
    ],

    "finance": [
        "finance",
        "financial",
        "banking",
        "bank",
        "investment",
        "investments",
        "stock",
        "stocks",
        "trading",
        "credit",
        "loan",
        "loans",
        "insurance",
        "fraud",
        "fintech",
        "accounting",
        "financial risk",
        "portfolio",
        "payment",
        "payments",
    ],

    "education": [
        "education",
        "educational",
        "school",
        "schools",
        "student",
        "students",
        "teacher",
        "teachers",
        "university",
        "universities",
        "college",
        "colleges",
        "learning",
        "teaching",
        "classroom",
        "curriculum",
        "academic",
        "assessment",
        "exam",
        "exams",
        "e-learning",
        "elearning",
    ],

    "software": [
        "software",
        "software development",
        "developer",
        "developers",
        "programming",
        "coding",
        "code",
        "code generation",
        "api",
        "apis",
        "crud",
        "repository",
        "repositories",
        "github",
        "architecture",
        "software architecture",
        "debugging",
        "testing",
        "deployment",
        "application",
        "applications",
        "programming language",
        "programming languages",
    ],

    "general_ai": [
        "artificial intelligence",
        "ai",
        "machine learning",
        "deep learning",
        "neural network",
        "neural networks",
        "large language model",
        "large language models",
        "llm",
        "llms",
        "generative ai",
        "generative artificial intelligence",
        "natural language processing",
        "nlp",
        "computer vision",
        "robotics",
        "automation",
        "intelligent system",
        "intelligent systems",
        "ai model",
        "ai models",
    ],
}


# =========================================================
# GENERAL TOPIC CONCEPTS
# =========================================================
#
# These help unknown-domain questions.
#
# IMPORTANT:
# Generic words such as "energy" are intentionally NOT
# enough by themselves to identify renewable energy.
#
# A concept such as "renewable energy" can match:
#   renewable energy
#   renewable electricity
#   clean energy
#   clean power
#   solar energy
#   wind energy
#   hydropower
#   geothermal energy
#
# This prevents an unrelated document mentioning the word
# "energy" from being considered renewable-energy evidence.
# =========================================================

TOPIC_CONCEPT_ALIASES = {

    "renewable energy": [
        "renewable energy",
        "renewable electricity",
        "renewable power",
        "clean energy",
        "clean electricity",
        "clean power",
        "solar energy",
        "solar power",
        "wind energy",
        "wind power",
        "hydropower",
        "hydroelectric power",
        "hydroelectricity",
        "geothermal energy",
        "geothermal power",
        "biomass energy",
        "bioenergy",
        "sustainable energy",
    ],

    "artificial intelligence": [
        "artificial intelligence",
        "ai system",
        "ai systems",
        "ai model",
        "ai models",
        "machine learning",
        "deep learning",
        "generative ai",
        "large language model",
        "large language models",
    ],

    "machine learning": [
        "machine learning",
        "ml model",
        "ml models",
        "supervised learning",
        "unsupervised learning",
        "deep learning",
        "neural network",
        "neural networks",
    ],

    "customer support": [
        "customer support",
        "customer service",
        "customer experience",
        "customer satisfaction",
        "support agent",
        "support agents",
        "chatbot",
        "chatbots",
        "contact center",
        "call center",
    ],

    "healthcare": [
        "healthcare",
        "health care",
        "medical care",
        "patient care",
        "health system",
        "health systems",
        "healthcare system",
        "medical system",
    ],

    "climate change": [
        "climate change",
        "global warming",
        "greenhouse gas",
        "greenhouse gases",
        "carbon emissions",
        "carbon emission",
        "climate crisis",
        "climate mitigation",
        "climate adaptation",
    ],
}


# =========================================================
# METADATA / NOISE TERMS
# =========================================================

METADATA_TERMS = {
    "creative commons",
    "copyright",
    "license",
    "attribution",
    "doi",
    "publisher",
    "correspondence",
    "author",
    "authors",
    "received",
    "accepted",
    "keywords",
    "references",
    "issn",
    "isbn",
    "volume",
    "issue",
    "department",
    "university",
    "faculty",
    "submitted",
    "manuscript",
    "publication",
}


# =========================================================
# NORMALIZATION
# =========================================================

def _normalize_text(text: str) -> str:
    """
    Normalize text for deterministic lexical matching.
    """

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# TOPIC TERM EXTRACTION
# =========================================================

def extract_topic_terms(
    research_question: str
):
    """
    Extract meaningful words and short phrases from the
    research question.

    Generic question words are removed.
    """

    normalized = _normalize_text(
        research_question
    )

    words = normalized.split()

    topic_terms = []

    for word in words:

        if len(word) < 3:
            continue

        if word in STOPWORDS:
            continue

        if word.isdigit():
            continue

        if word not in topic_terms:
            topic_terms.append(
                word
            )

    # Add meaningful two-word phrases.
    for index in range(
        len(words) - 1
    ):

        first = words[index]
        second = words[index + 1]

        phrase = (
            f"{first} {second}"
        )

        if (
            first not in STOPWORDS
            and second not in STOPWORDS
            and len(first) >= 3
            and len(second) >= 3
        ):

            if phrase not in topic_terms:

                topic_terms.append(
                    phrase
                )

    return topic_terms


# =========================================================
# CONCEPT DETECTION
# =========================================================

def detect_topic_concepts(
    research_question: str
):
    """
    Detect known multi-word concepts in the research
    question.

    Example:

        "What are the benefits of renewable energy?"

    returns:

        ["renewable energy"]
    """

    question = _normalize_text(
        research_question
    )

    detected_concepts = []

    for concept in TOPIC_CONCEPT_ALIASES:

        concept_normalized = _normalize_text(
            concept
        )

        if concept_normalized in question:

            detected_concepts.append(
                concept
            )

    return detected_concepts


# =========================================================
# CONCEPT MATCH SCORE
# =========================================================

def calculate_concept_match_score(
    research_question: str,
    document
) -> float:
    """
    Calculate whether the document matches a known
    research concept.

    For renewable energy, for example, "energy" alone
    is NOT enough.
    """

    if document is None:
        return 0.0

    concepts = detect_topic_concepts(
        research_question
    )

    if not concepts:
        return 0.0

    document_text = _normalize_text(
        document.page_content
    )

    best_score = 0.0

    for concept in concepts:

        aliases = TOPIC_CONCEPT_ALIASES.get(
            concept,
            []
        )

        if not aliases:
            continue

        matched_aliases = 0

        for alias in aliases:

            alias_normalized = _normalize_text(
                alias
            )

            if (
                alias_normalized
                and alias_normalized in document_text
            ):

                matched_aliases += 1

        if matched_aliases >= 2:
            score = 1.0

        elif matched_aliases == 1:
            score = 0.75

        else:
            score = 0.0

        best_score = max(
            best_score,
            score
        )

    return best_score


# =========================================================
# TOPIC MATCH SCORE
# =========================================================

def calculate_topic_match_score(
    research_question: str,
    document
) -> float:
    """
    Calculate how strongly a document matches the actual
    topic words in the research question.

    Known concepts receive special handling so that broad
    words do not create false matches.
    """

    if document is None:
        return 0.0

    document_text = _normalize_text(
        document.page_content
    )

    # -----------------------------------------------------
    # First check known concepts.
    # -----------------------------------------------------

    concept_score = calculate_concept_match_score(
        research_question,
        document
    )

    if concept_score > 0:
        return concept_score

    # -----------------------------------------------------
    # Fall back to lexical topic matching.
    # -----------------------------------------------------

    topic_terms = extract_topic_terms(
        research_question
    )

    if not topic_terms:
        return 0.0

    matched_terms = 0

    for term in topic_terms:

        if term in document_text:

            matched_terms += 1

    match_ratio = (
        matched_terms
        / len(topic_terms)
    )

    return min(
        match_ratio,
        1.0
    )


# =========================================================
# METADATA DETECTION
# =========================================================

def count_metadata_hits(text: str) -> int:
    """
    Count obvious PDF metadata terms.
    """

    normalized = _normalize_text(text)

    return sum(
        1
        for term in METADATA_TERMS
        if term in normalized
    )


def is_metadata_noise(document) -> bool:
    """
    Reject chunks that are primarily metadata/noise.
    """

    if document is None:
        return True

    content = (
        document.page_content
        if hasattr(
            document,
            "page_content"
        )
        else ""
    )

    content = content.strip()

    if not content:
        return True

    if len(content) < MIN_CONTENT_LENGTH:
        return True

    metadata_hits = count_metadata_hits(
        content
    )

    if metadata_hits >= MAX_METADATA_HITS:
        return True

    return False


# =========================================================
# DOMAIN DETECTION
# =========================================================

def detect_question_domains(
    research_question: str
):
    """
    Detect domains present in the research question.
    """

    question = _normalize_text(
        research_question
    )

    detected_domains = []

    for domain, keywords in DOMAIN_KEYWORDS.items():

        matches = 0

        for keyword in keywords:

            keyword_normalized = _normalize_text(
                keyword
            )

            if keyword_normalized in question:

                matches += 1

        if matches > 0:

            detected_domains.append(
                domain
            )

    return detected_domains


# =========================================================
# DOMAIN MATCH SCORE
# =========================================================

def calculate_domain_match_score(
    research_question: str,
    document
) -> float:
    """
    Calculate how strongly a document matches the
    domain of the research question.
    """

    if document is None:
        return 0.0

    question_domains = detect_question_domains(
        research_question
    )

    if not question_domains:
        return 0.0

    document_text = _normalize_text(
        document.page_content
    )

    best_score = 0.0

    for domain in question_domains:

        keywords = DOMAIN_KEYWORDS.get(
            domain,
            []
        )

        if not keywords:
            continue

        matched_keywords = []

        for keyword in keywords:

            keyword_normalized = _normalize_text(
                keyword
            )

            if (
                keyword_normalized
                and keyword_normalized in document_text
            ):

                matched_keywords.append(
                    keyword_normalized
                )

        matched_count = len(
            set(matched_keywords)
        )

        if matched_count >= 6:
            score = 1.00

        elif matched_count == 5:
            score = 0.85

        elif matched_count == 4:
            score = 0.75

        elif matched_count == 3:
            score = 0.60

        elif matched_count == 2:
            score = 0.40

        elif matched_count == 1:
            score = 0.20

        else:
            score = 0.0

        best_score = max(
            best_score,
            score
        )

    return best_score


# =========================================================
# DOMAIN RELEVANCE
# =========================================================

def is_domain_relevant(
    research_question: str,
    document
) -> bool:
    """
    Determine whether a document belongs to the same
    domain as the research question.

    Unknown/general domains are allowed through so that
    topic matching can evaluate them.
    """

    if document is None:
        return False

    domains = detect_question_domains(
        research_question
    )

    # Unknown domain.
    if not domains:
        return True

    domain_score = calculate_domain_match_score(
        research_question,
        document
    )

    specific_domains = [
        domain
        for domain in domains
        if domain != "general_ai"
    ]

    if specific_domains:

        document_text = _normalize_text(
            document.page_content
        )

        for domain in specific_domains:

            keywords = DOMAIN_KEYWORDS.get(
                domain,
                []
            )

            domain_matches = sum(
                1
                for keyword in keywords
                if _normalize_text(keyword)
                in document_text
            )

            if domain_matches >= 2:
                return True

        return False

    return (
        domain_score
        >= MIN_DOMAIN_SCORE
    )


# =========================================================
# EVIDENCE QUALITY SCORE
# =========================================================

def calculate_evidence_quality_score(
    result,
    research_question: str
) -> float:
    """
    Calculate evidence quality using:

        similarity
        domain relevance
        topic relevance
        source quality
        content quality
    """

    document = result["document"]

    similarity = float(
        result.get(
            "score",
            0.0
        )
    )

    domain_score = calculate_domain_match_score(
        research_question,
        document
    )

    topic_score = calculate_topic_match_score(
        research_question,
        document
    )

    source = document.metadata.get(
        "source",
        "Unknown source"
    )

    try:

        source_metadata = get_source_metadata(
            source
        )

        source_quality = float(
            source_metadata.get(
                "quality_score",
                0.3
            )
        )

    except Exception:

        source_quality = 0.3

    known_domains = detect_question_domains(
        research_question
    )

    known_concepts = detect_topic_concepts(
        research_question
    )

    # -----------------------------------------------------
    # Known domain
    # -----------------------------------------------------

    if known_domains:

        quality_score = (
            similarity * 0.45
            + domain_score * 0.25
            + topic_score * 0.15
            + source_quality * 0.15
        )

    # -----------------------------------------------------
    # Known concept but no predefined domain.
    #
    # Example:
    # renewable energy
    # climate change
    # etc.
    # -----------------------------------------------------

    elif known_concepts:

        quality_score = (
            similarity * 0.40
            + topic_score * 0.40
            + source_quality * 0.20
        )

    # -----------------------------------------------------
    # Completely unknown/general topic.
    # -----------------------------------------------------

    else:

        quality_score = (
            similarity * 0.45
            + topic_score * 0.35
            + source_quality * 0.20
        )

    content_length = len(
        document.page_content.strip()
    )

    if content_length >= 200:
        quality_score += 0.02

    metadata_hits = count_metadata_hits(
        document.page_content
    )

    quality_score -= (
        0.03 * metadata_hits
    )

    quality_score = max(
        0.0,
        min(
            quality_score,
            1.0
        )
    )

    return quality_score


# =========================================================
# REMOVE DUPLICATES
# =========================================================

def remove_duplicate_results(results):
    """
    Remove duplicate source/page/content combinations.
    """

    unique_results = []

    seen = set()

    for result in results:

        document = result["document"]

        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        page = document.metadata.get(
            "page",
            "Unknown page"
        )

        content = (
            document.page_content
            .strip()
        )

        identifier = (
            source,
            page,
            content
        )

        if identifier not in seen:

            seen.add(
                identifier
            )

            unique_results.append(
                result
            )

    return unique_results


# =========================================================
# SOURCE DIVERSITY
# =========================================================

def apply_source_diversity(
    results,
    max_chunks_per_source=MAX_CHUNKS_PER_SOURCE
):
    """
    Limit the number of evidence chunks coming from
    a single source.
    """

    selected_results = []

    source_counts = {}

    for result in results:

        document = result["document"]

        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        current_count = source_counts.get(
            source,
            0
        )

        if current_count >= max_chunks_per_source:
            continue

        selected_results.append(
            result
        )

        source_counts[source] = (
            current_count + 1
        )

    return selected_results


# =========================================================
# FILTER EVIDENCE
# =========================================================

def filter_evidence_quality(
    results,
    research_question: str
):
    """
    Remove empty, metadata, domain-incorrect and
    topic-incorrect evidence.
    """

    filtered_results = []

    removed_metadata = 0
    removed_empty = 0
    removed_domain = 0

    known_domains = detect_question_domains(
        research_question
    )

    known_concepts = detect_topic_concepts(
        research_question
    )

    for result in results:

        document = result.get(
            "document"
        )

        if document is None:

            removed_empty += 1

            continue

        content = (
            document.page_content
            if hasattr(
                document,
                "page_content"
            )
            else ""
        )

        if not content.strip():

            removed_empty += 1

            continue

        if is_metadata_noise(
            document
        ):

            removed_metadata += 1

            continue

        # -------------------------------------------------
        # DOMAIN FILTER
        # -------------------------------------------------

        if not is_domain_relevant(
            research_question,
            document
        ):

            removed_domain += 1

            continue

        # -------------------------------------------------
        # TOPIC FILTER
        # -------------------------------------------------

        topic_score = calculate_topic_match_score(
            research_question,
            document
        )

        document_text = _normalize_text(
            document.page_content
        )

        topic_terms = extract_topic_terms(
            research_question
        )

        matched_topic_terms = 0

        for term in topic_terms:

            if term in document_text:

                matched_topic_terms += 1

        # -------------------------------------------------
        # KNOWN CONCEPT
        # -------------------------------------------------

        if known_concepts:

            concept_score = calculate_concept_match_score(
                research_question,
                document
            )

            # A known concept must actually appear in
            # the evidence or be represented by one of
            # its meaningful aliases.
            if concept_score <= 0:

                removed_domain += 1

                continue

        # -------------------------------------------------
        # KNOWN DOMAIN
        # -------------------------------------------------

        elif known_domains:

            # Domain documents must have at least one
            # meaningful topic connection.
            if matched_topic_terms < MIN_TOPIC_MATCHES:

                removed_domain += 1

                continue

        # -------------------------------------------------
        # UNKNOWN GENERAL TOPIC
        # -------------------------------------------------

        else:

            if (
                matched_topic_terms
                < MIN_TOPIC_MATCHES
            ):

                removed_domain += 1

                continue

            if (
                topic_score
                < MIN_GENERAL_TOPIC_SCORE
            ):

                removed_domain += 1

                continue

        # -------------------------------------------------
        # STORE SCORES
        # -------------------------------------------------

        result_copy = dict(
            result
        )

        result_copy[
            "_domain_score"
        ] = calculate_domain_match_score(
            research_question,
            document
        )

        result_copy[
            "_topic_score"
        ] = topic_score

        result_copy[
            "_concept_score"
        ] = calculate_concept_match_score(
            research_question,
            document
        )

        result_copy[
            "_quality_score"
        ] = calculate_evidence_quality_score(
            result_copy,
            research_question
        )

        filtered_results.append(
            result_copy
        )

    return (
        filtered_results,
        removed_metadata,
        removed_empty,
        removed_domain
    )


# =========================================================
# BUILD EVIDENCE CONTEXT
# =========================================================

def build_evidence_context(
    evidence_items
):
    """
    Convert Evidence objects into Research Agent context.
    """

    evidence_parts = []

    for index, evidence in enumerate(
        evidence_items,
        start=1
    ):

        source_metadata = get_source_metadata(
            evidence.source
        )

        evidence_parts.append(
            f"""
EVIDENCE {index}

Evidence ID: {evidence.evidence_id}

Source: {evidence.source}

Source Type: {source_metadata["source_type"]}

Source Quality Score: {source_metadata["quality_score"]}

Page: {evidence.page}

Similarity Score: {evidence.similarity_score:.4f}

Content:
{evidence.content}
"""
        )

    return "\n".join(
        evidence_parts
    )


# =========================================================
# CREATE FALLBACK RESEARCH QUESTIONS
# =========================================================

def create_fallback_research_questions(
    research_question: str
):
    """
    Create deterministic retrieval questions when the
    Gemini Planner is unavailable.
    """

    domains = detect_question_domains(
        research_question
    )

    # =====================================================
    # HEALTHCARE
    # =====================================================

    if "healthcare" in domains:

        return [

            (
                "What are the benefits of artificial "
                "intelligence in healthcare related to: "
                f"{research_question}"
            ),

            (
                "What are the risks and limitations of "
                "artificial intelligence in healthcare "
                "related to: "
                f"{research_question}"
            ),

            (
                "How is artificial intelligence used for "
                "diagnosis, treatment, and patient care "
                "related to: "
                f"{research_question}"
            ),

            (
                "What are the measurable clinical outcomes "
                "of AI applications in healthcare related "
                "to: "
                f"{research_question}"
            ),

            (
                "How does AI improve healthcare efficiency, "
                "cost effectiveness, and resource allocation "
                "related to: "
                f"{research_question}"
            ),

            (
                "What ethical concerns including bias, "
                "fairness, transparency, and accountability "
                "are associated with AI in healthcare related "
                "to: "
                f"{research_question}"
            ),

            (
                "What privacy, cybersecurity, safety, and "
                "regulatory risks are associated with AI "
                "in healthcare related to: "
                f"{research_question}"
            ),

            (
                "How does AI affect patient autonomy, shared "
                "decision making, and the patient provider "
                "relationship related to: "
                f"{research_question}"
            ),

            (
                "What challenges exist when implementing "
                "AI in healthcare related to: "
                f"{research_question}"
            ),

            (
                "What infrastructure, data, training, and "
                "human requirements are necessary for AI "
                "in healthcare related to: "
                f"{research_question}"
            ),
        ]

    # =====================================================
    # CUSTOMER SUPPORT
    # =====================================================

    if "customer_support" in domains:

        return [

            (
                "What are the benefits of AI in customer "
                "support related to: "
                f"{research_question}"
            ),

            (
                "What customer support tasks can AI "
                "automate related to: "
                f"{research_question}"
            ),

            (
                "How can AI improve response time and "
                "customer satisfaction related to: "
                f"{research_question}"
            ),

            (
                "What measurable efficiency improvements "
                "can AI provide in customer support related "
                "to: "
                f"{research_question}"
            ),

            (
                "What risks and limitations exist when "
                "using AI in customer support related to: "
                f"{research_question}"
            ),

            (
                "What best practices improve AI customer "
                "support implementation related to: "
                f"{research_question}"
            ),

            (
                "What human and technical requirements are "
                "needed for AI customer support related to: "
                f"{research_question}"
            ),
        ]

    # =====================================================
    # FINANCE
    # =====================================================

    if "finance" in domains:

        return [

            (
                "What are the benefits of artificial "
                "intelligence in finance related to: "
                f"{research_question}"
            ),

            (
                "What are the risks and limitations of "
                "AI in finance related to: "
                f"{research_question}"
            ),

            (
                "How is AI used in banking, fraud detection, "
                "investment, and financial services related "
                "to: "
                f"{research_question}"
            ),

            (
                "What regulatory and ethical challenges "
                "exist for AI in finance related to: "
                f"{research_question}"
            ),

            (
                "What measurable business outcomes result "
                "from AI adoption in finance related to: "
                f"{research_question}"
            ),
        ]

    # =====================================================
    # EDUCATION
    # =====================================================

    if "education" in domains:

        return [

            (
                "What are the benefits of artificial "
                "intelligence in education related to: "
                f"{research_question}"
            ),

            (
                "What are the risks and limitations of "
                "AI in education related to: "
                f"{research_question}"
            ),

            (
                "How does AI support students, teachers, "
                "learning, and assessment related to: "
                f"{research_question}"
            ),

            (
                "What ethical and privacy concerns exist "
                "with AI in education related to: "
                f"{research_question}"
            ),

            (
                "What challenges exist when implementing "
                "AI in education related to: "
                f"{research_question}"
            ),
        ]

    # =====================================================
    # SOFTWARE
    # =====================================================

    if "software" in domains:

        return [

            (
                "What are the benefits of AI in software "
                "development related to: "
                f"{research_question}"
            ),

            (
                "How can AI improve programming, coding, "
                "testing, and debugging related to: "
                f"{research_question}"
            ),

            (
                "What risks and limitations exist when "
                "using AI for software development related "
                "to: "
                f"{research_question}"
            ),

            (
                "What security, reliability, and quality "
                "issues can arise from AI-generated code "
                "related to: "
                f"{research_question}"
            ),

            (
                "What human skills and requirements are "
                "needed when using AI in software development "
                "related to: "
                f"{research_question}"
            ),
        ]

    # =====================================================
    # GENERAL FALLBACK
    # =====================================================

    return [

        (
            "What specific technologies, methods, or "
            "approaches are relevant to: "
            f"{research_question}"
        ),

        (
            "What benefits and positive outcomes are "
            "associated with: "
            f"{research_question}"
        ),

        (
            "How does the topic improve efficiency, "
            "performance, cost, sustainability, or outcomes "
            "related to: "
            f"{research_question}"
        ),

        (
            "What measurable benefits or improvements "
            "can be identified for: "
            f"{research_question}"
        ),

        (
            "What limitations, risks, or challenges should "
            "be considered for: "
            f"{research_question}"
        ),

        (
            "What best practices can improve successful "
            "implementation of: "
            f"{research_question}"
        ),

        (
            "What data, infrastructure, and human "
            "requirements are necessary for: "
            f"{research_question}"
        ),
    ]


# =========================================================
# MAIN RAG RESEARCH FUNCTION
# =========================================================

def research_with_rag(
    research_question: str,
    top_k: int = 3,
    research_questions=None
):
    """
    Main RAG research pipeline.
    """

    print(
        "\nLoading persistent vector store..."
    )

    vector_store = VectorStore()

    vector_store.load()

    # =====================================================
    # 1. DETECT DOMAIN
    # =====================================================

    detected_domains = detect_question_domains(
        research_question
    )

    detected_concepts = detect_topic_concepts(
        research_question
    )

    print(
        "\nDetected domain keywords:"
    )

    if detected_domains:

        print(
            ", ".join(
                detected_domains
            )
        )

    else:

        print(
            "general"
        )

    if detected_concepts:

        print(
            "Detected topic concepts:"
        )

        print(
            ", ".join(
                detected_concepts
            )
        )

    else:

        topic_terms = extract_topic_terms(
            research_question
        )

        print(
            "Detected topic terms:"
        )

        print(
            ", ".join(
                topic_terms
            )
        )

    # =====================================================
    # 2. BUILD RETRIEVAL QUESTIONS
    # =====================================================

    if research_questions:

        retrieval_questions = list(
            dict.fromkeys(
                [research_question]
                + research_questions
            )
        )

    else:

        fallback_questions = (
            create_fallback_research_questions(
                research_question
            )
        )

        retrieval_questions = list(
            dict.fromkeys(
                [research_question]
                + fallback_questions
            )
        )

    print(
        f"\nRetrieval questions: "
        f"{len(retrieval_questions)}"
    )

    # =====================================================
    # 3. RETRIEVE EVIDENCE
    # =====================================================

    all_results = []

    for number, question in enumerate(
        retrieval_questions,
        start=1
    ):

        print(
            f"\n[{number}/{len(retrieval_questions)}] "
            f"Searching: {question}"
        )

        try:

            results = vector_store.search(
                question,
                top_k=top_k
            )

        except Exception as error:

            print(
                f"Vector search failed: {error}"
            )

            continue

        retrieved_count = len(
            results
        )

        similarity_results = [

            result

            for result in results

            if float(
                result.get(
                    "score",
                    0.0
                )
            ) >= MIN_SIMILARITY_SCORE
        ]

        relevant_results = []

        for result in similarity_results:

            document = result.get(
                "document"
            )

            if is_domain_relevant(
                research_question,
                document
            ):

                relevant_results.append(
                    result
                )

        print(
            f"Retrieved: {retrieved_count} | "
            f"Similarity relevant: "
            f"{len(similarity_results)} | "
            f"Domain relevant: "
            f"{len(relevant_results)}"
        )

        all_results.extend(
            relevant_results
        )

    # =====================================================
    # 4. REMOVE DUPLICATES
    # =====================================================

    unique_results = remove_duplicate_results(
        all_results
    )

    print(
        f"\nUnique retrieval results before "
        f"quality filtering: "
        f"{len(unique_results)}"
    )

    # =====================================================
    # 5. FILTER QUALITY
    # =====================================================

    (
        quality_results,
        removed_metadata,
        removed_empty,
        removed_domain
    ) = filter_evidence_quality(
        unique_results,
        research_question
    )

    if removed_metadata > 0:

        print(
            f"\nMetadata/noise chunks removed: "
            f"{removed_metadata}"
        )

    if removed_empty > 0:

        print(
            f"Empty chunks removed: "
            f"{removed_empty}"
        )

    if removed_domain > 0:

        print(
            f"Domain/topic-irrelevant chunks removed: "
            f"{removed_domain}"
        )

    # =====================================================
    # 6. RANK EVIDENCE
    # =====================================================

    quality_results = sorted(
        quality_results,
        key=lambda result: float(
            result.get(
                "_quality_score",
                0.0
            )
        ),
        reverse=True
    )

    # =====================================================
    # 7. APPLY SOURCE DIVERSITY
    # =====================================================

    before_diversity = len(
        quality_results
    )

    quality_results = apply_source_diversity(
        quality_results,
        MAX_CHUNKS_PER_SOURCE
    )

    after_diversity = len(
        quality_results
    )

    if before_diversity != after_diversity:

        print(
            "\nSource diversity applied:"
        )

        print(
            f"Evidence before diversity: "
            f"{before_diversity}"
        )

        print(
            f"Evidence after diversity: "
            f"{after_diversity}"
        )

        print(
            f"Maximum chunks per source: "
            f"{MAX_CHUNKS_PER_SOURCE}"
        )

    # =====================================================
    # 8. LIMIT EVIDENCE
    # =====================================================

    quality_results = quality_results[
        :MAX_EVIDENCE_CHUNKS
    ]

    # =====================================================
    # 9. HANDLE NO EVIDENCE
    # =====================================================

    if not quality_results:

        print(
            "\nNo domain/topic-relevant evidence "
            "passed the quality filters."
        )

        return {

            "research_question":
                research_question,

            "retrieval_questions":
                retrieval_questions,

            "retrieved_evidence":
                "",

            "evidence_items":
                [],

            "analysis": {

                "research_question":
                    research_question,

                "key_findings":
                    [],

                "supporting_evidence":
                    [],

                "limitations": [

                    "No retrieved evidence matched "
                    "the research topic.",

                    "The current document collection "
                    "may not contain sufficient evidence "
                    "for this research question."
                ],

                "confidence":
                    "Low",

                "analysis_status":
                    "insufficient_relevant_evidence"
            }
        }

    print(
        f"\nUnique evidence chunks selected: "
        f"{len(quality_results)}"
    )

    # =====================================================
    # 10. DISPLAY QUALITY RANKING
    # =====================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "EVIDENCE QUALITY RANKING"
    )

    print(
        "=" * 60
    )

    for rank, result in enumerate(
        quality_results,
        start=1
    ):

        document = result["document"]

        source = document.metadata.get(
            "source",
            "Unknown source"
        )

        page = document.metadata.get(
            "page",
            "Unknown page"
        )

        similarity = float(
            result.get(
                "score",
                0.0
            )
        )

        domain_score = float(
            result.get(
                "_domain_score",
                0.0
            )
        )

        topic_score = float(
            result.get(
                "_topic_score",
                0.0
            )
        )

        concept_score = float(
            result.get(
                "_concept_score",
                0.0
            )
        )

        quality_score = float(
            result.get(
                "_quality_score",
                0.0
            )
        )

        print(
            f"\nRank {rank}"
        )

        print(
            f"Source: {source}"
        )

        print(
            f"Page: {page}"
        )

        print(
            f"Similarity: {similarity:.4f}"
        )

        print(
            f"Domain Match: {domain_score:.4f}"
        )

        print(
            f"Topic Match: {topic_score:.4f}"
        )

        print(
            f"Concept Match: {concept_score:.4f}"
        )

        print(
            f"Evidence Quality: "
            f"{quality_score:.4f}"
        )

    # =====================================================
    # 11. BUILD STRUCTURED EVIDENCE
    # =====================================================

    evidence_items = build_evidence(
        quality_results
    )

    print(
        f"\nStructured evidence objects created: "
        f"{len(evidence_items)}"
    )

    # =====================================================
    # 12. DISPLAY SOURCE QUALITY
    # =====================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "SOURCE QUALITY"
    )

    print(
        "=" * 60
    )

    for evidence_item in evidence_items:

        source_metadata = get_source_metadata(
            evidence_item.source
        )

        print(
            f"\nEvidence ID: "
            f"{evidence_item.evidence_id}"
        )

        print(
            f"Source: "
            f"{evidence_item.source}"
        )

        print(
            f"Source Type: "
            f"{source_metadata['source_type']}"
        )

        print(
            f"Source Quality Score: "
            f"{source_metadata['quality_score']}"
        )

        print(
            f"Similarity Score: "
            f"{evidence_item.similarity_score:.4f}"
        )

    # =====================================================
    # 13. BUILD EVIDENCE CONTEXT
    # =====================================================

    evidence = build_evidence_context(
        evidence_items
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "RETRIEVED EVIDENCE"
    )

    print(
        "=" * 60
    )

    print(
        evidence
    )

    # =====================================================
    # 14. RESEARCH AGENT
    # =====================================================

    print(
        "\nAnalyzing evidence with Research Agent..."
    )

    try:

        analysis = analyze_evidence(
            research_question,
            evidence
        )

    except Exception as error:

        print(
            f"Research Agent failed: {error}"
        )

        analysis = {

            "research_question":
                research_question,

            "key_findings":
                [],

            "supporting_evidence":
                [],

            "limitations": [

                "Research Agent failed to analyze "
                "the retrieved evidence."
            ],

            "confidence":
                "Low",

            "analysis_status":
                "research_agent_failed"
        }

    # =====================================================
    # 15. RETURN RESULT
    # =====================================================

    return {

        "research_question":
            research_question,

        "retrieval_questions":
            retrieval_questions,

        "retrieved_evidence":
            evidence,

        "evidence_items": [

            evidence_item.to_dict()

            for evidence_item
            in evidence_items
        ],

        "analysis":
            analysis
    }


# =========================================================
# EVALUATION RETRIEVAL FUNCTION
# =========================================================

def retrieve_research_evidence(
    research_question: str,
    top_k: int = 3
):
    """
    Lightweight retrieval function used by the evaluation
    framework.
    """

    print(
        f"\nEvaluation retrieval: {research_question}"
    )

    # =====================================================
    # 1. LOAD VECTOR STORE
    # =====================================================

    vector_store = VectorStore()

    vector_store.load()

    # =====================================================
    # 2. VECTOR SEARCH
    # =====================================================

    try:

        results = vector_store.search(
            research_question,
            top_k=top_k
        )

    except Exception as error:

        print(
            f"Evaluation vector search failed: {error}"
        )

        return []

    if not results:
        return []

    # =====================================================
    # 3. SIMILARITY FILTERING
    # =====================================================

    similarity_results = [

        result

        for result in results

        if float(
            result.get(
                "score",
                0.0
            )
        ) >= MIN_SIMILARITY_SCORE
    ]

    # =====================================================
    # 4. DOMAIN FILTERING
    # =====================================================

    domain_results = []

    for result in similarity_results:

        document = result.get(
            "document"
        )

        if is_domain_relevant(
            research_question,
            document
        ):

            domain_results.append(
                result
            )

    # =====================================================
    # 5. REMOVE DUPLICATES
    # =====================================================

    unique_results = remove_duplicate_results(
        domain_results
    )

    # =====================================================
    # 6. QUALITY FILTERING
    # =====================================================

    (
        quality_results,
        _removed_metadata,
        _removed_empty,
        _removed_domain
    ) = filter_evidence_quality(
        unique_results,
        research_question
    )

    # =====================================================
    # 7. RANK BY QUALITY
    # =====================================================

    quality_results = sorted(
        quality_results,
        key=lambda result: float(
            result.get(
                "_quality_score",
                0.0
            )
        ),
        reverse=True
    )

    # =====================================================
    # 8. SOURCE DIVERSITY
    # =====================================================

    quality_results = apply_source_diversity(
        quality_results,
        MAX_CHUNKS_PER_SOURCE
    )

    # =====================================================
    # 9. LIMIT RESULTS
    # =====================================================

    quality_results = quality_results[
        :MAX_EVIDENCE_CHUNKS
    ]

    # =====================================================
    # 10. CONVERT TO EVALUATION FORMAT
    # =====================================================

    evaluation_results = []

    for result in quality_results:

        document = result.get(
            "document"
        )

        if document is None:
            continue

        evaluation_results.append({

            "content":
                document.page_content,

            "source":
                document.metadata.get(
                    "source",
                    "Unknown source"
                ),

            "page":
                document.metadata.get(
                    "page",
                    "Unknown page"
                ),

            "score":
                float(
                    result.get(
                        "score",
                        0.0
                    )
                ),

            "domain_score":
                float(
                    result.get(
                        "_domain_score",
                        0.0
                    )
                ),

            "topic_score":
                float(
                    result.get(
                        "_topic_score",
                        0.0
                    )
                ),

            "concept_score":
                float(
                    result.get(
                        "_concept_score",
                        0.0
                    )
                ),

            "quality_score":
                float(
                    result.get(
                        "_quality_score",
                        0.0
                    )
                )
        })

    return evaluation_results


# =========================================================
# DIRECT TEST
# =========================================================

if __name__ == "__main__":

    question = (
        "What are the benefits and risks of "
        "artificial intelligence in healthcare?"
    )

    result = research_with_rag(
        research_question=question,
        research_questions=None,
        top_k=3
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "RAG RESEARCH RESULT"
    )

    print(
        "=" * 60
    )

    print(
        json.dumps(
            result["analysis"],
            indent=4
        )
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "STRUCTURED EVIDENCE"
    )

    print(
        "=" * 60
    )

    print(
        json.dumps(
            result["evidence_items"],
            indent=4
        )
    )