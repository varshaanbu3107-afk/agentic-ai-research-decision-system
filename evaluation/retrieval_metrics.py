"""
Retrieval Metrics Evaluation
============================

Evaluates the RAG retrieval system against a golden-question
dataset.

Metrics:
- Evidence Coverage
- Retrieval Recall
- Retrieval Precision
- Retrieval F1

The evaluation uses:
    app.rag.research_retriever.retrieve_research_evidence()
"""

from pathlib import Path
import json
import re
import sys


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT)
    )


# ============================================================
# IMPORT RETRIEVER
# ============================================================

from app.rag.research_retriever import (
    retrieve_research_evidence
)


# ============================================================
# DATASET PATH
# ============================================================

DATASET_PATH = (
    Path(__file__).resolve().parent
    / "golden_questions.json"
)


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text)

    text = text.lower()

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


# ============================================================
# GET EVIDENCE CONTENT
# ============================================================

def get_evidence_content(evidence):

    if isinstance(
        evidence,
        dict
    ):

        possible_fields = [
            "content",
            "text",
            "page_content",
            "evidence",
            "chunk"
        ]

        for field in possible_fields:

            if field in evidence:

                value = evidence[field]

                if value:
                    return str(value)

    if hasattr(
        evidence,
        "page_content"
    ):

        return str(
            evidence.page_content
        )

    return str(evidence)


# ============================================================
# LOAD GOLDEN DATASET
# ============================================================

def load_dataset():

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{DATASET_PATH}"
        )

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # golden_questions.json has this structure:
    #
    # {
    #     "questions": [
    #         {...},
    #         {...}
    #     ]
    # }
    #
    # Therefore return data["questions"].
    # --------------------------------------------------------

    if isinstance(
        data,
        dict
    ):

        if "questions" not in data:

            raise ValueError(
                "Dataset JSON must contain "
                "'questions'."
            )

        data = data["questions"]

    if not isinstance(
        data,
        list
    ):

        raise ValueError(
            "Dataset 'questions' must be a list."
        )

    return data


# ============================================================
# VALIDATE DATASET
# ============================================================

def validate_dataset(
    dataset
):

    for index, item in enumerate(
        dataset,
        start=1
    ):

        if not isinstance(
            item,
            dict
        ):

            raise ValueError(
                f"Question {index} "
                f"must be an object."
            )

        if "question" not in item:

            raise ValueError(
                f"Question {index} "
                f"is missing 'question'."
            )

        if "expected_concepts" not in item:

            raise ValueError(
                f"Question {index} "
                f"is missing 'expected_concepts'."
            )

        if not isinstance(
            item["expected_concepts"],
            list
        ):

            raise ValueError(
                f"Question {index}: "
                f"'expected_concepts' "
                f"must be a list."
            )

        if not item["expected_concepts"]:

            raise ValueError(
                f"Question {index}: "
                f"'expected_concepts' "
                f"cannot be empty."
            )

    print(
        "Validation   : PASSED"
    )


# ============================================================
# CONCEPT MATCHING
# ============================================================

def concept_matches(
    concept,
    evidence_text
):

    concept_normalized = normalize_text(
        concept
    )

    evidence_normalized = normalize_text(
        evidence_text
    )

    if not concept_normalized:
        return False

    # --------------------------------------------------------
    # Direct phrase match
    # --------------------------------------------------------

    if concept_normalized in evidence_normalized:

        return True

    # --------------------------------------------------------
    # Word-level matching
    # --------------------------------------------------------

    concept_words = (
        concept_normalized
        .split()
    )

    evidence_words = set(
        evidence_normalized.split()
    )

    if len(concept_words) == 1:

        return concept_words[0] in evidence_words

    # --------------------------------------------------------
    # All important words present
    # --------------------------------------------------------

    matched_words = sum(
        1
        for word in concept_words
        if word in evidence_words
    )

    return (
        matched_words
        == len(concept_words)
    )


# ============================================================
# EVALUATE QUESTION
# ============================================================

def evaluate_question(
    question,
    expected_concepts
):

    print(
        "\n------------------------------------------------------------"
    )

    print(
        f"QUESTION: {question}"
    )

    print(
        "------------------------------------------------------------"
    )

    evidence = retrieve_research_evidence(
        question,
        top_k=3
    )

    retrieved_chunks = len(
        evidence
    )

    # --------------------------------------------------------
    # Combine evidence text
    # --------------------------------------------------------

    evidence_texts = []

    for item in evidence:

        evidence_texts.append(
            get_evidence_content(item)
        )

    combined_text = "\n".join(
        evidence_texts
    )

    # --------------------------------------------------------
    # Match expected concepts
    # --------------------------------------------------------

    matched_concepts = []
    missing_concepts = []

    for concept in expected_concepts:

        if concept_matches(
            concept,
            combined_text
        ):

            matched_concepts.append(
                concept
            )

        else:

            missing_concepts.append(
                concept
            )

    # --------------------------------------------------------
    # Relevant chunks
    #
    # A chunk is considered relevant when it contains
    # at least one expected concept.
    # --------------------------------------------------------

    relevant_chunks = 0

    for evidence_item in evidence:

        chunk_text = get_evidence_content(
            evidence_item
        )

        chunk_relevant = False

        for concept in expected_concepts:

            if concept_matches(
                concept,
                chunk_text
            ):

                chunk_relevant = True
                break

        if chunk_relevant:

            relevant_chunks += 1

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    total_concepts = len(
        expected_concepts
    )

    matched_count = len(
        matched_concepts
    )

    # Evidence coverage
    if total_concepts > 0:

        evidence_coverage = (
            matched_count
            / total_concepts
        )

    else:

        evidence_coverage = 0.0

    # Retrieval recall
    retrieval_recall = (
        evidence_coverage
    )

    # Retrieval precision
    if retrieved_chunks > 0:

        retrieval_precision = (
            relevant_chunks
            / retrieved_chunks
        )

    else:

        retrieval_precision = 0.0

    # Retrieval F1
    if (
        retrieval_precision
        + retrieval_recall
    ) > 0:

        retrieval_f1 = (
            2
            * retrieval_precision
            * retrieval_recall
            / (
                retrieval_precision
                + retrieval_recall
            )
        )

    else:

        retrieval_f1 = 0.0

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    print(
        f"Retrieved chunks       : "
        f"{retrieved_chunks}"
    )

    print(
        f"Expected concepts      : "
        f"{total_concepts}"
    )

    print(
        f"Matched concepts       : "
        f"{matched_count}"
    )

    print(
        f"Relevant chunks        : "
        f"{relevant_chunks}"
    )

    print(
        f"Missing concepts       : "
        f"{len(missing_concepts)}"
    )

    print(
        f"Evidence coverage     : "
        f"{evidence_coverage * 100:.2f}%"
    )

    print(
        f"Retrieval recall      : "
        f"{retrieval_recall * 100:.2f}%"
    )

    print(
        f"Retrieval precision   : "
        f"{retrieval_precision * 100:.2f}%"
    )

    print(
        f"Retrieval F1          : "
        f"{retrieval_f1 * 100:.2f}%"
    )

    print(
        f"Matched: {matched_concepts}"
    )

    print(
        f"Missing: {missing_concepts}"
    )

    return {
        "question": question,
        "expected_concepts": expected_concepts,
        "matched_concepts": matched_concepts,
        "missing_concepts": missing_concepts,
        "retrieved_chunks": retrieved_chunks,
        "relevant_chunks": relevant_chunks,
        "evidence_coverage": evidence_coverage,
        "retrieval_recall": retrieval_recall,
        "retrieval_precision": retrieval_precision,
        "retrieval_f1": retrieval_f1
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    averages,
    status
):

    output_path = (
        Path(__file__).resolve().parent
        / "evaluation_results.json"
    )

    output_data = {

        "questions_evaluated": len(
            results
        ),

        "average_evidence_coverage": (
            averages[
                "evidence_coverage"
            ]
        ),

        "average_retrieval_recall": (
            averages[
                "retrieval_recall"
            ]
        ),

        "average_retrieval_precision": (
            averages[
                "retrieval_precision"
            ]
        ),

        "average_retrieval_f1": (
            averages[
                "retrieval_f1"
            ]
        ),

        "overall_status": status,

        "question_results": results
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output_data,
            file,
            indent=2
        )

    print(
        "\nEvaluation results saved:"
    )

    print(
        output_path
    )

    print(
        "============================================================"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "============================================================"
    )

    print(
        "RETRIEVAL METRICS TEST"
    )

    print(
        "============================================================"
    )

    print(
        f"Dataset path : {DATASET_PATH}"
    )

    dataset = load_dataset()

    print(
        f"Questions    : {len(dataset)}"
    )

    validate_dataset(
        dataset
    )

    results = []

    # --------------------------------------------------------
    # Evaluate every question
    # --------------------------------------------------------

    for item in dataset:

        question = item[
            "question"
        ]

        expected_concepts = item[
            "expected_concepts"
        ]

        result = evaluate_question(
            question,
            expected_concepts
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # Calculate averages
    # --------------------------------------------------------

    if results:

        average_evidence_coverage = (
            sum(
                result[
                    "evidence_coverage"
                ]
                for result in results
            )
            / len(results)
        )

        average_recall = (
            sum(
                result[
                    "retrieval_recall"
                ]
                for result in results
            )
            / len(results)
        )

        average_precision = (
            sum(
                result[
                    "retrieval_precision"
                ]
                for result in results
            )
            / len(results)
        )

        average_f1 = (
            sum(
                result[
                    "retrieval_f1"
                ]
                for result in results
            )
            / len(results)
        )

    else:

        average_evidence_coverage = 0.0
        average_recall = 0.0
        average_precision = 0.0
        average_f1 = 0.0

    averages = {

        "evidence_coverage":
            average_evidence_coverage,

        "retrieval_recall":
            average_recall,

        "retrieval_precision":
            average_precision,

        "retrieval_f1":
            average_f1
    }

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n\n============================================================"
    )

    print(
        "RETRIEVAL EVALUATION SUMMARY"
    )

    print(
        "============================================================"
    )

    print(
        f"Questions evaluated        : "
        f"{len(results)}"
    )

    print(
        f"Average evidence coverage : "
        f"{average_evidence_coverage * 100:.2f}%"
    )

    print(
        f"Average retrieval recall  : "
        f"{average_recall * 100:.2f}%"
    )

    print(
        f"Average retrieval precision: "
        f"{average_precision * 100:.2f}%"
    )

    print(
        f"Average retrieval F1      : "
        f"{average_f1 * 100:.2f}%"
    )

    print(
        "\nQuestion-wise results:"
    )

    print(
        "------------------------------------------------------------"
    )

    for index, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{index}. "
            f"Coverage="
            f"{result['evidence_coverage'] * 100:.2f}% | "
            f"Precision="
            f"{result['retrieval_precision'] * 100:.2f}% | "
            f"Recall="
            f"{result['retrieval_recall'] * 100:.2f}% | "
            f"F1="
            f"{result['retrieval_f1'] * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Quality assessment
    # --------------------------------------------------------

    if average_f1 >= 0.90:

        status = "EXCELLENT"

    elif average_f1 >= 0.75:

        status = "GOOD"

    elif average_f1 >= 0.60:

        status = "FAIR"

    else:

        status = "POOR"

    print(
        "\n\n============================================================"
    )

    print(
        "RETRIEVAL QUALITY ASSESSMENT"
    )

    print(
        "============================================================"
    )

    print(
        f"Overall retrieval status : "
        f"{status}"
    )

    print(
        "============================================================"
    )

    # --------------------------------------------------------
    # Save JSON results
    # --------------------------------------------------------

    save_results(
        results,
        averages,
        status
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

