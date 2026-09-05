def calculate_average_quality(evidence_items):
    """
    Calculate the average evidence quality score.

    Expected format:

    [
        {
            "quality": 0.76
        },
        {
            "quality": 0.74
        }
    ]

    Quality should be between 0 and 1.
    """

    if not evidence_items:
        return 0.0

    quality_scores = []

    for item in evidence_items:
        quality = item.get("quality", 0.0)

        try:
            quality = float(quality)
        except (TypeError, ValueError):
            quality = 0.0

        quality = max(0.0, min(1.0, quality))

        quality_scores.append(quality)

    average = sum(quality_scores) / len(quality_scores)

    return round(average * 100, 2)


def calculate_relevance_rate(evidence_items):
    """
    Calculate the percentage of evidence considered relevant.

    An evidence item is relevant when its relationship is:

    - supports
    - partially_supports
    """

    if not evidence_items:
        return 0.0

    relevant_count = 0

    for item in evidence_items:

        relationship = str(
            item.get("relationship", "")
        ).lower().strip()

        if relationship in {
            "supports",
            "partially_supports",
            "partially supports",
        }:
            relevant_count += 1

    rate = (
        relevant_count / len(evidence_items)
    ) * 100

    return round(rate, 2)


def calculate_support_rate(evidence_items):
    """
    Calculate the percentage of evidence that directly supports
    the research question.
    """

    if not evidence_items:
        return 0.0

    supporting_count = 0

    for item in evidence_items:

        relationship = str(
            item.get("relationship", "")
        ).lower().strip()

        if relationship == "supports":
            supporting_count += 1

    rate = (
        supporting_count / len(evidence_items)
    ) * 100

    return round(rate, 2)


def calculate_partial_support_rate(evidence_items):
    """
    Calculate the percentage of evidence that partially supports
    the research question.
    """

    if not evidence_items:
        return 0.0

    partial_count = 0

    for item in evidence_items:

        relationship = str(
            item.get("relationship", "")
        ).lower().strip()

        if relationship in {
            "partially_supports",
            "partially supports",
        }:
            partial_count += 1

    rate = (
        partial_count / len(evidence_items)
    ) * 100

    return round(rate, 2)


def calculate_contradiction_rate(evidence_items):
    """
    Calculate the percentage of evidence that contradicts
    the research question.
    """

    if not evidence_items:
        return 0.0

    contradiction_count = 0

    for item in evidence_items:

        relationship = str(
            item.get("relationship", "")
        ).lower().strip()

        if relationship == "contradicts":
            contradiction_count += 1

    rate = (
        contradiction_count / len(evidence_items)
    ) * 100

    return round(rate, 2)


def evaluate_evidence(evidence_items):
    """
    Calculate all evidence-level metrics.
    """

    return {
        "total_evidence": len(evidence_items),
        "average_quality": calculate_average_quality(
            evidence_items
        ),
        "relevance_rate": calculate_relevance_rate(
            evidence_items
        ),
        "support_rate": calculate_support_rate(
            evidence_items
        ),
        "partial_support_rate": calculate_partial_support_rate(
            evidence_items
        ),
        "contradiction_rate": calculate_contradiction_rate(
            evidence_items
        ),
    }


if __name__ == "__main__":

    # Standalone test data
    evidence_items = [
        {
            "evidence_id": "E001",
            "relationship": "supports",
            "quality": 0.76,
        },
        {
            "evidence_id": "E002",
            "relationship": "supports",
            "quality": 0.74,
        },
        {
            "evidence_id": "E003",
            "relationship": "supports",
            "quality": 0.73,
        },
        {
            "evidence_id": "E004",
            "relationship": "supports",
            "quality": 0.72,
        },
        {
            "evidence_id": "E005",
            "relationship": "partially_supports",
            "quality": 0.70,
        },
    ]

    results = evaluate_evidence(evidence_items)

    print("=" * 60)
    print("EVIDENCE METRICS TEST")
    print("=" * 60)

    print(
        f"Total evidence          : "
        f"{results['total_evidence']}"
    )

    print(
        f"Average evidence quality: "
        f"{results['average_quality']}%"
    )

    print(
        f"Evidence relevance      : "
        f"{results['relevance_rate']}%"
    )

    print(
        f"Support rate            : "
        f"{results['support_rate']}%"
    )

    print(
        f"Partial support rate    : "
        f"{results['partial_support_rate']}%"
    )

    print(
        f"Contradiction rate      : "
        f"{results['contradiction_rate']}%"
    )

    print("=" * 60)