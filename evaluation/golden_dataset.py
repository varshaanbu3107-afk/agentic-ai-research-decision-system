import json
from pathlib import Path


DATASET_PATH = Path(__file__).parent / "golden_questions.json"


def load_golden_dataset():
    """
    Load the evaluation questions from golden_questions.json.

    Returns:
        list: A list of evaluation question dictionaries.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Golden dataset not found: {DATASET_PATH}"
        )

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list):
        raise ValueError(
            "Golden dataset must contain a JSON list."
        )

    return dataset


def validate_golden_dataset(dataset):
    """
    Validate the structure of the golden dataset.
    """

    required_fields = {
        "question",
        "expected_domains",
        "expected_evidence",
        "expected_decision",
    }

    for index, item in enumerate(dataset, start=1):

        if not isinstance(item, dict):
            raise ValueError(
                f"Dataset item {index} must be a dictionary."
            )

        missing_fields = required_fields - item.keys()

        if missing_fields:
            raise ValueError(
                f"Dataset item {index} is missing fields: "
                f"{sorted(missing_fields)}"
            )

        if not isinstance(item["question"], str):
            raise ValueError(
                f"Dataset item {index}: question must be a string."
            )

        if not isinstance(item["expected_domains"], list):
            raise ValueError(
                f"Dataset item {index}: expected_domains must be a list."
            )

        if not isinstance(item["expected_evidence"], list):
            raise ValueError(
                f"Dataset item {index}: expected_evidence must be a list."
            )

        if not isinstance(item["expected_decision"], str):
            raise ValueError(
                f"Dataset item {index}: expected_decision must be a string."
            )

    return True


if __name__ == "__main__":

    dataset = load_golden_dataset()

    validate_golden_dataset(dataset)

    print("=" * 60)
    print("GOLDEN DATASET")
    print("=" * 60)

    print(f"Dataset path : {DATASET_PATH}")
    print(f"Questions    : {len(dataset)}")
    print("Validation   : PASSED")

    print("\nQuestions:")

    for index, item in enumerate(dataset, start=1):
        print(f"{index}. {item['question']}")

    print("=" * 60)