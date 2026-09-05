from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Evidence:
    """
    Represents a single piece of research evidence.
    """

    content: str

    source: str

    page: Optional[int]

    similarity_score: float

    evidence_id: str

    source_type: str = "unknown"

    source_quality: float = 0.30

    def to_dict(self):
        """
        Convert evidence into a JSON-serializable dictionary.
        """

        return asdict(self)