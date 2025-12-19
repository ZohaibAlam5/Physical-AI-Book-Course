from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
import re


@dataclass
class QueryResponse:
    """Represents the chatbot's answer to a user's query"""
    id: str
    query_id: str
    confidence: str
    has_answer: bool
    query_type: str
    answer: Optional[str] = None
    citations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Validation after initialization"""
        if not re.match(r"^(high|medium|low)$", self.confidence):
            raise ValueError("confidence must be one of: high, medium, low")