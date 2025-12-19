from dataclasses import dataclass
from typing import Optional, Dict, Any
import re


@dataclass
class RetrievedChunk:
    """Represents a section of book content retrieved from the vector store"""
    chunk_id: str
    content: str
    module: str
    chapter: str
    page_url: str
    heading: str
    difficulty: str
    embedding: Optional[list] = None
    metadata: Optional[Dict[str, Any]] = None
    similarity_score: Optional[float] = None

    def __post_init__(self):
        """Validation after initialization"""
        if len(self.content) < 10 or len(self.content) > 20000:
            raise ValueError("Content must be between 10 and 20000 characters")

        if not re.match(r"^(beginner|intermediate|advanced)$", self.difficulty):
            raise ValueError("difficulty must be one of: beginner, intermediate, advanced")

        if self.similarity_score is not None and (self.similarity_score < 0.0 or self.similarity_score > 1.0):
            raise ValueError("similarity_score must be between 0.0 and 1.0")