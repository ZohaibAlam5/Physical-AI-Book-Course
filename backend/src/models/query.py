from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
import re


@dataclass
class PageContext:
    """Metadata about the current page for page-specific queries"""
    module: str
    chapter: str
    url: str


@dataclass
class QueryRequest:
    """Represents a user's request to the chatbot system"""
    question: str
    query_type: str
    id: Optional[str] = None
    selected_text: Optional[str] = None
    page_context: Optional[PageContext] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validation after initialization"""
        if len(self.question) < 1 or len(self.question) > 2000:
            raise ValueError("Question must be between 1 and 2000 characters")

        if not re.match(r"^(global|selection|page)$", self.query_type):
            raise ValueError("query_type must be one of: global, selection, page")

    def validate_for_selection_type(self):
        """Validate that selected_text is provided when query_type is 'selection'"""
        if self.query_type == "selection" and not self.selected_text:
            raise ValueError("selected_text must be provided for selection-type queries")