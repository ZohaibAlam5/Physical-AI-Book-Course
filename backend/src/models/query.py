from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class PageContext(BaseModel):
    """Metadata about the current page for page-specific queries"""
    module: str = Field(..., description="The module identifier")
    chapter: str = Field(..., description="The chapter identifier")
    url: str = Field(..., description="The page URL")


class QueryRequest(BaseModel):
    """Represents a user's request to the chatbot system"""
    id: Optional[str] = None
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The natural language question from the user"
    )
    query_type: str = Field(
        ...,
        regex=r"^(global|selection|page)$",
        description="Type of query: global, selection, or page"
    )
    selected_text: Optional[str] = Field(
        None,
        description="Text selected by user for selection-specific queries"
    )
    page_context: Optional[PageContext] = Field(
        None,
        description="Metadata about current page (module, chapter, URL)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the query was submitted"
    )
    user_metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context about the user's session"
    )

    class Config:
        schema_extra = {
            "example": {
                "question": "What is embodied intelligence in robotics?",
                "query_type": "global",
                "selected_text": None,
                "page_context": {
                    "module": "module-1",
                    "chapter": "chapter-1",
                    "url": "/docs/module-1/chapter-1"
                }
            }
        }

    def validate_for_selection_type(self):
        """Validate that selected_text is provided when query_type is 'selection'"""
        if self.query_type == "selection" and not self.selected_text:
            raise ValueError("selected_text must be provided for selection-type queries")