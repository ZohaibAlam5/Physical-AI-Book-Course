from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class RetrievedChunk(BaseModel):
    """Represents a section of book content retrieved from the vector store"""
    chunk_id: str = Field(
        ...,
        description="Stable identifier for the content chunk"
    )
    content: str = Field(
        ...,
        min_length=10,
        max_length=20000,  # Approximately 2000 tokens
        description="The raw text content of the chunk"
    )
    module: str = Field(
        ...,
        description="The module number/name from the book"
    )
    chapter: str = Field(
        ...,
        description="The chapter number/name from the book"
    )
    page_url: str = Field(
        ...,
        description="URL to the page containing this content"
    )
    heading: str = Field(
        ...,
        description="The heading/section title"
    )
    difficulty: str = Field(
        ...,
        regex=r"^(beginner|intermediate|advanced)$",
        description="Difficulty level"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata for filtering"
    )
    similarity_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Relevance score from vector search"
    )

    class Config:
        schema_extra = {
            "example": {
                "chunk_id": "chunk-m1c1-001",
                "content": "Embodied intelligence is the concept that intelligence emerges from the interaction between an agent and its environment...",
                "module": "module-1",
                "chapter": "chapter-1",
                "page_url": "/docs/module-1/chapter-1",
                "heading": "Introduction to Embodied Intelligence",
                "difficulty": "intermediate",
                "similarity_score": 0.85
            }
        }