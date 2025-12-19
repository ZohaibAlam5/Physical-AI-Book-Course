from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class QueryResponse(BaseModel):
    """Represents the chatbot's answer to a user's query"""
    id: str = Field(
        ...,
        description="Unique identifier for the response"
    )
    query_id: str = Field(
        ...,
        description="Reference to the original query"
    )
    answer: Optional[str] = Field(
        None,
        description="The AI-generated answer based on retrieved content"
    )
    citations: List[str] = Field(
        default_factory=list,
        description="List of chunk IDs that contributed to the response"
    )
    confidence: str = Field(
        ...,
        regex=r"^(high|medium|low)$",
        description="Confidence level in the response"
    )
    has_answer: bool = Field(
        ...,
        description="Whether the book content contained the requested information"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the response was generated"
    )
    query_type: str = Field(
        ...,
        description="The type of query that generated this response"
    )

    class Config:
        schema_extra = {
            "example": {
                "id": "resp-12345",
                "query_id": "query-67890",
                "answer": "Embodied intelligence refers to the concept where intelligence emerges from the interaction between an agent and its environment...",
                "citations": ["chunk-m1c1-001", "chunk-m1c1-002"],
                "confidence": "high",
                "has_answer": True,
                "timestamp": "2025-12-18T10:30:00Z",
                "query_type": "global"
            }
        }