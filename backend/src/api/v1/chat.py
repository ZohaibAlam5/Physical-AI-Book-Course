"""
Chat API endpoint for the RAG Chatbot
Handles user queries and returns AI-generated responses based on book content.
"""
import os
import logging
import time
import uuid
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai

from src.models.query import QueryRequest
from src.models.response import QueryResponse
from src.services.rag_service import RAGService
from src.services.gemini_service import GeminiService
from src.services.qdrant_service import QdrantService
from src.middleware.rate_limit import RateLimitMiddleware


# Initialize FastAPI app
app = FastAPI(title="RAG Chatbot API for Physical AI & Humanoid Robotics Book",
              description="API for question-answering service based on book content using Google Gemini",
              version="1.0.0")

# Add CORS middleware first
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware
rate_limit_middleware = RateLimitMiddleware()
app.middleware('http')(rate_limit_middleware.__call__)

# Initialize services
gemini_service = GeminiService()
qdrant_service = QdrantService()
rag_service = RAGService()  # RAGService internally initializes its own services

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.get("/")
def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "message": "RAG Chatbot API is running"}


@app.get("/health")
def detailed_health_check():
    """Detailed health check endpoint."""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "gemini": "connected" if rag_service.gemini_service.model else "disconnected",
            "qdrant": "connected" if rag_service.qdrant_service.client else "disconnected",
        }
    }
    return health_status


@app.post("/chat")
async def chat_endpoint(query_request: QueryRequest):
    """
    Submit a question to the RAG chatbot.

    Process a user's question and return an answer based on book content.
    """
    try:
        start_time = time.time()
        logger.info(f"Received query: {query_request.question[:100]}...")

        # Validate the query request (dataclass validation is done during instantiation)
        if not query_request.question or len(query_request.question.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )

        if len(query_request.question) > 2000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question must be between 1 and 2000 characters"
            )

        # Generate a unique query ID
        query_id = f"query-{uuid.uuid4()}"

        # Process the query using the RAG service
        response = await rag_service.process_query(query_request)

        # Calculate response time
        response_time = time.time() - start_time
        logger.info(f"Query processed in {response_time:.2f}s")

        # Add the query ID to the response if not already set
        if not hasattr(response, 'query_id') or response.query_id is None:
            response.query_id = query_id

        # Log query type for observability
        logger.info(f"Query type: {query_request.query_type}, Response time: {response_time:.2f}s")

        return response

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your query"
        )


@app.post("/v1/chat")
async def chat_endpoint_v1(query_request: QueryRequest):
    """
    V1 chat endpoint (alias for /chat).
    """
    return await chat_endpoint(query_request)


# Error handlers
from starlette.responses import JSONResponse

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "message": "The requested endpoint was not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": "An unexpected error occurred"}
    )


# Additional utility endpoints
@app.get("/stats")
async def get_stats():
    """Get basic statistics about the service."""
    # This would connect to your metrics system in a real implementation
    return {
        "total_queries": 0,  # Would come from a metrics system
        "avg_response_time": 0.0,  # Would come from a metrics system
        "uptime": time.time()  # Simple uptime since last restart
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )