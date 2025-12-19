"""
Main entry point for the RAG Chatbot Backend
"""
import os
import uvicorn
from src.api.v1.chat import app


if __name__ == "__main__":
    # Get host and port from environment variables or use defaults
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))

    # Run the FastAPI application
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )