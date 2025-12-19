# Physical AI & Humanoid Robotics Book - RAG Chatbot Backend

Backend service for the RAG chatbot that powers the question-answering feature for the Physical AI & Humanoid Robotics book.

## Overview

This FastAPI-based service provides:
- RAG (Retrieval-Augmented Generation) orchestration
- Integration with Qdrant vector store
- Google Gemini API integration
- API endpoints for chat functionality

## Prerequisites

- Python 3.11+
- Qdrant Cloud account with API key
- Google AI API key for Gemini

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

3. Run the server:
```bash
uvicorn src.api.v1.chat:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

- `POST /v1/chat` - Submit a question to the RAG chatbot
- `GET /v1/health` - Health check endpoint

## Architecture

The backend follows a service-oriented architecture:
- Models: Data models for queries, responses, and retrieved chunks
- Services: Business logic for RAG orchestration, Qdrant integration, and Gemini API
- API: FastAPI endpoints

## Environment Variables

- `GEMINI_API_KEY`: Google AI API key for Gemini
- `QDRANT_URL`: Qdrant cluster URL
- `QDRANT_API_KEY`: Qdrant API key
- `QDRANT_COLLECTION_NAME`: Name of the collection in Qdrant
- `DEBUG`: Enable debug mode
- `RATE_LIMIT_REQUESTS`: Number of requests allowed per window
- `RATE_LIMIT_WINDOW`: Time window in seconds for rate limiting

## Indexing Book Content

To index the book content to Qdrant, run:

```bash
python src/scripts/index_book_content.py
```

This will:
- Process all markdown files in the `website/docs` directory
- Chunk the content semantically
- Generate embeddings using Google's embedding model
- Index the content to your Qdrant collection

The indexing process may take several minutes depending on the size of your book content.