# Quickstart Guide: RAG Chatbot for Physical AI & Humanoid Robotics Book

**Feature**: 002-rag-chatbot
**Date**: 2025-12-18

## Overview

This guide provides instructions for setting up and running the RAG chatbot system that powers the question-answering feature for the Physical AI & Humanoid Robotics book.

## Prerequisites

- Python 3.11+
- Node.js 18+ (for frontend development)
- Qdrant Cloud account with API key
- Google AI API key for Gemini
- Git

## Backend Setup (FastAPI)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/physical-ai-book.git
cd physical-ai-book
```

### 2. Set up Python environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install fastapi uvicorn python-dotenv google-generativeai qdrant-client pydantic
```

### 3. Configure environment variables

Create a `.env` file in the backend directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=book_content_chunks
DEBUG=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600  # 1 hour in seconds
```

### 4. Run the backend server

```bash
cd backend
uvicorn src.api.v1.chat:app --reload --host 0.0.0.0 --port 8000
```

The backend API will be available at `http://localhost:8000`.

## Frontend Integration

### 1. Install dependencies in website directory

```bash
cd website
npm install
```

### 2. Add the chat widget to your Docusaurus site

The chat widget is a React component that can be added to your Docusaurus pages. It communicates with the backend API to process queries.

### 3. Run the Docusaurus site

```bash
cd website
npm run start
```

## API Usage

### Submit a question

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "question": "What is embodied intelligence?",
    "query_type": "global",
    "selected_text": null,
    "page_context": null
  }'
```

### Query with selected text context

```bash
curl -X POST http://localhost:8000/v1/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "question": "Explain this concept further",
    "query_type": "selection",
    "selected_text": "Embodied intelligence is the concept that intelligence emerges from the interaction...",
    "page_context": {
      "module": "module-1",
      "chapter": "chapter-1",
      "url": "/docs/module-1/chapter-1"
    }
  }'
```

## Data Preparation

### 1. Index book content to Qdrant

Before the chatbot can answer questions, the book content must be indexed in Qdrant:

```python
from src.services.qdrant_service import QdrantService

# Initialize the service
qdrant_service = QdrantService()

# Index your book content (chunks with metadata)
chunks = [
    {
        "chunk_id": "chunk-m1c1-001",
        "content": "Embodied intelligence is the concept that intelligence...",
        "module": "module-1",
        "chapter": "chapter-1",
        "page_url": "/docs/module-1/chapter-1",
        "heading": "Introduction to Embodied Intelligence",
        "difficulty": "intermediate"
    }
    # ... more chunks
]

qdrant_service.upsert_chunks(chunks)
```

### 2. Verify indexing

Check that content is properly indexed:

```bash
curl -X GET http://localhost:8000/v1/health
```

## Testing

### Backend tests

```bash
cd backend
pip install pytest pytest-asyncio
pytest tests/
```

### Frontend tests

```bash
cd website
npm test
```

## Deployment

### Backend (to production)

```bash
# Build and deploy with your preferred platform (e.g., Heroku, AWS, GCP)
gunicorn src.api.v1.chat:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Frontend (to GitHub Pages)

The chatbot widget is already integrated into the Docusaurus site and will be deployed with the existing GitHub Pages setup.

## Troubleshooting

### Common Issues

1. **API rate limits**: Check your OpenAI and Qdrant usage quotas
2. **No results returned**: Verify that book content is properly indexed in Qdrant
3. **Slow responses**: Check network connectivity to Qdrant and OpenAI APIs
4. **Hallucinations**: Ensure the system prompts are properly configured to limit responses to retrieved context

### Logging

The backend logs query types, response times, and any errors for monitoring and debugging.