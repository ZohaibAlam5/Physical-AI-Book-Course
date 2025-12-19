---
title: Physical AI & Humanoid Robotics Assistant
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: mit
---

# Physical AI & Humanoid Robotics Assistant

This is a RAG (Retrieval-Augmented Generation) chatbot that can answer questions about the Physical AI & Humanoid Robotics book content.

## Overview

This application uses a Retrieval-Augmented Generation (RAG) system to answer questions about the Physical AI & Humanoid Robotics book. It retrieves relevant sections from the book and uses them to generate accurate responses.

## Prerequisites

- Qdrant Cloud account with API key
- Google AI API key for Gemini (optional, only needed for LLM responses)

## Deployment on Hugging Face Spaces

This application can be deployed directly on Hugging Face Spaces using the Docker SDK. The interface is built with Gradio for easy interaction.

### Environment Variables

To run this Space, you'll need to set the following environment variables:

- `QDRANT_URL`: URL for your Qdrant vector database
- `QDRANT_API_KEY`: API key for Qdrant (if required)
- `QDRANT_COLLECTION_NAME`: Name of the collection containing book content
- `GEMINI_API_KEY`: Google Gemini API key (optional, only needed for LLM responses)

### How to Use

1. Ask questions about the Physical AI & Humanoid Robotics book content
2. The assistant will retrieve relevant information and provide accurate answers
3. The system uses vector similarity search to find the most relevant book sections

## Local Development

If you want to run the application locally:

1. Install dependencies:
```bash
pip install -r requirements-hf.txt
```

2. Configure environment variables:
```bash
# Set environment variables for QDRANT and GEMINI
```

3. Run the Gradio interface:
```bash
python app.py
```

## API Endpoints (when running as FastAPI)

- `POST /chat` - Submit a question to the RAG chatbot
- `GET /health` - Health check endpoint

## Architecture

The backend follows a service-oriented architecture:
- Models: Data models for queries, responses, and retrieved chunks
- Services: Business logic for RAG orchestration, Qdrant integration, and Gemini API
- API: FastAPI endpoints with Gradio interface for Hugging Face Spaces

## Technical Details

- Uses Sentence Transformers for local embeddings
- Qdrant for vector storage and similarity search
- Google Gemini for response generation
- Built with FastAPI and Gradio