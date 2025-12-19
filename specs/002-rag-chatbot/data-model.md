# Data Model: Integrated RAG Chatbot for Physical AI & Humanoid Robotics Book

**Feature**: 002-rag-chatbot
**Date**: 2025-12-18
**Status**: Complete

## Overview

Data models for the RAG chatbot system, defining the structure of queries, responses, and vector store entries.

## Query Model

Represents a user's request to the chatbot system.

**Fields**:
- `id` (string): Unique identifier for the query
- `question` (string): The natural language question from the user
- `query_type` (enum): "global", "selection", or "page" to indicate the scope
- `selected_text` (string, optional): Text selected by user for selection-specific queries
- `page_context` (object, optional): Metadata about current page (module, chapter, URL)
- `timestamp` (datetime): When the query was submitted
- `user_metadata` (object, optional): Additional context about the user's session

**Validation rules**:
- `question` must be 1-2000 characters
- `query_type` must be one of the allowed values
- If `query_type` is "selection", `selected_text` must be provided

## Retrieved Chunk Model

Represents a section of book content retrieved from the vector store.

**Fields**:
- `chunk_id` (string): Stable identifier for the content chunk
- `content` (string): The raw text content of the chunk
- `module` (string): The module number/name from the book
- `chapter` (string): The chapter number/name from the book
- `page_url` (string): URL to the page containing this content
- `heading` (string): The heading/section title
- `difficulty` (string): Difficulty level (e.g., "beginner", "intermediate", "advanced")
- `metadata` (object): Additional metadata for filtering
- `similarity_score` (float): Relevance score from vector search

**Validation rules**:
- `chunk_id` must be unique and stable
- `content` must be 10-2000 tokens
- `module`, `chapter`, `page_url`, `heading` must be non-empty

## Response Model

Represents the chatbot's answer to a user's query.

**Fields**:
- `id` (string): Unique identifier for the response
- `query_id` (string): Reference to the original query
- `answer` (string): The AI-generated answer based on retrieved content
- `citations` (array): List of chunk IDs that contributed to the response
- `confidence` (enum): "high", "medium", "low" based on content match
- `has_answer` (boolean): Whether the book content contained the requested information
- `timestamp` (datetime): When the response was generated
- `query_type` (string): The type of query that generated this response

**Validation rules**:
- `answer` must be provided if `has_answer` is true
- `citations` must reference valid chunk IDs
- `confidence` must be one of the allowed values

## Vector Store Schema (Qdrant)

Definition of the collection structure in Qdrant Cloud.

**Collection name**: `book_content_chunks`

**Vector configuration**:
- `vector_size`: 1536 (for OpenAI embeddings) or appropriate size for Google embedding models
- `distance`: Cosine

**Payload schema**:
- `chunk_id` (keyword): Unique identifier for the chunk
- `content` (text): Raw text content
- `module` (keyword): Module identifier
- `chapter` (keyword): Chapter identifier
- `page_url` (keyword): URL to the page
- `heading` (text): Section heading
- `difficulty` (keyword): Difficulty level
- `metadata` (object): Additional filtering metadata

**Indexing strategy**:
- Keyword indexes on `module`, `chapter`, `page_url`, `difficulty` for fast filtering
- Text index on `content` and `heading` for semantic search

## State Transitions

The system follows a simple stateless pattern:

1. **Query received**: User submits a question with context
2. **Content retrieval**: System searches vector store based on query type and context
3. **Response generation**: LLM generates answer based on retrieved content
4. **Response returned**: System returns answer with citations to user

No persistent state is maintained between requests as per requirements.