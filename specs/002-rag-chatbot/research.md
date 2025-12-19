# Research: Integrated RAG Chatbot for Physical AI & Humanoid Robotics Book

**Feature**: 002-rag-chatbot
**Date**: 2025-12-18
**Status**: Complete

## Overview

Research summary for implementing a Retrieval-Augmented Generation (RAG) chatbot for the Physical AI & Humanoid Robotics book. This research addresses all technical unknowns and provides the foundation for the implementation plan.

## Decision: RAG Architecture Pattern

**Rationale**: RAG (Retrieval-Augmented Generation) is the optimal pattern for question-answering systems that need to ground responses in specific source content. For a book Q&A system, RAG allows:
- Accurate retrieval of relevant book sections
- Prevention of hallucinations by limiting context to book content
- Proper citations to specific sections
- Scalable vector search across large content corpus

**Alternatives considered**:
- Pure generative model: Would hallucinate beyond book content
- Keyword-based search: Would lack semantic understanding
- Rule-based system: Would be too rigid for natural language questions

## Decision: Technology Stack

**Rationale**: The selected technology stack aligns with the project requirements and constraints:

1. **Backend**: FastAPI for Python-based API with excellent async support and OpenAPI documentation
2. **Vector Store**: Qdrant Cloud for managed vector database with metadata filtering capabilities
3. **LLM Integration**: Google Generative AI SDK (Gemini) via ChatKit for reliable, well-documented API access
4. **Frontend**: React components for integration with Docusaurus site
5. **Embeddings**: Google embedding models or OpenAI embeddings for consistent vector representations

**Alternatives considered**:
- LangChain vs custom RAG: Custom implementation preferred for better control and understanding
- Different vector stores (Pinecone, Weaviate): Qdrant chosen for cost-effectiveness and features
- Different LLM providers: OpenAI chosen for reliability and quality

## Decision: Query Modes Implementation

**Rationale**: The two required query modes (global and selection-specific) can be implemented through metadata filtering in Qdrant:

1. **Global mode**: Query entire book corpus without filters
2. **Selection mode**: Apply metadata filters to restrict to selected text context
3. **Page mode**: Apply metadata filters for current page/module

**Implementation approach**: Use Qdrant's payload filtering to scope search results based on metadata (module, chapter, section) associated with each chunk.

## Decision: Frontend Integration Strategy

**Rationale**: The chatbot widget needs to integrate seamlessly with the existing Docusaurus site without requiring changes to book content:

1. **Component approach**: React component that can be embedded in Docusaurus pages
2. **Text selection API**: Browser Selection API to capture user-selected text
3. **Context awareness**: Metadata about current page to enable "Ask about this page" feature

**Alternatives considered**:
- Separate application: Would require context switching
- Docusaurus plugin: Would be more complex to implement and maintain

## Decision: Data Chunking Strategy

**Rationale**: For book content, semantic chunking at section level with 500-800 tokens provides optimal balance between:
- Retrieval precision (not too broad)
- Context completeness (not too narrow)
- Token efficiency for LLM processing

**Implementation approach**:
- Split book content at section headings
- Ensure chunks maintain semantic coherence
- Include metadata (module, chapter, page URL, heading) with each chunk
- Use stable IDs mapped to book structure for consistent referencing

## Decision: Hallucination Prevention

**Rationale**: Critical requirement to maintain trust and accuracy of the system. Multiple strategies needed:

1. **Strict context limitation**: Only use retrieved chunks as LLM context
2. **System prompts**: Enforce "answer only from context" behavior
3. **Response validation**: Check that answers are grounded in provided context
4. **Clear disclaimers**: Explicitly state when information is not in book content

## Decision: Rate Limiting and Security

**Rationale**: To prevent abuse while maintaining good user experience:

1. **API-level rate limiting**: Implement at FastAPI layer
2. **Request validation**: Validate input and filter for security
3. **Environment variables**: Secure storage of API keys
4. **Read-only access**: Qdrant access limited to search operations