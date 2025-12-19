# Implementation Plan: Integrated RAG Chatbot for Physical AI & Humanoid Robotics Book

**Branch**: `002-rag-chatbot` | **Date**: 2025-12-18 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/002-rag-chatbot/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implement a Retrieval-Augmented Generation (RAG) chatbot that allows users to ask questions about the Physical AI & Humanoid Robotics book content. The system will provide two query modes: global (entire book) and selection-specific (selected text only), with responses grounded exclusively in book content and proper citations to relevant sections. The solution includes a FastAPI backend for RAG orchestration and a React-based frontend chat widget embedded in the Docusaurus website.

## Technical Context

**Language/Version**: Python 3.11 (for FastAPI backend), JavaScript/TypeScript (for frontend)
**Primary Dependencies**: FastAPI, Google Generative AI SDK (Gemini), Qdrant client, React, Docusaurus
**Storage**: Qdrant Cloud (vector database), no relational database for chatbot functionality
**Testing**: pytest for backend, Jest/React Testing Library for frontend
**Target Platform**: Linux server (backend), Web browser (frontend)
**Project Type**: web (backend + frontend integration)
**Performance Goals**: <10 seconds response time, 100 concurrent users support
**Constraints**: <10 seconds p95 response time, no chat history persistence, hallucination-free responses
**Scale/Scope**: Up to 100 concurrent users, book content with 48 chapters across 4 modules

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

1. **Technical Accuracy and Verification**: All technical explanations must be correct and implementation-ready. The RAG chatbot must return accurate, citation-aware responses grounded in book content.

2. **Developer-Focused Clarity**: Implementation must include clear documentation, proper examples, and well-structured code for developers maintaining the system.

3. **Practical Applicability**: Code examples must be runnable with proper integration into the existing Docusaurus site.

4. **Maintainability and Extensibility**: Architecture must support future enhancements without major refactoring, with clear separation of concerns between frontend and backend.

5. **Content Integrity and Non-Hallucination**: AI-generated responses must be grounded strictly in the book's content. The system must not hallucinate beyond indexed content.

6. **Vector Storage and Retrieval Standards**: All book content must be chunked and embedded consistently, with metadata including module, chapter, page URL, and headings.

7. **RAG Chatbot Requirements**: Backend must use FastAPI, LLM integration with Google Gemini via ChatKit, responses must cite relevant sections and state when answers cannot be derived from content.

## Project Structure

### Documentation (this feature)

```text
specs/002-rag-chatbot/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── models/
│   │   ├── query.py          # Query model with type, selected text, metadata
│   │   ├── retrieved_chunk.py # Retrieved chunk with content and metadata
│   │   └── response.py       # Response model with citations and disclaimers
│   ├── services/
│   │   ├── rag_service.py    # Core RAG orchestration logic
│   │   ├── qdrant_service.py # Qdrant vector store integration
│   │   └── gemini_service.py # Google Gemini API integration
│   └── api/
│       └── v1/
│           └── chat.py       # Chat endpoint with query processing
└── tests/
    ├── unit/
    ├── integration/
    └── contract/

frontend/
├── src/
│   ├── components/
│   │   ├── ChatWidget.jsx     # Main chat interface component
│   │   ├── ChatInterface.jsx  # Chat input/output UI
│   │   └── SelectionHandler.jsx # Text selection handling
│   └── services/
│       └── api.js            # API client for backend communication
└── tests/
    ├── unit/
    └── integration/
```

**Structure Decision**: Web application structure with separate backend (FastAPI) and frontend (React components) to maintain clear separation of concerns. Backend handles RAG orchestration and API, frontend handles UI integration with existing Docusaurus site.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
