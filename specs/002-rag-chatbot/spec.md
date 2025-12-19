# Feature Specification: Integrated RAG Chatbot for Physical AI & Humanoid Robotics Book

**Feature Branch**: `002-rag-chatbot`
**Created**: 2025-12-18
**Status**: Draft
**Input**: User description: "Integrated RAG Chatbot for \"Physical AI & Humanoid Robotics\" Book

Project context:
- The Docusaurus-based book website is already live and deployed on GitHub Pages
- Book content is complete, structured, and modular
- The goal is to augment the existing site with an embedded AI chatbot

Target users:
- Students reading the Physical AI & Humanoid Robotics book
- Instructors using the book as a teaching reference
- Self-learners seeking clarification on specific concepts

Primary objective:
- Build and embed a Retrieval-Augmented Generation (RAG) chatbot that answers questions strictly based on the book's content
- Allow users to ask questions either about the entire book or about text they explicitly select on a page

Core functionality:
- Natural language Q&A grounded exclusively in book content
- Accurate, citation-aware responses referencing book sections
- Support two query modes:
  1. Global mode: answers based on the entire book corpus
  2. Selection mode: answers based only on user-selected text
- Prevent hallucinations by restricting generation to retrieved context only

RAG architecture:
- Embedding source: Book content already chunked and embedded
- Vector store: Qdrant Cloud (Free Tier)
- Stored data per chunk:
  - Vector embedding
  - Raw text content
  - Metadata (module, chapter, page URL, heading, difficulty)
- Chunking strategy:
  - Section-level semantic chunks (500–800 tokens)
  - Stable chunk IDs mapped to book structure
- Retrieval:
  - Top-k similarity search
  - Metadata-based filtering (page-level, module-level)
  - Optional re-ranking for relevance

AI stack:
- LLM orchestration using Google Gemini via ChatKit SDKs
- Embeddings generated using Google embedding models or OpenAI embedding models
- Strict system prompts enforcing:
  - 'Answer only from retrieved context'
  - 'If not found in context, say so explicitly'

Backend services:
- FastAPI server acting as a stateless RAG orchestration layer
- API responsibilities:
  - Accept user queries and optional selected text
  - Perform similarity search in Qdrant
  - Assemble retrieved chunks into prompt context
  - Generate and return grounded answers
- No session persistence or user state stored

Data & persistence:
- Qdrant is the only database used
- No relational database or chat history storage
- No long-term storage of user queries or responses
- All RAG operations are stateless per request

Frontend integration:
- Embedded chat widget within the Docusaurus site
- UI requirements:
  - Docked or floating chatbot interface
  - 'Ask about this page' option (metadata-filtered retrieval)
  - 'Ask about selected text' option (selection-scoped answering)
  - Clear UI indicator when selection mode is active
- Lightweight JavaScript / React integration compatible with Docusaurus

Security & constraints:
- Read-only access to Qdrant collections
- No training or fine-tuning on user data
- API keys stored securely via environment variables
- Rate limiting and request validation to prevent abuse

Answer quality constraints:
- Responses must:
  - Use only retrieved Qdrant content
  - Avoid speculative, inferred, or external knowledge
  - Explicitly state when an answer is not present in the book
- Tone: instructional, concise, and technically accurate

Observability:
- Minimal logging for:
  - Query type (global vs selection)
  - Retrieved chunk IDs
  - Response latency
- No storage of personal or identifiable user data

Deployment:
- Backend deployed as a standalone FastAPI service
- Frontend embedded into the existing GitHub Pages site
- Qdrant Cloud configured via environment variables
- No changes required to existing book content or structure

Not building:
- Chat history persistence
- User authentication or accounts
- Analytics dashboards
- General-purpose or internet-enabled chatbot
- Fine-tuned language models

Out of scope:
- Voice-based interaction
- Multilingual support
- Mobile-native applications
- Offline usage

Final outcome:
- A fully integrated, Qdrant-only RAG chatbot embedded in the published Docusaurus book
- Accurate, hallucination-resistant answers grounded strictly in book content
- A lightweight, scalable AI assistant that enhances reader learning without additional databases"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Ask Questions About Book Content (Priority: P1)

A student reading the Physical AI & Humanoid Robotics book encounters a concept they don't understand and wants immediate clarification. They open the chatbot, type their question about the book content, and receive an accurate answer grounded in the book's text with proper citations to relevant sections.

**Why this priority**: This is the core value proposition of the feature - providing immediate, accurate answers to readers based on the book content, which directly enhances the learning experience.

**Independent Test**: Can be fully tested by asking various questions about book content and verifying that responses are accurate, grounded in book text, and include proper citations. Delivers immediate value by enabling self-paced learning with instant support.

**Acceptance Scenarios**:

1. **Given** a user is on any book page with the RAG chatbot available, **When** they type a question related to book content and submit it, **Then** they receive an accurate answer based only on book content with citations to relevant sections.

2. **Given** a user asks a question not covered in the book content, **When** they submit the query, **Then** the system explicitly states that the answer is not found in the book content.

3. **Given** a user submits a query, **When** the system processes the request, **Then** the response is delivered within 10 seconds and maintains instructional, concise, and technically accurate tone.

---

### User Story 2 - Ask Questions About Selected Text (Priority: P2)

A student is reading a specific section of the book and wants to understand a particular paragraph or concept better. They select the text on the page, activate the "Ask about selected text" feature, and receive an explanation focused only on that specific content.

**Why this priority**: This provides contextual assistance that is highly relevant to the user's immediate reading context, enhancing comprehension of specific passages.

**Independent Test**: Can be tested by selecting text on various book pages, asking questions about the selection, and verifying that responses are scoped only to the selected text and related concepts from the book.

**Acceptance Scenarios**:

1. **Given** a user has selected text on a book page, **When** they activate the "Ask about selected text" option and ask a question, **Then** the response is based only on the selected text and closely related book content.

2. **Given** a user has selected text on a page, **When** they ask a question that cannot be answered from the selected text, **Then** the system indicates the limitation and suggests asking about the broader page or book content.

---

### User Story 3 - Ask Questions About Current Page (Priority: P3)

A student wants to ask questions specifically about the current page they're viewing. They use the "Ask about this page" feature to get answers that are filtered to only include content from the current module and chapter.

**Why this priority**: This provides page-level context that helps users understand how concepts relate within the current section before expanding to broader book content.

**Independent Test**: Can be tested by using the "Ask about this page" feature on different pages and verifying that responses are appropriately scoped to the current page's content and context.

**Acceptance Scenarios**:

1. **Given** a user is on a specific book page, **When** they use the "Ask about this page" feature, **Then** the response is filtered to content relevant to the current module and chapter.

2. **Given** a user activates page-specific mode, **When** they ask a question, **Then** the system indicates when the answer requires broader book context beyond the current page.

---

### Edge Cases

- What happens when the user submits an extremely long query or question?
- How does the system handle queries that are ambiguous or poorly formed?
- What occurs when Qdrant vector store is temporarily unavailable?
- How does the system respond when the user asks about content that exists in multiple chapters with different meanings?
- What happens when the API rate limit is exceeded?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a chat interface that allows users to ask natural language questions about the book content
- **FR-002**: System MUST retrieve relevant book content using vector similarity search in Qdrant Cloud
- **FR-003**: Users MUST be able to ask questions in two modes: global (entire book) and selection-specific (selected text only)
- **FR-004**: System MUST return answers that are grounded exclusively in retrieved book content without hallucination
- **FR-005**: System MUST include citations to specific book sections (module, chapter, page URL) in responses
- **FR-006**: System MUST provide an "Ask about this page" option that filters retrieval to current page context
- **FR-007**: System MUST indicate when requested information is not available in the book content
- **FR-008**: System MUST handle user text selection and provide selection-specific Q&A capability
- **FR-009**: System MUST implement rate limiting to prevent abuse of the API
- **FR-010**: System MUST log query types (global vs selection) and response latency for observability

### Key Entities

- **Query**: A user's natural language question about book content, including metadata about query type (global/selection/page-specific) and optional selected text
- **Retrieved Chunk**: A section of book content retrieved from Qdrant that matches the user's query, containing text content and metadata (module, chapter, page URL, heading, difficulty)
- **Response**: An AI-generated answer based on retrieved chunks, including citations and appropriate disclaimers when content is not found
- **Book Content**: The structured content of the Physical AI & Humanoid Robotics book, chunked into semantic sections with metadata

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Users can ask questions about book content and receive accurate, grounded answers within 10 seconds 95% of the time
- **SC-002**: 90% of user queries result in responses that are properly grounded in book content without hallucination
- **SC-003**: Users can successfully use both global and selection-specific query modes with 95% task completion rate
- **SC-004**: The system handles at least 100 concurrent users without performance degradation
- **SC-005**: 80% of users report that the chatbot helped them better understand book concepts when surveyed after use
