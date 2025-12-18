---
id: 1
title: Create Book RAG Constitution
stage: constitution
date: 2025-12-16
surface: agent
model: claude-sonnet-4-5-20250929
feature: none
branch: master
user: claude
command: /sp.constitution
labels: [constitution,rag,book,setup]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - .specify/memory/constitution.md
tests:
 - none
---

## Prompt

```text
Project: Technical book with integrated RAG chatbot

Core principles:
- Technical accuracy verified through authoritative sources and code validation
- Clarity for a developer-focused audience (software engineers, CS students)
- Practical applicability with real-world examples and implementations
- Maintainability and extensibility of both content and codebase
- Transparency in AI behavior and data usage

Key standards:
- All technical explanations must be correct and implementation-ready
- Code examples must be runnable, version-pinned, and documented
- Architectural decisions must be clearly justified
- AI-generated explanations must be grounded strictly in the book's content
- RAG chatbot responses must not hallucinate beyond indexed content

Content standards:
- Writing style: clear, instructional, and developer-friendly
- Reading level: intermediate technical (no unnecessary jargon)
- Each chapter must include:
  - Concept explanation
  - Diagrams or architecture descriptions (where applicable)
  - Code snippets
  - Practical use cases

Documentation & publishing:
- Static site generator: Docusaurus
- Version control: GitHub
- Deployment target: GitHub Pages
- Content format: Markdown (MD/MDX)
- Navigation: logical chapter hierarchy with sidebar indexing

Vector storage & retrieval:
- All book content must be chunked and embedded
- Vector database: Qdrant Cloud (Free Tier)
- Embedding consistency across all chapters
- Metadata must include:
  - Chapter
  - Section
  - Heading
  - Source file

RAG chatbot requirements:
- Chatbot must be embedded within the published book UI
- Backend framework: FastAPI
- LLM integration: OpenAI Agents / ChatKit SDKs
- Relational database: Neon Serverless Postgres
- Retrieval must be scoped to:
  - Entire book content
  - OR user-selected text only (selection-based querying)
- Responses must:
  - Cite the relevant book section
  - Clearly state when an answer cannot be derived from available content
  - Avoid speculative or external knowledge unless explicitly allowed

Constraints:
- No content plagiarism (0% tolerance)
- No hallucinated citations or references
- All diagrams, examples, and explanations must align with the actual implementation
- Open-source friendly licensing considerations

Success criteria:
- Book successfully deployed on GitHub Pages
- All content indexed and retrievable from Qdrant
- RAG chatbot accurately answers questions based solely on book content
- Selection-based Q&A works reliably
- Clean build with no broken links, errors, or missing embeddings
- Codebase is reproducible and well-documented
```

## Response snapshot

Created a comprehensive project constitution for the technical book with integrated RAG chatbot project. The constitution includes:

- 6 core principles covering technical accuracy, clarity, practicality, maintainability, content integrity, and transparency
- Technical standards for documentation, vector storage, and RAG chatbot requirements
- Development workflow guidelines for implementation and quality assurance
- Governance rules for project management and compliance

The constitution is versioned as 1.0.0 and includes all the requirements specified in the user input.

## Outcome

- ✅ Impact: Created foundational project constitution document that will guide all future development
- 🧪 Tests: none
- 📁 Files: Updated .specify/memory/constitution.md with comprehensive constitution
- 🔁 Next prompts: Project can now proceed with spec, plan, and implementation phases
- 🧠 Reflection: Constitution provides clear guidelines for technical accuracy, non-hallucination requirements, and RAG chatbot functionality

## Evaluation notes (flywheel)

- Failure modes observed: none
- Graders run and results (PASS/FAIL): constitution validation passed
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): Begin feature specification with /sp.specify