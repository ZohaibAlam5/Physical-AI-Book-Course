<!--
SYNC IMPACT REPORT:
Version change: N/A → 1.0.0
Modified principles: N/A (new constitution)
Added sections: All principles and sections
Removed sections: N/A
Templates requiring updates:
  - .specify/templates/plan-template.md ✅ updated
  - .specify/templates/spec-template.md ✅ updated
  - .specify/templates/tasks-template.md ✅ updated
  - .specify/commands/*.md ⚠ pending review
Follow-up TODOs: None
-->

# Technical Book with RAG Chatbot Constitution

## Core Principles

### Technical Accuracy and Verification
All technical explanations must be correct and implementation-ready. All technical content must be verified through authoritative sources and code validation. No technical information may be presented without verification of its accuracy.

### Developer-Focused Clarity
Content must prioritize clarity for a developer-focused audience (software engineers, CS students). Writing must be instructional and developer-friendly with intermediate technical reading level and no unnecessary jargon.

### Practical Applicability
All content must include real-world examples and implementations. Code examples must be runnable, version-pinned, and properly documented. Each chapter must include concept explanation, diagrams/architecture descriptions, code snippets, and practical use cases.

### Maintainability and Extensibility
Both content and codebase must be maintainable and extensible. The architecture must support future enhancements without major refactoring. Code must follow clean architecture principles with clear separation of concerns.

### Content Integrity and Non-Hallucination
AI-generated explanations must be grounded strictly in the book's content. RAG chatbot responses must not hallucinate beyond indexed content. No content plagiarism is permitted (0% tolerance). No hallucinated citations or references.

### Transparency and Open Source
All AI behavior and data usage must be transparent. All diagrams, examples, and explanations must align with the actual implementation. Open-source friendly licensing considerations must be observed throughout development.

## Technical Standards

### Documentation and Publishing Standards
- Static site generator: Docusaurus
- Version control: GitHub
- Deployment target: GitHub Pages
- Content format: Markdown (MD/MDX)
- Navigation: logical chapter hierarchy with sidebar indexing
- All content must have clean build with no broken links, errors, or missing embeddings

### Vector Storage and Retrieval Standards
- All book content must be chunked and embedded consistently across all chapters
- Vector database: Qdrant Cloud (Free Tier)
- Metadata must include: Chapter, Section, Heading, Source file
- All content must be indexed and retrievable from Qdrant

### RAG Chatbot Requirements
- Chatbot must be embedded within the published book UI
- Backend framework: FastAPI
- LLM integration: Google Gemini via ChatKit SDKs
- Retrieval must support: entire book content AND user-selected text only (selection-based querying)
- Responses must cite relevant book sections and clearly state when answers cannot be derived from available content

## Development Workflow

### Implementation Standards
- Codebase must be reproducible and well-documented
- Smallest viable changes with no unrelated edits
- Explicit error paths and constraints must be stated
- Code references to modified/inspected files where relevant
- Prefer smallest viable diff; do not refactor unrelated code

### Quality Assurance Process
- All technical explanations must be implementation-ready
- Code examples must be runnable and tested
- Architectural decisions must be clearly justified
- All outputs must strictly follow user intent
- Prompt History Records (PHRs) must be created automatically and accurately for every user prompt
- Architectural Decision Records (ADRs) must be suggested intelligently for significant decisions

## Governance

This constitution supersedes all other development practices and standards. All project work must comply with these principles. Amendments require documentation of the change, approval from project stakeholders, and a migration plan for existing codebase. All pull requests and code reviews must verify compliance with these principles. All architectural decisions that meet significance criteria (long-term consequences, multiple viable options considered, cross-cutting influence) must be documented as ADRs with `/sp.adr <decision-title>`. Use CLAUDE.md for runtime development guidance and agent behavior.

**Version**: 1.0.0 | **Ratified**: 2025-12-16 | **Last Amended**: 2025-12-16
