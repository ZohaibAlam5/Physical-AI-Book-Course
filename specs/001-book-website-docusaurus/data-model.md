# Data Model: Physical AI & Humanoid Robotics Book

## Entity: Book Module
- **Fields**:
  - id: string (unique identifier, e.g., "module-1")
  - title: string (e.g., "Physical AI Foundations & the Robotic Nervous System")
  - description: string (brief description of the module content)
  - order: integer (sequence number 1-4)
  - learningOutcomes: array of strings (what readers should understand after completing)
  - chapters: array of BookChapter references
- **Relationships**:
  - Contains 10-12 BookChapter entities
  - Belongs to one Book (the Physical AI book)
- **Validation rules**:
  - title must be 5-100 characters
  - order must be between 1-4
  - must have at least one chapter

## Entity: Book Chapter
- **Fields**:
  - id: string (unique identifier, e.g., "module-1/chapter-1")
  - title: string (chapter title)
  - content: string (Markdown/MDX content)
  - module: string (reference to parent module)
  - order: integer (sequence within module, 1-12)
  - learningObjectives: array of strings (what readers should learn)
  - prerequisites: array of strings (required knowledge)
  - difficultyLevel: enum (Beginner, Intermediate, Advanced)
  - estimatedReadingTime: integer (in minutes)
  - metadata: object (additional information for search and indexing)
- **Relationships**:
  - Belongs to one BookModule
  - May reference other BookChapters (cross-references)
- **Validation rules**:
  - title must be 5-100 characters
  - content must be valid Markdown/MDX
  - order must be positive integer
  - difficultyLevel must be one of the allowed values

## Entity: Learning Outcome
- **Fields**:
  - id: string (unique identifier)
  - module: string (reference to parent module)
  - description: string (what reader should be able to do)
  - type: enum (Conceptual, Practical, Analytical)
- **Relationships**:
  - Belongs to one BookModule
  - May be referenced by multiple BookChapters
- **Validation rules**:
  - description must be 10-200 characters
  - type must be one of the allowed values

## Entity: Technical Concept
- **Fields**:
  - id: string (unique identifier)
  - name: string (name of the concept)
  - definition: string (clear definition)
  - module: string (primary module where concept is taught)
  - relatedConcepts: array of strings (other related concepts)
  - examples: array of strings (practical examples)
- **Relationships**:
  - May be referenced by multiple BookChapters
  - Belongs to one primary BookModule
- **Validation rules**:
  - name must be 2-50 characters
  - definition must be 10-500 characters

## Entity: Book Navigation Item
- **Fields**:
  - id: string (unique identifier)
  - label: string (display text)
  - type: enum (category, link)
  - href: string (URL path for links)
  - items: array of BookNavigationItem (for hierarchical structure)
  - collapsed: boolean (whether section is collapsed by default)
- **Relationships**:
  - Forms hierarchical tree structure
  - Links to BookChapter entities (for type=link)
- **Validation rules**:
  - label must be 1-50 characters
  - href must be valid URL path (for type=link)

## Entity: Search Index Entry
- **Fields**:
  - id: string (unique identifier)
  - title: string (content title)
  - content: string (content excerpt for search)
  - url: string (URL to content)
  - module: string (module identifier)
  - chapter: string (chapter identifier)
  - tags: array of strings (search tags)
  - createdAt: timestamp
  - updatedAt: timestamp
- **Relationships**:
  - Corresponds to BookChapter entities
- **Validation rules**:
  - title must be 5-200 characters
  - content must be 50-1000 characters for search excerpts
  - url must be valid relative path

## State Transitions
- Book Chapter: Draft → Review → Published → Archived
- Module: Planning → Development → Review → Published → Maintenance

## Relationships Summary
- Book contains 4 BookModules
- Each BookModule contains 10-12 BookChapters
- Each BookChapter has multiple LearningOutcomes
- Technical Concepts may be referenced across multiple chapters
- Book Navigation Items organize the content hierarchy
- Search Index Entries enable content discovery