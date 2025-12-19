import logging
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime
from src.services.qdrant_service import QdrantService
from src.services.gemini_service import GeminiService
from src.models.query import QueryRequest
from src.models.response import QueryResponse
from src.models.retrieved_chunk import RetrievedChunk

logger = logging.getLogger(__name__)


class RAGService:
    """Core RAG orchestration service integrating Qdrant and Gemini services"""

    def __init__(self):
        self.qdrant_service = QdrantService()
        self.gemini_service = GeminiService()

    def process_query(self, query_request: QueryRequest) -> QueryResponse:
        """
        Process a query request and return a response.

        Args:
            query_request: The query request with question and context

        Returns:
            QueryResponse with the answer and citations
        """
        # Validate the query request
        if query_request.query_type == "selection" and not query_request.selected_text:
            raise ValueError("selected_text must be provided for selection-type queries")

        # Generate embedding for the question
        query_embedding = self.gemini_service.embed_text(query_request.question)

        # Search for relevant chunks based on query type
        retrieved_chunks = self._search_chunks(query_request, query_embedding)

        if not retrieved_chunks:
            # No relevant content found
            response_id = f"resp-{uuid.uuid4().hex[:8]}"
            return QueryResponse(
                id=response_id,
                query_id=query_request.id or f"query-{uuid.uuid4().hex[:8]}",
                answer="The information requested is not available in the book content.",
                citations=[],
                confidence="low",
                has_answer=False,
                timestamp=datetime.utcnow(),
                query_type=query_request.query_type
            )

        # Extract content from retrieved chunks for context
        context_contents = [chunk.content for chunk in retrieved_chunks]

        # Generate response using Gemini
        answer = self.gemini_service.generate_response(
            context_chunks=context_contents,
            question=query_request.question,
            selected_text=query_request.selected_text if query_request.query_type == "selection" else None
        )

        # Determine confidence based on similarity scores
        avg_similarity = sum(
            chunk.similarity_score or 0 for chunk in retrieved_chunks
        ) / len(retrieved_chunks) if retrieved_chunks else 0

        if avg_similarity > 0.7:
            confidence = "high"
        elif avg_similarity > 0.4:
            confidence = "medium"
        else:
            confidence = "low"

        # Create response object
        response_id = f"resp-{uuid.uuid4().hex[:8]}"
        response = QueryResponse(
            id=response_id,
            query_id=query_request.id or f"query-{uuid.uuid4().hex[:8]}",
            answer=answer,
            citations=[chunk.chunk_id for chunk in retrieved_chunks],
            confidence=confidence,
            has_answer=True,
            timestamp=datetime.utcnow(),
            query_type=query_request.query_type
        )

        return response

    def _search_chunks(
        self,
        query_request: QueryRequest,
        query_embedding: List[float]
    ) -> List[RetrievedChunk]:
        """
        Search for relevant chunks based on the query type and context.

        Args:
            query_request: The original query request
            query_embedding: Embedding of the query question

        Returns:
            List of relevant RetrievedChunk objects
        """
        if query_request.query_type == "page" and query_request.page_context:
            # Page-specific search - filter by module, chapter, or URL
            return self.qdrant_service.search_chunks(
                query_vector=query_embedding,
                limit=10,
                query_type="page",
                module=query_request.page_context.module,
                chapter=query_request.page_context.chapter,
                page_url=query_request.page_context.url
            )
        elif query_request.query_type == "selection" and query_request.selected_text:
            # For selection-specific queries, we might want to embed the selected text
            # and use it as additional context, but for now we'll do a regular search
            # with potential filtering if needed
            selection_embedding = self.gemini_service.embed_text(query_request.selected_text)
            # Combine the question and selected text embeddings or just use the question embedding
            # For now, using the question embedding but we could also search for the selected text
            return self.qdrant_service.search_chunks(
                query_vector=query_embedding,
                limit=10,
                query_type="global"  # We'll use global search but potentially filter differently
            )
        else:
            # Global search - search across entire book corpus
            return self.qdrant_service.search_chunks(
                query_vector=query_embedding,
                limit=10,
                query_type="global"
            )

    def process_global_query(self, question: str) -> QueryResponse:
        """Process a global query (entire book corpus search)"""
        query_request = QueryRequest(
            question=question,
            query_type="global"
        )
        return self.process_query(query_request)

    def process_selection_query(
        self,
        question: str,
        selected_text: str,
        page_context: Optional[Dict[str, str]] = None
    ) -> QueryResponse:
        """Process a selection-specific query"""
        query_request = QueryRequest(
            question=question,
            query_type="selection",
            selected_text=selected_text,
            page_context=page_context
        ) if page_context else QueryRequest(
            question=question,
            query_type="selection",
            selected_text=selected_text
        )
        return self.process_query(query_request)

    def process_page_query(
        self,
        question: str,
        module: str,
        chapter: str,
        url: str
    ) -> QueryResponse:
        """Process a page-specific query"""
        from src.models.query import PageContext

        query_request = QueryRequest(
            question=question,
            query_type="page",
            page_context=PageContext(
                module=module,
                chapter=chapter,
                url=url
            )
        )
        return self.process_query(query_request)