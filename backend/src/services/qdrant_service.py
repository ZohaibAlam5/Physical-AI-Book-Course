import logging
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct, Distance, VectorParams
from src.config import settings
from src.models.retrieved_chunk import RetrievedChunk

logger = logging.getLogger(__name__)


class QdrantService:
    """Service for interacting with Qdrant vector store"""

    def __init__(self):
        # Initialize Qdrant client
        if settings.qdrant_api_key:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                prefer_grpc=True
            )
        else:
            # For local instances without API key
            self.client = QdrantClient(url=settings.qdrant_url)

        self.collection_name = settings.qdrant_collection_name
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Ensure the collection exists with proper configuration"""
        try:
            # Check if collection exists
            self.client.get_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} already exists")
        except Exception:
            # Create collection if it doesn't exist
            # Using 768 dimensions for Google embedding models (you might need to adjust based on the specific model)
            # If using OpenAI embeddings, change to 1536
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )

            # Create payload indexes for fast filtering
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="module",
                field_schema=models.PayloadSchemaType.KEYWORD
            )

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="chapter",
                field_schema=models.PayloadSchemaType.KEYWORD
            )

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="page_url",
                field_schema=models.PayloadSchemaType.KEYWORD
            )

            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="difficulty",
                field_schema=models.PayloadSchemaType.KEYWORD
            )

            logger.info(f"Created collection {self.collection_name} with indexes")

    def upsert_chunks(self, chunks: List[Dict[str, Any]]):
        """Upsert content chunks to Qdrant collection"""
        points = []

        for chunk_data in chunks:
            # Create a RetrievedChunk instance to validate the data
            retrieved_chunk = RetrievedChunk(**chunk_data)

            # Create a Qdrant point
            point = PointStruct(
                id=retrieved_chunk.chunk_id,
                vector=[0.0] * 768,  # Placeholder - in real implementation, this would be the actual embedding
                payload={
                    "content": retrieved_chunk.content,
                    "module": retrieved_chunk.module,
                    "chapter": retrieved_chunk.chapter,
                    "page_url": retrieved_chunk.page_url,
                    "heading": retrieved_chunk.heading,
                    "difficulty": retrieved_chunk.difficulty,
                    "metadata": retrieved_chunk.metadata or {}
                }
            )
            points.append(point)

        # Upsert the points to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        logger.info(f"Upserted {len(points)} chunks to Qdrant")

    def search_chunks(
        self,
        query_vector: List[float],
        limit: int = 10,
        query_type: str = "global",
        module: Optional[str] = None,
        chapter: Optional[str] = None,
        page_url: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """Search for relevant chunks based on query vector and filters"""

        # Build the search filter based on query type and provided filters
        search_filter = models.Filter()

        if query_type == "page" and (module or chapter or page_url):
            # For page-specific queries, filter by the page context
            conditions = []
            if module:
                conditions.append(models.FieldCondition(
                    key="module",
                    match=models.MatchValue(value=module)
                ))
            if chapter:
                conditions.append(models.FieldCondition(
                    key="chapter",
                    match=models.MatchValue(value=chapter)
                ))
            if page_url:
                conditions.append(models.FieldCondition(
                    key="page_url",
                    match=models.MatchValue(value=page_url)
                ))

            if conditions:
                search_filter = models.Filter(must=conditions)

        elif filters:
            # Apply custom filters (for selection-specific or other filtered queries)
            search_filter = models.Filter(**filters)

        # Perform the search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=search_filter,
            limit=limit,
            with_payload=True
        )

        # Convert results to RetrievedChunk objects
        retrieved_chunks = []
        for result in results:
            payload = result.payload
            chunk = RetrievedChunk(
                chunk_id=result.id,
                content=payload.get("content", ""),
                module=payload.get("module", ""),
                chapter=payload.get("chapter", ""),
                page_url=payload.get("page_url", ""),
                heading=payload.get("heading", ""),
                difficulty=payload.get("difficulty", "intermediate"),
                metadata=payload.get("metadata", {}),
                similarity_score=result.score
            )
            retrieved_chunks.append(chunk)

        return retrieved_chunks

    def get_chunk_by_id(self, chunk_id: str) -> Optional[RetrievedChunk]:
        """Retrieve a specific chunk by its ID"""
        try:
            records = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[chunk_id],
                with_payload=True
            )

            if not records:
                return None

            record = records[0]
            payload = record.payload
            return RetrievedChunk(
                chunk_id=record.id,
                content=payload.get("content", ""),
                module=payload.get("module", ""),
                chapter=payload.get("chapter", ""),
                page_url=payload.get("page_url", ""),
                heading=payload.get("heading", ""),
                difficulty=payload.get("difficulty", "intermediate"),
                metadata=payload.get("metadata", {}),
                similarity_score=None
            )
        except Exception as e:
            logger.error(f"Error retrieving chunk {chunk_id}: {e}")
            return None

    def add_point(self, point_data: Dict[str, Any]):
        """Add a single point to the Qdrant collection"""
        point = PointStruct(
            id=point_data['id'],
            vector=point_data['vector'],
            payload=point_data['payload']
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point]
        )

    def delete_collection(self):
        """Delete the entire collection (useful for re-indexing)"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection {self.collection_name}: {e}")