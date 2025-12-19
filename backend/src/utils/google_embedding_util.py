"""
Google Embedding Utility for the RAG Chatbot
Handles generating embeddings using Google's embedding models for book content.
"""
import os
import logging
from typing import List, Optional
import google.generativeai as genai
from google.generativeai.types import embedding_types


class GoogleEmbeddingService:
    """
    Service class for generating embeddings using Google's embedding models.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "models/embedding-001"):
        """
        Initialize the Google Embedding Service.

        Args:
            api_key: Google API key (if not provided, will use environment variable)
            model_name: Name of the embedding model to use
        """
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is required")

        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using Google's embedding model.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as a list of floats
        """
        try:
            # Use Google's embed_content method
            response = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="RETRIEVAL_DOCUMENT",  # Appropriate for document chunks
                title=None  # Can add a title if needed
            )

            if 'embedding' in response and response['embedding']:
                return response['embedding']
            else:
                self.logger.warning(f"No embedding returned for text: {text[:100]}...")
                return []

        except Exception as e:
            self.logger.error(f"Error generating embedding for text: {str(e)}")
            raise

    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 5) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts using Google's embedding model.
        This method batches requests to stay within API limits.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch (Google has limits)

        Returns:
            List of embedding vectors
        """
        embeddings = []

        # Process in batches to respect API limits
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

            try:
                # Use Google's embed_content method for batch
                response = genai.embed_content(
                    model=self.model_name,
                    content=batch,
                    task_type="RETRIEVAL_DOCUMENT",
                    title=None
                )

                batch_embeddings = response['embedding']
                embeddings.extend(batch_embeddings)

            except Exception as e:
                self.logger.error(f"Error generating embeddings for batch {i//batch_size + 1}: {str(e)}")
                # Add empty embeddings for failed items to maintain alignment
                embeddings.extend([[] for _ in range(len(batch))])

        return embeddings

    def validate_embedding(self, embedding: List[float], expected_dimension: Optional[int] = None) -> bool:
        """
        Validate that an embedding is properly formed.

        Args:
            embedding: Embedding vector to validate
            expected_dimension: Expected dimension of the embedding (optional)

        Returns:
            True if embedding is valid, False otherwise
        """
        if not embedding:
            return False

        if not isinstance(embedding, list):
            return False

        if not all(isinstance(val, (int, float)) for val in embedding):
            return False

        if expected_dimension and len(embedding) != expected_dimension:
            return False

        return True

    def get_model_dimension(self) -> int:
        """
        Get the dimension of the embedding model.

        Returns:
            Dimension of the embedding vectors
        """
        # For Google's embedding-001 model, the dimension is typically 768
        # We'll return the standard dimension for Google's embedding model
        # In practice, this would be determined dynamically if needed
        return 768


def create_embedding_for_content(text: str, api_key: Optional[str] = None) -> List[float]:
    """
    Convenience function to create a single embedding for content.

    Args:
        text: Text to embed
        api_key: Optional API key (will use env var if not provided)

    Returns:
        Embedding vector as a list of floats
    """
    service = GoogleEmbeddingService(api_key=api_key)
    return service.generate_embedding(text)


def create_embeddings_for_contents(texts: List[str], api_key: Optional[str] = None, batch_size: int = 5) -> List[List[float]]:
    """
    Convenience function to create embeddings for multiple contents.

    Args:
        texts: List of texts to embed
        api_key: Optional API key (will use env var if not provided)
        batch_size: Batch size for processing

    Returns:
        List of embedding vectors
    """
    service = GoogleEmbeddingService(api_key=api_key)
    return service.generate_embeddings_batch(texts, batch_size)