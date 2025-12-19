"""
Sentence Transformer Embedding Utility for the RAG Chatbot
Handles generating embeddings using Sentence Transformers for book content.
"""
import os
import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer


class GoogleEmbeddingService:  # Keeping the name for compatibility
    """
    Service class for generating embeddings using Sentence Transformers.
    NOTE: Despite the name, this now uses Sentence Transformers instead of Google's API.
    """

    def __init__(self, api_key: Optional[str] = None, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Sentence Transformer Embedding Service.
        NOTE: api_key parameter is kept for compatibility but not used.

        Args:
            api_key: Not used, kept for compatibility
            model_name: Name of the Sentence Transformer model to use
        """
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)

        # Download and load the model
        self.logger.info(f"Loading Sentence Transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.logger.info(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using Sentence Transformers.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as a list of floats
        """
        try:
            embedding = self.model.encode([text])
            # Convert to list of floats (the model returns numpy arrays)
            return embedding[0].tolist()
        except Exception as e:
            self.logger.error(f"Error generating embedding for text: {str(e)}")
            raise

    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts using Sentence Transformers.
        Sentence Transformers is efficient with batching.

        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch

        Returns:
            List of embedding vectors
        """
        embeddings = []

        # Process in batches to manage memory usage
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

            try:
                batch_embeddings = self.model.encode(batch)
                # Convert numpy arrays to lists of floats
                batch_embeddings_list = [emb.tolist() for emb in batch_embeddings]
                embeddings.extend(batch_embeddings_list)
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
        return self.model.get_sentence_embedding_dimension()


def create_embedding_for_content(text: str, api_key: Optional[str] = None) -> List[float]:
    """
    Convenience function to create a single embedding for content.
    NOTE: api_key parameter is kept for compatibility but not used.

    Args:
        text: Text to embed
        api_key: Not used, kept for compatibility

    Returns:
        Embedding vector as a list of floats
    """
    service = GoogleEmbeddingService(api_key=api_key)
    return service.generate_embedding(text)


def create_embeddings_for_contents(texts: List[str], api_key: Optional[str] = None, batch_size: int = 32) -> List[List[float]]:
    """
    Convenience function to create embeddings for multiple contents.
    NOTE: api_key parameter is kept for compatibility but not used.

    Args:
        texts: List of texts to embed
        api_key: Not used, kept for compatibility
        batch_size: Batch size for processing

    Returns:
        List of embedding vectors
    """
    service = GoogleEmbeddingService(api_key=api_key)
    return service.generate_embeddings_batch(texts, batch_size=batch_size)