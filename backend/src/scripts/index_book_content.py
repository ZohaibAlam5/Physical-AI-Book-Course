"""
Index all book content from docs/ directory (48 chapters across 4 modules) to Qdrant
"""
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
import asyncio

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add the backend src directory to the path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.chunking_util import ContentChunker
from src.utils.google_embedding_util import GoogleEmbeddingService
from src.services.qdrant_service import QdrantService


def setup_logging():
    """Set up logging for the indexing process."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('indexing.log'),
            logging.StreamHandler()
        ]
    )


def main():
    """Main indexing function."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting book content indexing process...")

    # Initialize services
    try:
        qdrant_service = QdrantService()
        embedding_service = GoogleEmbeddingService()
        chunker = ContentChunker(max_tokens=600, overlap_tokens=50)

        logger.info("Services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        return

    # Get the path to the docs directory (relative to the website folder)
    # Use absolute path to ensure it works correctly
    backend_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    docs_path = os.path.join(backend_root, '..', 'website', 'docs')
    docs_path = os.path.abspath(docs_path)  # Convert to absolute path

    if not os.path.exists(docs_path):
        logger.error(f"Docs directory not found at {docs_path}")
        return

    logger.info(f"Processing content from {docs_path}")

    # Process all markdown files in the docs directory
    try:
        chunks = chunker.process_directory(docs_path)
        logger.info(f"Generated {len(chunks)} content chunks")
    except Exception as e:
        logger.error(f"Error processing directory: {str(e)}")
        return

    if not chunks:
        logger.warning("No chunks were generated. Check the docs directory content.")
        return

    # Extract text content for embedding
    texts = [chunk['text'] for chunk in chunks]

    logger.info("Generating embeddings for content chunks...")

    try:
        # Generate embeddings in batches
        embeddings = embedding_service.generate_embeddings_batch(texts, batch_size=5)
        logger.info(f"Generated {len(embeddings)} embeddings")
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        return

    # Validate embeddings
    valid_chunks_with_embeddings = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        if embedding and len(embedding) > 0:
            # Add the embedding to the chunk
            chunk_with_embedding = chunk.copy()
            chunk_with_embedding['embedding'] = embedding
            valid_chunks_with_embeddings.append(chunk_with_embedding)
        else:
            logger.warning(f"Empty embedding for chunk {i}, skipping: {chunk['id']}")

    logger.info(f"Successfully prepared {len(valid_chunks_with_embeddings)} chunks with embeddings for indexing")

    # Index to Qdrant
    logger.info("Starting indexing to Qdrant...")

    try:
        # Prepare all chunks for upsertion
        formatted_chunks = []
        for i, chunk_data in enumerate(valid_chunks_with_embeddings):
            if i % 50 == 0:  # Log progress every 50 chunks
                logger.info(f"Preparing {i}/{len(valid_chunks_with_embeddings)} chunks for indexing...")

            # Prepare the point for Qdrant
            formatted_chunk = {
                'id': chunk_data['id'],
                'text': chunk_data['text'],
                'embedding': chunk_data['embedding'],
                'metadata': {
                    'module': chunk_data['metadata']['module'],
                    'chapter': chunk_data['metadata']['chapter'],
                    'url': chunk_data['metadata']['url'],
                    'heading': chunk_data['metadata']['heading'],
                    'difficulty': chunk_data['metadata']['difficulty'],
                }
            }
            formatted_chunks.append(formatted_chunk)

        # Upsert all chunks at once for better performance
        qdrant_service.upsert_chunks(formatted_chunks)
        logger.info(f"Successfully indexed {len(valid_chunks_with_embeddings)} chunks to Qdrant")

    except Exception as e:
        logger.error(f"Error indexing to Qdrant: {str(e)}")
        return

    logger.info("Indexing process completed successfully!")

    # Print summary
    modules = set(chunk['metadata']['module'] for chunk in valid_chunks_with_embeddings)
    chapters = set(f"{chunk['metadata']['module']}/{chunk['metadata']['chapter']}"
                   for chunk in valid_chunks_with_embeddings)

    logger.info(f"Indexed content from {len(modules)} modules: {', '.join(sorted(modules))}")
    logger.info(f"Indexed content from {len(chapters)} chapters")
    logger.info(f"Total chunks indexed: {len(valid_chunks_with_embeddings)}")


if __name__ == "__main__":
    main()