"""
Index all book content from docs/ directory (48 chapters across 4 modules) to Qdrant
"""
import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any
import asyncio

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
    docs_path = os.path.join(os.path.dirname(__file__), '..', '..', 'website', 'docs')

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
        # Index all valid chunks
        for i, chunk_data in enumerate(valid_chunks_with_embeddings):
            if i % 50 == 0:  # Log progress every 50 chunks
                logger.info(f"Indexed {i}/{len(valid_chunks_with_embeddings)} chunks...")

            # Prepare the point for Qdrant
            point_data = {
                'id': chunk_data['id'],
                'vector': chunk_data['embedding'],
                'payload': {
                    'content': chunk_data['text'],
                    'module': chunk_data['metadata']['module'],
                    'chapter': chunk_data['metadata']['chapter'],
                    'page_url': chunk_data['metadata']['url'],
                    'heading': chunk_data['metadata']['heading'],
                    'difficulty': chunk_data['metadata']['difficulty'],
                    'metadata': {}
                }
            }

            # Add to Qdrant collection
            qdrant_service.add_point(point_data)

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