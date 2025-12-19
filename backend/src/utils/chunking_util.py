"""
Content chunking utility for processing book markdown files into vector store format.
Uses Google embedding models to create semantic chunks of book content with metadata.
"""
import os
import re
from typing import List, Dict, Any
from pathlib import Path
import markdown
from bs4 import BeautifulSoup
import tiktoken
from google.generativeai import embedding_utils


class ContentChunker:
    """
    Utility class for chunking book content into semantic sections suitable for vector storage.
    Each chunk includes metadata about the source (module, chapter, page URL, heading, difficulty).
    """

    def __init__(self, max_tokens: int = 600, overlap_tokens: int = 50):
        """
        Initialize the chunker with token limits.

        Args:
            max_tokens: Maximum tokens per chunk (500-800 range as per spec)
            overlap_tokens: Number of tokens to overlap between chunks for continuity
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.enc = tiktoken.get_encoding("cl100k_base")  # Good for most models including Google's

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        return len(self.enc.encode(text))

    def extract_metadata_from_path(self, file_path: str) -> Dict[str, str]:
        """
        Extract metadata from the file path following the book structure.
        Expected format: docs/module-{n}/chapter-{n}/filename.md
        """
        path_parts = Path(file_path).parts

        # Find the 'docs' directory and extract module/chapter info
        docs_index = -1
        for i, part in enumerate(path_parts):
            if part == 'docs':
                docs_index = i
                break

        if docs_index == -1:
            return {
                'module': 'unknown',
                'chapter': 'unknown',
                'url': '/docs/unknown',
                'heading': 'Unknown',
                'difficulty': 'intermediate'  # Default difficulty
            }

        # Extract module and chapter from path
        remaining_parts = path_parts[docs_index + 1:]

        module = 'unknown'
        chapter = 'unknown'
        heading = 'Unknown'

        for part in remaining_parts:
            if part.startswith('module-'):
                module = part
            elif part.startswith('chapter-'):
                chapter = part
            elif part.endswith('.md'):
                heading = part.replace('.md', '').replace('-', ' ').title()
                break

        # Create URL path
        url_parts = [f'/{part}' for part in remaining_parts if not part.endswith('.md')]
        url_parts.append(f'/{heading.lower().replace(" ", "-")}')
        url = ''.join(url_parts) if url_parts else f'/docs/{module}/{chapter}'

        return {
            'module': module,
            'chapter': chapter,
            'url': url,
            'heading': heading,
            'difficulty': 'intermediate'  # Default, could be enhanced based on content analysis
        }

    def split_markdown_by_headings(self, content: str) -> List[str]:
        """
        Split markdown content by headings to preserve semantic boundaries.
        """
        # Split content by markdown headings (# ## ###)
        heading_pattern = r'^(#{1,6})\s+(.+)$'
        lines = content.split('\n')

        sections = []
        current_section = []

        for line in lines:
            if re.match(heading_pattern, line.strip()):
                # If we have accumulated content, save it as a section
                if current_section:
                    sections.append('\n'.join(current_section))
                # Start new section with the heading
                current_section = [line]
            else:
                current_section.append(line)

        # Add the last section if it exists
        if current_section:
            sections.append('\n'.join(current_section))

        # Filter out empty sections
        sections = [section.strip() for section in sections if section.strip()]

        return sections

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a text into semantic pieces with metadata.

        Args:
            text: Text to chunk
            metadata: Metadata to include with each chunk

        Returns:
            List of chunks with text, metadata, and generated IDs
        """
        chunks = []

        # Split by semantic boundaries (headings)
        sections = self.split_markdown_by_headings(text)

        for section in sections:
            # If section is small enough, use as-is
            if self.count_tokens(section) <= self.max_tokens:
                chunk_id = self.generate_chunk_id(metadata, len(chunks))
                chunks.append({
                    'id': chunk_id,
                    'text': section.strip(),
                    'metadata': metadata
                })
            else:
                # Section is too large, need to split further
                sub_chunks = self.split_large_section(section)
                for sub_chunk in sub_chunks:
                    chunk_id = self.generate_chunk_id(metadata, len(chunks))
                    chunks.append({
                        'id': chunk_id,
                        'text': sub_chunk.strip(),
                        'metadata': metadata
                    })

        return chunks

    def split_large_section(self, text: str) -> List[str]:
        """
        Split a large section of text into smaller chunks while maintaining sentence boundaries.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Test if adding this sentence would exceed token limit
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if self.count_tokens(test_chunk) <= self.max_tokens:
                current_chunk = test_chunk
            else:
                # If current chunk is not empty, save it and start a new one
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # If the single sentence is too long, split it by length
                if self.count_tokens(sentence) > self.max_tokens:
                    sub_chunks = self.split_by_length(sentence)
                    chunks.extend(sub_chunks[:-1])  # Add all but the last chunk
                    current_chunk = sub_chunks[-1]  # Keep the last part as current chunk
                else:
                    current_chunk = sentence

        # Add the final chunk if it exists
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def split_by_length(self, text: str) -> List[str]:
        """
        Split text by character length as a fallback when sentences are too long.
        """
        chunks = []
        tokens = self.enc.encode(text)

        for i in range(0, len(tokens), self.max_tokens):
            chunk_tokens = tokens[i:i + self.max_tokens]
            chunk_text = self.enc.decode(chunk_tokens)
            chunks.append(chunk_text)

        return chunks

    def generate_chunk_id(self, metadata: Dict[str, str], index: int) -> str:
        """
        Generate a stable chunk ID based on metadata and position.
        Format: chunk-{module}-{chapter}-{index}
        """
        module_clean = metadata['module'].replace('module-', 'm')
        chapter_clean = metadata['chapter'].replace('chapter-', 'c')
        return f"chunk-{module_clean}{chapter_clean}-{index:03d}"

    def process_markdown_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a single markdown file into chunks with metadata.

        Args:
            file_path: Path to the markdown file to process

        Returns:
            List of chunks with text, metadata, and IDs
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        metadata = self.extract_metadata_from_path(file_path)
        chunks = self.chunk_text(content, metadata)

        return chunks

    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Process all markdown files in a directory recursively.

        Args:
            directory_path: Path to directory containing markdown files

        Returns:
            List of all chunks from all files
        """
        all_chunks = []

        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith('.md'):
                    file_path = os.path.join(root, file)
                    file_chunks = self.process_markdown_file(file_path)
                    all_chunks.extend(file_chunks)

        return all_chunks


class EmbeddingGenerator:
    """
    Utility class for generating embeddings using Google's embedding models.
    """

    def __init__(self, model_name: str = "embedding-001"):
        """
        Initialize with Google's embedding model.

        Args:
            model_name: Name of the Google embedding model to use
        """
        self.model_name = model_name
        # The actual initialization will depend on Google's API
        # For now, we'll document the expected interface

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a text using Google's embedding model.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as a list of floats
        """
        # This is a placeholder - actual implementation would call Google's API
        # Example implementation when using google-generativeai:
        # import google.generativeai as genai
        # response = genai.embed_content(
        #     model=self.model_name,
        #     content=text,
        #     task_type="retrieval_document"
        # )
        # return response['embedding']

        # For now, return a mock embedding (this should be replaced with actual API call)
        # In a real implementation, this would call the Google API
        raise NotImplementedError(
            "Embedding generation requires Google API setup. "
            "Implement the actual API call using google.generativeai.embed_content"
        )

    def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embedding = self.generate_embedding(text)
            embeddings.append(embedding)
        return embeddings


def main():
    """
    Main function to demonstrate usage of the chunking utility.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Chunk book markdown files for RAG system')
    parser.add_argument('--input-dir', required=True, help='Directory containing markdown files')
    parser.add_argument('--output-file', help='Output file for chunks (JSON format)')
    parser.add_argument('--max-tokens', type=int, default=600, help='Maximum tokens per chunk')
    parser.add_argument('--overlap-tokens', type=int, default=50, help='Overlap tokens between chunks')

    args = parser.parse_args()

    chunker = ContentChunker(max_tokens=args.max_tokens, overlap_tokens=args.overlap_tokens)
    chunks = chunker.process_directory(args.input_dir)

    print(f"Processed {len(chunks)} chunks from {args.input_dir}")

    # Print summary of chunk sizes
    token_counts = [chunker.count_tokens(chunk['text']) for chunk in chunks]
    print(f"Average chunk size: {sum(token_counts) / len(token_counts):.1f} tokens")
    print(f"Max chunk size: {max(token_counts)} tokens")
    print(f"Min chunk size: {min(token_counts)} tokens")

    # Optionally save to file
    if args.output_file:
        import json
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        print(f"Chunks saved to {args.output_file}")


if __name__ == "__main__":
    main()