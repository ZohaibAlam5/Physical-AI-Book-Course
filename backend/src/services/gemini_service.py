import logging
import google.generativeai as genai
from typing import List, Optional
from src.config import settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class GeminiService:
    """Service for interacting with Google Gemini API"""

    def __init__(self):
        # Configure the API key for LLM
        if settings.gemini_api_key:
            genai.configure(api_key=settings.gemini_api_key)
            # Initialize the model
            self.model_name = settings.gemini_model_name
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Initialized Gemini service with model: {self.model_name}")
        else:
            logger.warning("GEMINI_API_KEY not set. LLM functionality will not work.")
            self.model = None

        # Initialize Sentence Transformer for embeddings
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info(f"Initialized Sentence Transformer for embeddings. Dimension: {self.embedding_model.get_sentence_embedding_dimension()}")

        self.model_name = settings.gemini_model_name

    def generate_response(
        self,
        context_chunks: List[str],
        question: str,
        selected_text: Optional[str] = None
    ) -> str:
        """
        Generate a response based on the provided context chunks and question.

        Args:
            context_chunks: List of text chunks that provide context for the answer
            question: The question to answer
            selected_text: Optional selected text for selection-specific queries

        Returns:
            Generated response string
        """
        # Combine context chunks into a single context string
        context = "\n\n".join(context_chunks)

        # Determine the query type and construct the prompt accordingly
        if selected_text:
            # Selection-specific query
            prompt = f"""
            You are an assistant for the Physical AI & Humanoid Robotics book.
            Answer the user's question based ONLY on the provided context from selected text.
            If the answer is not in the provided context, explicitly state that the information is not available in the selected text.

            Selected text: {selected_text}

            Context: {context}

            Question: {question}

            Answer:
            """
        else:
            # Global or page-specific query
            prompt = f"""
            You are an assistant for the Physical AI & Humanoid Robotics book.
            Answer the user's question based ONLY on the provided context from the book.
            If the answer is not in the provided context, explicitly state that the information is not available in the book.

            Context: {context}

            Question: {question}

            Answer:
            """

        try:
            # Generate content using the model
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.1,  # Low temperature for more consistent, factual responses
                    "max_output_tokens": 2000,
                    "candidate_count": 1
                }
            )

            # Extract and return the text
            if response.candidates and len(response.candidates) > 0:
                return response.candidates[0].content.parts[0].text
            else:
                logger.warning("No candidates returned from Gemini API")
                return "I couldn't generate a response based on the provided context."

        except Exception as e:
            logger.error(f"Error generating response from Gemini: {e}")
            return "Sorry, I encountered an error while processing your request."

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embeddings for the given text using Sentence Transformers.

        Args:
            text: The text to embed

        Returns:
            List of embedding values
        """
        try:
            embedding = self.embedding_model.encode([text])
            # Convert to list of floats (the model returns numpy arrays)
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings with Sentence Transformers: {e}")
            # Return a zero vector as fallback
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            return [0.0] * embedding_dim

    def embed_multiple_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using Sentence Transformers.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.embedding_model.encode(texts)
            # Convert numpy arrays to lists of floats
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Error generating embeddings for multiple texts with Sentence Transformers: {e}")
            # Fallback to individual embedding generation
            embeddings = []
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            for text in texts:
                try:
                    embedding = self.embedding_model.encode([text])
                    embeddings.append(embedding[0].tolist())
                except:
                    # Return a zero vector as fallback for this text
                    embeddings.append([0.0] * embedding_dim)
            return embeddings