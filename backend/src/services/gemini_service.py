import logging
import google.generativeai as genai
from typing import List, Optional
from src.config import settings

logger = logging.getLogger(__name__)


class GeminiService:
    """Service for interacting with Google Gemini API"""

    def __init__(self):
        # Configure the API key
        genai.configure(api_key=settings.gemini_api_key)

        # Initialize the model
        self.model_name = settings.gemini_model_name
        self.model = genai.GenerativeModel(self.model_name)

        logger.info(f"Initialized Gemini service with model: {self.model_name}")

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
        Generate embeddings for the given text using Google's embedding model.

        Args:
            text: The text to embed

        Returns:
            List of embedding values
        """
        try:
            result = genai.embed_content(
                model="models/embedding-001",  # Google's embedding model
                content=text,
                task_type="retrieval_document"  # Appropriate task type for document retrieval
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error generating embeddings from Gemini: {e}")
            # Return a zero vector as fallback
            return [0.0] * 768  # Assuming 768-dim embeddings

    def embed_multiple_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_text(text))
        return embeddings