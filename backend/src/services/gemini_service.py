import logging
import os
from google import genai
from google.genai import types
from typing import List, Optional
from src.config import settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class GeminiService:
    """Service for interacting with Google Gemini API (New SDK)"""

    def __init__(self):
        # 1. Initialize the new Client
        self.api_key = settings.gemini_api_key
        
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
            
            # --- MODEL CONFIGURATION ---
            # We use 'gemini-flash-latest' because:
            # 1. It is stable (fixes 404 errors).
            # 2. It has a high free tier quota (fixes 429 errors).
            
            target_model = getattr(settings, 'gemini_model_name', 'gemini-flash-latest')
            
            # List of models known to cause issues on free tiers or v1beta
            unsafe_models = ['gemini-pro', 'gemini-1.5-flash', 'gemini-1.5-flash-001', 'gemini-2.0-flash']
            
            if target_model in unsafe_models:
                self.model_name = 'gemini-flash-latest'
            else:
                self.model_name = target_model
                
            logger.info(f"Initialized Gemini service with model: {self.model_name}")
        else:
            logger.warning("GEMINI_API_KEY not set. LLM functionality will not work.")
            self.client = None

        # Initialize Sentence Transformer for embeddings
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info(f"Initialized Sentence Transformer for embeddings.")

    def generate_response(
        self,
        context_chunks: List[str],
        question: str,
        selected_text: Optional[str] = None
    ) -> str:
        """Generate a response using the new SDK syntax."""
        if not self.client:
            return "Error: API Key missing."

        # Combine context
        context = "\n\n".join(context_chunks)

        # --- IMPROVED PROMPT ---
        # This instructs the AI to be conversational for greetings, 
        # but strict for technical questions.
        base_instructions = """
        You are an intelligent assistant for the 'Physical AI & Humanoid Robotics' book.
        
        GUIDELINES:
        1. GREETINGS: If the user says "hello", "hi", or similar, reply naturally as a helpful assistant (e.g., "Hello! I can help you answer questions about the book content.").
        2. KNOWLEDGE: For all other questions, answer based ONLY on the provided Context.
        3. HONESTY: If the answer is not in the Context, explicitly state that the information is not covered in the book.
        """

        if selected_text:
            prompt = f"""
            {base_instructions}
            
            QUERY CONTEXT:
            Selected text: {selected_text}
            Book Context: {context}
            
            USER QUESTION: {question}
            
            ANSWER:
            """
        else:
            prompt = f"""
            {base_instructions}
            
            QUERY CONTEXT:
            Book Context: {context}
            
            USER QUESTION: {question}
            
            ANSWER:
            """

        try:
            # Call Gemini API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3, # Slightly higher for more natural language
                    max_output_tokens=2000,
                    candidate_count=1
                )
            )

            if response.text:
                return response.text
            else:
                return "I couldn't generate a response."

        except Exception as e:
            logger.error(f"Error generating response from Gemini: {e}")
            return "Sorry, I encountered an error while processing your request."

    def embed_text(self, text: str) -> List[float]:
        """Generate embeddings (Unchanged)."""
        try:
            embedding = self.embedding_model.encode([text])
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return [0.0] * 384

    def embed_multiple_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate multiple embeddings."""
        try:
            embeddings = self.embedding_model.encode(texts)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []