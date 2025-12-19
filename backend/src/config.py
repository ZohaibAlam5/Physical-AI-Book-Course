import os
from typing import Optional

# Load environment variables from .env file if it exists
from dotenv import load_dotenv
load_dotenv()


class Settings:
    """Configuration settings loaded from environment variables"""

    def __init__(self):
        # API Configuration
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8000"))
        self.api_debug = os.getenv("API_DEBUG", "True").lower() in ("true", "1", "yes")

        # Gemini Configuration (now only needed for the LLM, not embeddings)
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            print("WARNING: GEMINI_API_KEY environment variable is not set. This is required for the chatbot functionality, but embeddings will use Sentence Transformers.")

        self.gemini_model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-pro")

        # Qdrant Configuration
        self.qdrant_url = os.getenv("QDRANT_URL")
        if not self.qdrant_url:
            raise ValueError("QDRANT_URL environment variable is required")

        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME", "book_content_chunks")

        # Rate Limiting
        self.rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        self.rate_limit_window = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))  # 1 hour in seconds


settings = Settings()