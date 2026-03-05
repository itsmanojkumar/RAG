from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Sentence Transformers (local embeddings)
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Lightweight, fast, 80MB
    
    # Reranking - using Llama inference via HuggingFace
    RERANK_TOP_K: int = 3  # Return top 3 after reranking
    RERANK_LLM_URL: str = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct"  # Llama model for reranking
    
    # HF Inference API for LLM only
    HF_TOKEN: str = ""
    HF_INFERENCE_URL: str = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

    # Vector store
    VECTOR_STORE: Literal["faiss", "pinecone"] = "faiss"
    # Base path for FAISS index directory (LangChain will create a folder here)
    FAISS_INDEX_PATH: str = "data/faiss.index"
    FAISS_METADATA_PATH: str = "data/faiss_metadata.json"
    PINECONE_API_KEY: str = ""
    PINECONE_INDEX: str = "rag-index"
    PINECONE_ENV: str = ""

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # Chunking & retrieval
    UPLOAD_DIR: str = "uploads"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 10  # Initial retrieval before reranking
    
    # Production limits
    MAX_QUERY_LENGTH: int = 1000  # Max characters in query
    MAX_CONTEXT_LENGTH: int = 8000  # Max characters in context sent to LLM
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50 MB
    REQUEST_TIMEOUT: float = 300.0  # 5 minutes max request time
    MIN_DISK_SPACE_MB: int = 100  # Minimum free disk space required (MB)

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_JSON: bool = False  # True in production for JSON-structured logs

    # Production
    ENVIRONMENT: Literal["development", "production"] = "development"
    CORS_ORIGINS: str = "*"  # Comma-separated; use specific origins in production
    HTTP_TIMEOUT: float = 60.0  # Seconds for HF embedding calls
    HTTP_LLM_TIMEOUT: float = 120.0  # Seconds for HF text-generation
    HTTP_RETRIES: int = 3  # Retries for HF API calls

    @model_validator(mode="after")
    def validate_settings(self) -> "Settings":
        """Validate settings for production readiness."""
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError("CHUNK_OVERLAP must be < CHUNK_SIZE")
        
        # Production validation
        if self.ENVIRONMENT == "production":
            if not self.HF_TOKEN:
                raise ValueError("HF_TOKEN is required in production")
            if self.CORS_ORIGINS == "*":
                raise ValueError("CORS_ORIGINS cannot be '*' in production. Set specific origins.")
            if self.VECTOR_STORE == "pinecone" and not self.PINECONE_API_KEY:
                raise ValueError("PINECONE_API_KEY is required when VECTOR_STORE=pinecone")
            if self.MAX_CONTEXT_LENGTH > 10000:
                import warnings
                warnings.warn("MAX_CONTEXT_LENGTH > 10000 may cause token limit issues with some models")
        
        return self

    def faiss_index_path(self) -> Path:
        return Path(self.FAISS_INDEX_PATH)

    def faiss_metadata_path(self) -> Path:
        return Path(self.FAISS_METADATA_PATH)

    def upload_dir_path(self) -> Path:
        return Path(self.UPLOAD_DIR)

    @property
    def cors_origins_list(self) -> list[str]:
        return [x.strip() for x in self.CORS_ORIGINS.split(",") if x.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
