"""Embeddings using LangChain HuggingFaceEmbeddings."""

from __future__ import annotations

import asyncio
import logging
import time
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    """Load and cache the LangChain embedding model."""
    logger.info(f"Loading LangChain embedding model: {model_name}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )


async def embed(texts: list[str]) -> tuple[list[list[float]], float]:
    """Embed texts using LangChain HuggingFaceEmbeddings. Returns (vectors, elapsed_seconds)."""
    settings = get_settings()
    embeddings = _get_embedding_model(settings.EMBEDDING_MODEL)
    
    t0 = time.perf_counter()
    # Run in thread pool to avoid blocking
    vectors = await asyncio.to_thread(embeddings.embed_documents, texts)
    elapsed = time.perf_counter() - t0
    
    return vectors, elapsed


async def embed_single(text: str) -> tuple[list[float], float]:
    """Embed a single string. Returns (vector, elapsed_seconds)."""
    settings = get_settings()
    embeddings = _get_embedding_model(settings.EMBEDDING_MODEL)
    
    t0 = time.perf_counter()
    # Run in thread pool to avoid blocking
    vector = await asyncio.to_thread(embeddings.embed_query, text)
    elapsed = time.perf_counter() - t0
    
    return vector, elapsed
