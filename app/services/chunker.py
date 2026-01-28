"""Text chunking using LangChain RecursiveCharacterTextSplitter."""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_text_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Get or create a cached RecursiveCharacterTextSplitter instance."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_text(text: str, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[str]:
    """Split text into overlapping chunks using LangChain RecursiveCharacterTextSplitter."""
    s = get_settings()
    size = chunk_size if chunk_size is not None else s.CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else s.CHUNK_OVERLAP
    
    splitter = _get_text_splitter(size, overlap)
    chunks = splitter.split_text(text.strip())
    
    # Filter out empty chunks
    return [chunk for chunk in chunks if chunk.strip()]
