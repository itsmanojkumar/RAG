"""Retrieve relevant chunks using LangChain retriever + reranking."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from app.config import get_settings
from app.services.embeddings import embed_single
from app.services.rerank import rerank_chunks
from app.services.store import ensure_faiss_loaded, _get_faiss_store, _get_pinecone_store

logger = logging.getLogger(__name__)


def _get_retriever():
    """Get LangChain retriever from vector store."""
    settings = get_settings()
    ensure_faiss_loaded()
    
    if settings.VECTOR_STORE == "faiss":
        store = _get_faiss_store()
    else:
        store = _get_pinecone_store()
    
    # Create retriever with TOP_K
    return store.as_retriever(
        search_kwargs={"k": settings.TOP_K},
        search_type="similarity",
    )


async def retrieve_chunks(question: str) -> tuple[list[dict[str, Any]], float, float, float]:
    """
    RAG Retrieval Pipeline: Retrieve → Rerank (using Llama) → Return top chunks.
    
    This function implements the retrieval and reranking steps of RAG:
    1. Embed the question
    2. Retrieve candidate chunks from vector store
    3. Rerank chunks using Llama inference (scores relevance 0-10)
    4. Return top K reranked chunks (these will be passed to LLM for answer generation)
    
    Returns (reranked_chunks, embed_time, retrieve_time, rerank_time).
    The reranked chunks are ready to be passed to the LLM for answer generation.
    """
    settings = get_settings()
    
    # Step 1: Embed the question (for timing/metrics)
    t0 = time.perf_counter()
    vec, embed_t = await embed_single(question)
    if len(vec) == 0:
        return [], embed_t, 0.0, 0.0

    # Step 2: Retrieve candidate chunks from vector store using LangChain retriever
    t1 = time.perf_counter()
    try:
        retriever = _get_retriever()
        # LangChain retriever API changed across versions:
        # - Newer: retriever.invoke(query) -> list[Document]
        # - Older: retriever.get_relevant_documents(query)
        if hasattr(retriever, "invoke"):
            docs = await asyncio.to_thread(retriever.invoke, question)
        else:
            docs = await asyncio.to_thread(retriever.get_relevant_documents, question)  # type: ignore[attr-defined]
        retrieve_t = time.perf_counter() - t1
    except Exception as e:
        logger.exception("Retrieval failed")
        retrieve_t = time.perf_counter() - t1
        return [], embed_t, retrieve_t, 0.0

    # Convert LangChain documents to our format
    chunks = []
    for doc in docs:
        text = doc.page_content if hasattr(doc, "page_content") else str(doc)
        if not text:
            continue
        metadata = doc.metadata if hasattr(doc, "metadata") else {}
        chunks.append({
            "text": text,
            "source": metadata.get("source"),
            "score": metadata.get("score", 0.0),
        })
    
    if not chunks:
        return [], embed_t, retrieve_t, 0.0
    
    # Step 3: Rerank chunks using Llama inference
    # Llama scores each chunk's relevance (0-10) and we return top K
    # These reranked chunks will be passed to the LLM for answer generation
    reranked_chunks, rerank_t = await rerank_chunks(question, chunks)
    
    logger.info(
        f"Retrieval pipeline: retrieved {len(chunks)} chunks, "
        f"reranked to top {len(reranked_chunks)} using Llama, "
        f"ready for LLM generation"
    )
    
    return reranked_chunks, embed_t, retrieve_t, rerank_t
