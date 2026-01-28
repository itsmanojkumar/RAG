"""Simple reranking based on keyword matching and similarity.

Avoids external API calls for better reliability.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)


def _simple_score(question: str, chunk_text: str) -> float:
    """Score chunk based on keyword overlap with question."""
    question_words = set(question.lower().split())
    chunk_words = set(chunk_text.lower().split())
    
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                  'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                  'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
                  'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                  'through', 'during', 'before', 'after', 'above', 'below',
                  'between', 'under', 'again', 'further', 'then', 'once',
                  'what', 'which', 'who', 'whom', 'this', 'that', 'these',
                  'those', 'am', 'and', 'but', 'if', 'or', 'because', 'until',
                  'while', 'how', 'all', 'each', 'few', 'more', 'most', 'other',
                  'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                  'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here'}
    
    question_words = question_words - stop_words
    chunk_words = chunk_words - stop_words
    
    if not question_words:
        return 0.0
    
    # Calculate overlap score
    overlap = question_words & chunk_words
    score = len(overlap) / len(question_words) * 10  # Scale to 0-10
    
    return min(10.0, score)


async def rerank_chunks(
    question: str, 
    chunks: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], float]:
    """Rerank chunks using simple keyword matching. Returns (reranked_chunks, elapsed_seconds)."""
    if not chunks:
        return [], 0.0
    
    settings = get_settings()
    t0 = time.perf_counter()
    
    # Score each chunk
    for chunk in chunks:
        chunk["rerank_score"] = _simple_score(question, chunk["text"])
    
    # Sort by rerank score descending
    reranked = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    
    elapsed = time.perf_counter() - t0
    
    # Return top K after reranking
    top_k = settings.RERANK_TOP_K
    scores_str = ", ".join([f"{c['rerank_score']:.2f}" for c in reranked[:top_k]])
    logger.info(f"Reranked {len(chunks)} chunks, returning top {top_k}. Scores: {scores_str}")
    
    return reranked[:top_k], elapsed
