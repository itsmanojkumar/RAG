from __future__ import annotations

import json
import logging
import time

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from app.config import get_settings
from app.limiter import limiter
from app.schemas.query import QueryRequest, QueryResponse, SourceChunk
from app.services.metrics import log_query_latency
from app.services.retrieval import retrieve_chunks
from app.services.llm import generate_answer, generate_answer_stream
from app.utils.security import validate_query_length, validate_context_length, sanitize_error_message

logger = logging.getLogger(__name__)
router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
@limiter.limit("20/minute")
async def query(request: Request, body: QueryRequest):
    """
    RAG Query Endpoint: Complete RAG pipeline.
    
    Flow:
    1. Retrieve chunks from vector store (retrieve_chunks)
    2. Rerank chunks using Llama inference (scoring relevance 0-10)
    3. Pass reranked chunks to LLM for answer generation (generate_answer)
    
    The reranked chunks are automatically passed to the LLM after reranking.
    """
    settings = get_settings()
    
    # Validate input
    try:
        validate_query_length(body.question, settings.MAX_QUERY_LENGTH)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        ) from e
    
    t0 = time.perf_counter()
    embed_t, retrieve_t, rerank_t, llm_t = 0.0, 0.0, 0.0, 0.0

    # Step 1: Retrieve and rerank chunks (using Llama for reranking)
    try:
        chunks, embed_t, retrieve_t, rerank_t = await retrieve_chunks(body.question)
    except Exception as e:
        logger.exception("Retrieval failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval failed. Have you ingested any documents?",
        ) from e

    if not chunks:
        return QueryResponse(
            answer="No relevant context found. Please ingest documents first, then try again.",
            sources=[],
        )

    # Step 2: Generate answer from reranked chunks
    # The chunks here are already reranked by Llama and filtered to top K
    try:
        answer, llm_t = await generate_answer(question=body.question, chunks=chunks)
        
        # Validate answer
        if not answer or not answer.strip():
            logger.warning("LLM returned empty answer, using fallback")
            answer = "I couldn't generate a proper answer. Please try rephrasing your question or ensure relevant documents are ingested."
        
    except ValueError as e:
        error_msg = str(e)
        if "HF_TOKEN" in error_msg:
            logger.error("HF_TOKEN not configured")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="LLM service not configured. Please set HF_TOKEN in environment variables.",
            ) from e
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Answer generation failed: {error_msg}",
        ) from e
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        logger.exception(f"LLM generation failed: {error_type}: {error_msg}")
        
        # Sanitize error message for production
        sanitized_msg = sanitize_error_message(e, settings.ENVIRONMENT)
        
        # Provide more specific error messages
        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            detail = "Request timed out. The model may be loading or overloaded. Please try again in a moment."
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            detail = "Rate limit exceeded. Please wait a moment and try again."
        elif "503" in error_msg or "loading" in error_msg.lower():
            detail = "Model is currently loading. Please wait a moment and try again."
        else:
            detail = sanitized_msg if settings.ENVIRONMENT == "production" else f"Answer generation failed: {error_msg[:200]}"
        
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        ) from e

    total = time.perf_counter() - t0
    logger.info(
        f"Query latency: embed={embed_t:.3f}s retrieve={retrieve_t:.3f}s "
        f"rerank={rerank_t:.3f}s llm={llm_t:.3f}s total={total:.3f}s"
    )
    log_query_latency(embed_s=embed_t, retrieve_s=retrieve_t, llm_s=llm_t, total_s=total)

    # Build sources with proper validation
    sources = []
    for c in chunks:
        try:
            source = SourceChunk(
                text=c.get("text", "")[:1000],  # Limit text length
                source=c.get("source"),
                score=c.get("score") or c.get("rerank_score", 0.0)
            )
            sources.append(source)
        except Exception as e:
            logger.warning(f"Error creating source chunk: {e}")
            continue
    
    # Ensure we have at least an answer
    if not answer:
        answer = "No answer could be generated. Please try again."
    
    return QueryResponse(answer=answer, sources=sources)


@router.post("/query/stream")
@limiter.limit("20/minute")
async def query_stream(request: Request, body: QueryRequest):
    """
    RAG Query Endpoint with Streaming: Returns answer as Server-Sent Events (SSE).
    
    Flow:
    1. Retrieve and rerank chunks
    2. Stream LLM response token by token
    3. Send sources at the end
    
    SSE Events:
    - data: {"type": "token", "content": "..."}  - Answer tokens
    - data: {"type": "sources", "sources": [...]} - Source chunks
    - data: {"type": "done"} - Stream complete
    - data: {"type": "error", "message": "..."} - Error occurred
    """
    settings = get_settings()
    
    # Validate input
    try:
        validate_query_length(body.question, settings.MAX_QUERY_LENGTH)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        ) from e
    
    async def event_generator():
        try:
            # Step 1: Retrieve and rerank chunks
            chunks, embed_t, retrieve_t, rerank_t = await retrieve_chunks(body.question)
            
            if not chunks:
                yield f"data: {json.dumps({'type': 'token', 'content': 'No relevant context found. Please ingest documents first.'})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return
            
            # Step 2: Stream the answer
            async for token in generate_answer_stream(question=body.question, chunks=chunks):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            
            # Step 3: Send sources
            sources = []
            for c in chunks:
                sources.append({
                    "text": c.get("text", "")[:1000],
                    "source": c.get("source"),
                    "score": c.get("score") or c.get("rerank_score", 0.0)
                })
            
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            error_msg = str(e)
            logger.exception(f"Stream error: {error_msg}")
            yield f"data: {json.dumps({'type': 'error', 'message': error_msg[:200]})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )
