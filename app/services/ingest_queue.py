"""Enqueue ingest jobs (ARQ) and get job status (Redis)."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

REDIS_PREFIX = "rag:job:"
STATUS_TTL = 86400  # 1 day

# In-memory job status for development without Redis
_memory_jobs: dict[str, dict] = {}


def _run_ingest_in_thread(job_id: str, document_id: str, file_path: str) -> None:
    """Run ingest in a background thread."""
    import asyncio
    from app.services.chunker import chunk_text
    from app.services.embeddings import embed
    from app.services.parser import extract_text
    from app.services.store import add_vectors

    try:
        text = extract_text(file_path)
        if not text.strip():
            _memory_jobs[job_id] = {"status": "failed", "message": "No text extracted"}
            return

        chunks = chunk_text(text)
        if not chunks:
            _memory_jobs[job_id] = {"status": "failed", "message": "No chunks produced"}
            return

        metadata = [{"text": c, "source": document_id} for c in chunks]
        
        # Run async embed in a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            vectors, _ = loop.run_until_complete(embed(chunks))
        finally:
            loop.close()
        
        if len(vectors) != len(metadata):
            _memory_jobs[job_id] = {"status": "failed", "message": "Embedding count mismatch"}
            return

        add_vectors(vectors, metadata)
        _memory_jobs[job_id] = {"status": "completed", "message": None}
        logger.info("Background ingest completed: job_id=%s doc=%s chunks=%d", job_id, document_id, len(chunks))
    except Exception as e:
        logger.exception("Background ingest failed: job_id=%s", job_id)
        _memory_jobs[job_id] = {"status": "failed", "message": str(e)}


async def sync_ingest(document_id: str, file_path: str) -> str:
    """Start ingest in background thread and return job_id immediately."""
    import threading
    
    job_id = str(uuid.uuid4())
    _memory_jobs[job_id] = {"status": "processing", "message": None}
    
    # Start processing in background thread
    thread = threading.Thread(
        target=_run_ingest_in_thread,
        args=(job_id, document_id, file_path),
        daemon=True
    )
    thread.start()
    
    logger.info("Started background ingest: job_id=%s doc=%s", job_id, document_id)
    return job_id


async def enqueue_ingest(document_id: str, file_path: str) -> str:
    """Enqueue ingest job; set status pending; return job_id."""
    import asyncio
    import redis.asyncio as aioredis
    from arq.connections import ArqRedis, RedisSettings, create_pool

    settings = get_settings()
    job_id = str(uuid.uuid4())
    key = f"{REDIS_PREFIX}{job_id}"

    # Use short timeouts to fail fast if Redis unavailable
    redis = aioredis.from_url(
        settings.REDIS_URL, 
        decode_responses=True,
        socket_connect_timeout=1,
        socket_timeout=1
    )
    try:
        await asyncio.wait_for(
            redis.set(key, json.dumps({"status": "pending", "message": None}), ex=STATUS_TTL),
            timeout=2
        )
    finally:
        await redis.aclose()

    rs = RedisSettings.from_dsn(settings.REDIS_URL)
    pool: ArqRedis = await asyncio.wait_for(create_pool(rs), timeout=2)
    try:
        await pool.enqueue_job(
            "ingest_task",
            document_id,
            file_path,
            _job_id=job_id,
        )
    finally:
        await pool.close()

    return job_id


async def set_job_status(job_id: str, status: str, message: str | None = None) -> None:
    """Update job status (used by worker)."""
    import redis.asyncio as aioredis

    settings = get_settings()
    key = f"{REDIS_PREFIX}{job_id}"
    redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        await redis.set(key, json.dumps({"status": status, "message": message}), ex=STATUS_TTL)
    finally:
        await redis.aclose()


async def get_job_status(job_id: str) -> dict[str, Any] | None:
    """Return {'status': '...', 'message': '...'} or None if not found."""
    # Check in-memory jobs first (for development without Redis)
    if job_id in _memory_jobs:
        return _memory_jobs[job_id]
    
    # Try Redis
    try:
        import redis.asyncio as aioredis

        settings = get_settings()
        key = f"{REDIS_PREFIX}{job_id}"
        redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True, socket_connect_timeout=1, socket_timeout=1)
        try:
            raw = await redis.get(key)
        finally:
            await redis.aclose()
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return {"status": "unknown", "message": raw}
    except Exception:
        # Redis unavailable
        return None
