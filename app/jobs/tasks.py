"""Ingest task: parse -> chunk -> embed -> store."""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import get_settings
from app.services.chunker import chunk_text
from app.services.embeddings import embed
from app.services.parser import extract_text
from app.services.store import add_vectors
from app.services.ingest_queue import set_job_status

logger = logging.getLogger(__name__)


async def ingest_task(ctx: dict, document_id: str, file_path: str) -> None:
    """ARQ task: parse PDF/TXT, chunk, embed via HF API, add to FAISS/Pinecone."""
    job_id = ctx["job_id"]
    await set_job_status(job_id, "processing", None)
    try:
        text = extract_text(file_path)
        if not text.strip():
            await set_job_status(job_id, "failed", "No text extracted")
            return

        chunks = chunk_text(text)
        if not chunks:
            await set_job_status(job_id, "failed", "No chunks produced")
            return

        metadata = [{"text": c, "source": document_id} for c in chunks]
        vectors, _ = await embed(chunks)
        if len(vectors) != len(metadata):
            await set_job_status(job_id, "failed", "Embedding count mismatch")
            return

        add_vectors(vectors, metadata)
        await set_job_status(job_id, "completed", None)
        logger.info("Ingest completed: job_id=%s doc=%s chunks=%d", job_id, document_id, len(chunks))
    except Exception as e:
        logger.exception("Ingest failed: job_id=%s", job_id)
        await set_job_status(job_id, "failed", str(e))
