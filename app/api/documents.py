from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.config import get_settings
from app.limiter import limiter
from app.schemas.documents import DocumentInfo, DocumentListResponse, JobStatusResponse, UploadResponse
from app.services.ingest_queue import enqueue_ingest, get_job_status, sync_ingest
from app.services.store import delete_document
from app.utils.security import sanitize_filename, check_disk_space

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".txt"}


def _validate_file(file: UploadFile) -> None:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Allowed formats: PDF, TXT. Got: {ext or 'unknown'}",
        )


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
@limiter.limit("10/minute")
async def upload_document(request: Request, file: UploadFile = File(...)):
    """Upload a PDF or TXT document. Ingestion runs in a background job."""
    settings = get_settings()
    
    _validate_file(file)
    
    # Check disk space before reading file
    upload_dir = settings.upload_dir_path()
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    if not check_disk_space(upload_dir, settings.MIN_DISK_SPACE_MB):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Insufficient disk space. Need at least {settings.MIN_DISK_SPACE_MB} MB free.",
        )
    
    content = await file.read()
    if len(content) > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large (max {settings.MAX_FILE_SIZE // (1024*1024)} MB)",
        )

    # Sanitize filename
    original_filename = file.filename or "unnamed"
    sanitized_filename = sanitize_filename(original_filename)
    
    document_id = str(uuid.uuid4())
    ext = Path(sanitized_filename).suffix.lower() or Path(original_filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        ext = ".txt"  # Default fallback
    
    save_path = upload_dir / f"{document_id}{ext}"
    save_path.write_bytes(content)

    try:
        job_id = await enqueue_ingest(document_id=document_id, file_path=str(save_path))
    except Exception as e:
        logger.warning("Redis unavailable, using synchronous ingest: %s", e)
        try:
            # Fallback to synchronous processing when Redis is unavailable
            job_id = await sync_ingest(document_id=document_id, file_path=str(save_path))
        except Exception as sync_error:
            logger.exception("Synchronous ingest also failed")
            save_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Ingestion failed: {str(sync_error)}",
            ) from sync_error

    return UploadResponse(job_id=job_id, document_id=document_id)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def job_status(job_id: str):
    """Get status of an ingestion job."""
    status_info = await get_job_status(job_id)
    if status_info is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return JobStatusResponse(
        job_id=job_id,
        status=status_info["status"],
        message=status_info.get("message"),
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents():
    """List all uploaded documents."""
    from datetime import datetime
    
    settings = get_settings()
    upload_dir = settings.upload_dir_path()
    
    if not upload_dir.exists():
        return DocumentListResponse(documents=[])
    
    documents = []
    for file_path in upload_dir.glob("*"):
        if file_path.is_file() and file_path.suffix.lower() in ALLOWED_EXTENSIONS:
            stat = file_path.stat()
            doc_id = file_path.stem
            documents.append(
                DocumentInfo(
                    document_id=doc_id,
                    filename=file_path.name,
                    size_bytes=stat.st_size,
                    uploaded_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                )
            )
    
    # Sort by upload time, newest first
    documents.sort(key=lambda d: d.uploaded_at, reverse=True)
    return DocumentListResponse(documents=documents)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_uploaded_document(document_id: str):
    """Delete an uploaded document and best-effort remove its vectors from the store."""
    settings = get_settings()
    upload_dir = settings.upload_dir_path()

    # Delete any matching uploaded files (pdf/txt)
    deleted_file = False
    for ext in ALLOWED_EXTENSIONS:
        p = upload_dir / f"{document_id}{ext}"
        if p.exists():
            p.unlink(missing_ok=True)
            deleted_file = True

    # Remove from vector store (best-effort; may raise if misconfigured)
    try:
        delete_document(document_id)
    except Exception:
        logger.exception("Failed to delete vectors for document_id=%s", document_id)

    if not deleted_file:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    return None
