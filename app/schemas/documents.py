from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    job_id: str = Field(..., description="ARQ job ID for ingestion")
    document_id: str = Field(..., description="Unique document identifier")
    message: str = Field(default="Document uploaded; ingestion enqueued.", description="Status message")


class JobStatusResponse(BaseModel):
    job_id: str = Field(..., description="ARQ job ID")
    status: Literal["pending", "processing", "completed", "failed"] = Field(..., description="Job status")
    message: str | None = Field(default=None, description="Optional message, e.g. error if failed")


class DocumentInfo(BaseModel):
    document_id: str = Field(..., description="Document ID")
    filename: str = Field(..., description="Original or stored filename")
    size_bytes: int = Field(..., description="File size in bytes")
    uploaded_at: str = Field(..., description="Upload timestamp (ISO format)")


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo] = Field(default_factory=list, description="List of uploaded documents")
