from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="User question")


class SourceChunk(BaseModel):
    text: str = Field(..., description="Chunk text")
    source: str | None = Field(default=None, description="Source file or document id")
    score: float | None = Field(default=None, description="Similarity score if available")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    sources: list[SourceChunk] = Field(default_factory=list, description="Retrieved source chunks")
