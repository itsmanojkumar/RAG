"""Extract text from PDF and TXT. No local ML models."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text(path: str | Path) -> str:
    """Extract text from a PDF or TXT file. Uses PyMuPDF for PDF."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".pdf":
        return _extract_pdf(path)
    if suf == ".txt":
        return _extract_txt(path)
    raise ValueError(f"Unsupported format: {suf}. Use .pdf or .txt.")


def _extract_pdf(path: Path) -> str:
    import fitz

    parts: list[str] = []
    with fitz.open(path) as doc:
        for page in doc:
            t = page.get_text()
            if t:
                parts.append(t)
    raw = "\n".join(parts).strip()
    return _normalize(raw)


def _extract_txt(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    return _normalize(raw)


def _normalize(text: str) -> str:
    """Normalize whitespace and strip."""
    return " ".join(text.split())
