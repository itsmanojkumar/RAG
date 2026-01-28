"""Log query latency (embed, retrieve, LLM)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def log_query_latency(
    embed_s: float,
    retrieve_s: float,
    llm_s: float,
    total_s: float,
    extra: dict[str, Any] | None = None,
) -> None:
    """Log segmented and total query latency. Important for production."""
    d: dict[str, Any] = {
        "embed_s": round(embed_s, 4),
        "retrieve_s": round(retrieve_s, 4),
        "llm_s": round(llm_s, 4),
        "total_s": round(total_s, 4),
    }
    if extra:
        d.update(extra)
    logger.info("query_latency %s", d)
