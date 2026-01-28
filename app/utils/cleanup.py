"""Cleanup utilities for production maintenance."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

from app.config import get_settings

logger = logging.getLogger(__name__)


def cleanup_old_files(upload_dir: Path, max_age_days: int = 30) -> int:
    """
    Clean up old uploaded files.
    
    Args:
        upload_dir: Directory containing uploaded files
        max_age_days: Maximum age in days before deletion
        
    Returns:
        Number of files deleted
    """
    if not upload_dir.exists():
        return 0
    
    deleted_count = 0
    cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
    
    for file_path in upload_dir.glob("*"):
        if file_path.is_file():
            try:
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted old file: {file_path.name}")
            except Exception as e:
                logger.warning(f"Failed to delete {file_path}: {e}")
    
    if deleted_count > 0:
        logger.info(f"Cleanup completed: deleted {deleted_count} old files")
    
    return deleted_count


def cleanup_old_jobs(redis_client, max_age_days: int = 7) -> int:
    """
    Clean up old job statuses from Redis.
    
    Args:
        redis_client: Redis client instance
        max_age_days: Maximum age in days before deletion
        
    Returns:
        Number of jobs cleaned up
    """
    # Jobs already have TTL, but we can clean up manually if needed
    # This is handled by STATUS_TTL in ingest_queue.py
    return 0
