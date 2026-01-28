"""Security utilities for production."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal and injection attacks.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for filesystem operations
    """
    if not filename:
        return "unnamed"
    
    # Remove path components
    filename = Path(filename).name
    
    # Remove dangerous characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Limit length
    filename = filename[:255]
    
    # Ensure it's not empty or just dots
    if not filename or filename.replace('.', '').replace('_', '') == '':
        filename = "unnamed"
    
    return filename


def validate_query_length(query: str, max_length: int) -> None:
    """
    Validate query length.
    
    Args:
        query: Query string
        max_length: Maximum allowed length
        
    Raises:
        ValueError: If query exceeds max length
    """
    if len(query) > max_length:
        raise ValueError(f"Query too long. Maximum {max_length} characters allowed.")


def validate_context_length(context: str, max_length: int) -> str:
    """
    Validate and truncate context if needed.
    
    Args:
        context: Context string
        max_length: Maximum allowed length
        
    Returns:
        Context (truncated if necessary)
    """
    if len(context) > max_length:
        logger.warning(f"Context truncated from {len(context)} to {max_length} characters")
        return context[:max_length]
    return context


def check_disk_space(path: Path, min_free_mb: int) -> bool:
    """
    Check if sufficient disk space is available.
    
    Args:
        path: Path to check
        min_free_mb: Minimum free space required in MB
        
    Returns:
        True if sufficient space available
    """
    try:
        stat = os.statvfs(path)
        free_mb = (stat.f_bavail * stat.f_frsize) / (1024 * 1024)
        return free_mb >= min_free_mb
    except (OSError, AttributeError):
        # Windows doesn't have statvfs, skip check
        logger.warning("Could not check disk space (Windows or permission issue)")
        return True


def sanitize_error_message(error: Exception, environment: str) -> str:
    """
    Sanitize error messages for production.
    
    Args:
        error: Exception object
        environment: Current environment
        
    Returns:
        Sanitized error message
    """
    error_msg = str(error)
    
    # In production, hide internal details
    if environment == "production":
        # Hide file paths
        error_msg = re.sub(r'/[^\s]+', '[path]', error_msg)
        # Hide tokens/keys
        error_msg = re.sub(r'(token|key|secret|password)\s*[:=]\s*[^\s]+', r'\1=[hidden]', error_msg, flags=re.IGNORECASE)
        # Genericize common errors
        if "HF_TOKEN" in error_msg:
            return "Configuration error. Please contact administrator."
        if "Redis" in error_msg or "redis" in error_msg:
            return "Service temporarily unavailable. Please try again."
        if "timeout" in error_msg.lower():
            return "Request timed out. Please try again."
    
    return error_msg[:200]  # Limit length
