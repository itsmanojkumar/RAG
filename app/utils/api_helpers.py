"""Helper functions for API response parsing and error handling."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def extract_text_from_hf_response(result: Any) -> str:
    """
    Extract generated text from HuggingFace API response.
    
    Handles multiple response formats:
    - List of dicts: [{"generated_text": "..."}]
    - Single dict: {"generated_text": "..."}
    - String: "text"
    - Other formats
    
    Args:
        result: The JSON response from HuggingFace API
        
    Returns:
        Extracted text string, or empty string if not found
    """
    if isinstance(result, list):
        if result and isinstance(result[0], dict):
            return result[0].get("generated_text", result[0].get("text", ""))
        elif result:
            return str(result[0])
    elif isinstance(result, dict):
        # Try multiple possible keys
        text = (
            result.get("generated_text") or 
            result.get("text") or 
            result.get("output") or
            result.get("content") or
            ""
        )
        # If still empty, try to get first string value
        if not text:
            for v in result.values():
                if isinstance(v, str) and v.strip():
                    text = v
                    break
        return text
    elif isinstance(result, str):
        return result
    
    return str(result) if result else ""


def is_model_loading_error(response_data: Any) -> bool:
    """
    Check if the error indicates the model is loading.
    
    Args:
        response_data: The error response data
        
    Returns:
        True if model is loading, False otherwise
    """
    if not response_data:
        return False
    
    error_str = str(response_data).lower()
    loading_indicators = ["loading", "model", "warmup", "starting", "initializing"]
    return any(indicator in error_str for indicator in loading_indicators)


def should_retry_error(status_code: int, error_data: Any = None) -> tuple[bool, float]:
    """
    Determine if an error should be retried and how long to wait.
    
    Args:
        status_code: HTTP status code
        error_data: Optional error response data
        
    Returns:
        Tuple of (should_retry, wait_seconds)
    """
    # Model loading - wait longer
    if status_code == 503 and error_data and is_model_loading_error(error_data):
        return True, 10.0
    
    # Rate limiting - exponential backoff
    if status_code == 429:
        return True, 5.0
    
    # Server errors - retry with backoff
    if status_code >= 500:
        return True, 2.0
    
    # Client errors - don't retry
    if 400 <= status_code < 500:
        return False, 0.0
    
    return False, 0.0
