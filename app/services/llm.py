"""Generate RAG answer using HuggingFace Inference Providers API.

Documentation: https://huggingface.co/docs/api-inference/en/index
Uses OpenAI-compatible chat completions endpoint via router.huggingface.co
Supports both streaming and non-streaming responses.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import AsyncGenerator

import httpx

from app.config import get_settings
from app.utils.security import validate_context_length

logger = logging.getLogger(__name__)

# Default model - Groq provides FREE fast inference
DEFAULT_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
DEFAULT_PROVIDER = "groq"


def _build_messages(context: str, question: str) -> list[dict]:
    """Build chat messages for RAG prompt (OpenAI-compatible format)."""
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant that answers questions based on the provided context. "
                   "If the context does not contain relevant information to answer the question, say so clearly."
    }
    
    user_message = {
        "role": "user",
        "content": f"""Context:
{context}

Question: {question}

Please answer the question based only on the context provided above."""
    }
    
    return [system_message, user_message]


def _get_model_and_provider() -> tuple[str, str]:
    """Get model name and provider from settings."""
    settings = get_settings()
    model_name = DEFAULT_MODEL
    provider = DEFAULT_PROVIDER
    
    if settings.HF_INFERENCE_URL:
        url_str = settings.HF_INFERENCE_URL.strip()
        
        if "/models/" in url_str:
            model_name = url_str.split("/models/")[-1].split(":")[0]
        elif ":" in url_str and not url_str.startswith("http"):
            last_colon = url_str.rfind(":")
            model_name = url_str[:last_colon]
            provider = url_str[last_colon + 1:]
        else:
            model_name = url_str
    
    return model_name, provider


async def generate_answer_stream(
    question: str, 
    chunks: list[dict]
) -> AsyncGenerator[str, None]:
    """
    Stream RAG answer generation token by token.
    Yields chunks of text as they are generated.
    """
    settings = get_settings()
    if not settings.HF_TOKEN:
        raise ValueError("HF_TOKEN is required for text generation.")

    context = "\n\n".join(c["text"] for c in chunks)
    context = validate_context_length(context, settings.MAX_CONTEXT_LENGTH)
    
    logger.info(f"Streaming answer from {len(chunks)} chunks (context: {len(context)} chars)")
    
    model_name, provider = _get_model_and_provider()
    model_with_provider = f"{model_name}:{provider}"
    
    url = "https://router.huggingface.co/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {settings.HF_TOKEN}",
        "Content-Type": "application/json",
    }
    
    messages = _build_messages(context, question)
    
    payload = {
        "model": model_with_provider,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.1,
        "stream": True,  # Enable streaming
    }
    
    logger.info(f"Streaming with model: {model_with_provider}")
    
    async with httpx.AsyncClient(timeout=settings.HTTP_LLM_TIMEOUT) as client:
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            if response.status_code != 200:
                error_text = ""
                async for chunk in response.aiter_text():
                    error_text += chunk
                logger.error(f"Stream API error {response.status_code}: {error_text[:500]}")
                raise Exception(f"API error: {response.status_code} - {error_text[:200]}")
            
            async for line in response.aiter_lines():
                if not line:
                    continue
                    
                # SSE format: data: {...}
                if line.startswith("data: "):
                    data_str = line[6:]  # Remove "data: " prefix
                    
                    if data_str.strip() == "[DONE]":
                        break
                    
                    try:
                        data = json.loads(data_str)
                        
                        # Extract content from delta
                        if "choices" in data and data["choices"]:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                                
                    except json.JSONDecodeError:
                        continue


async def generate_answer(question: str, chunks: list[dict]) -> tuple[str, float]:
    """
    RAG Answer Generation: Generate answer from reranked chunks using LLM.
    Uses HuggingFace Inference Providers with OpenAI-compatible chat completions API.
    Non-streaming version for backward compatibility.
    """
    settings = get_settings()
    if not settings.HF_TOKEN:
        raise ValueError("HF_TOKEN is required for text generation.")

    # Build context from reranked chunks
    context = "\n\n".join(c["text"] for c in chunks)
    context = validate_context_length(context, settings.MAX_CONTEXT_LENGTH)
    
    logger.info(f"Generating answer from {len(chunks)} chunks (context: {len(context)} chars)")
    
    model_name, provider = _get_model_and_provider()
    model_with_provider = f"{model_name}:{provider}"
    
    # Use OpenAI-compatible chat completions endpoint
    url = "https://router.huggingface.co/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {settings.HF_TOKEN}",
        "Content-Type": "application/json",
    }
    
    messages = _build_messages(context, question)
    
    payload = {
        "model": model_with_provider,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.1,
        "stream": False,
    }
    
    logger.info(f"Using model: {model_with_provider}")
    
    t0 = time.perf_counter()
    max_retries = 5
    
    async with httpx.AsyncClient(timeout=settings.HTTP_LLM_TIMEOUT) as client:
        for attempt in range(max_retries):
            try:
                response = await client.post(url, json=payload, headers=headers)
                
                # Handle model loading (503)
                if response.status_code == 503:
                    try:
                        error_data = response.json()
                        error_str = str(error_data).lower()
                        if "loading" in error_str or "starting" in error_str or "warming" in error_str:
                            estimated_time = error_data.get("estimated_time", 20)
                            wait = min(estimated_time, 30)
                            logger.info(f"Model loading, waiting {wait}s (attempt {attempt + 1}/{max_retries})...")
                            await asyncio.sleep(wait)
                            continue
                    except Exception:
                        pass
                    if attempt < max_retries - 1:
                        wait = 5 * (attempt + 1)
                        logger.info(f"503 error, waiting {wait}s...")
                        await asyncio.sleep(wait)
                        continue
                
                # Handle rate limiting (429)
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait = 5 * (attempt + 1)
                        logger.warning(f"Rate limited, waiting {wait}s...")
                        await asyncio.sleep(wait)
                        continue
                
                # Handle other errors
                if response.status_code != 200:
                    error_text = response.text[:500]
                    logger.error(f"API error {response.status_code}: {error_text}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise Exception(f"API error: {response.status_code} - {error_text}")
                
                # Parse OpenAI-compatible response
                result = response.json()
                
                # Extract answer from chat completion response
                answer = ""
                if "choices" in result and result["choices"]:
                    choice = result["choices"][0]
                    if "message" in choice:
                        answer = choice["message"].get("content", "")
                    elif "text" in choice:
                        answer = choice["text"]
                elif "generated_text" in result:
                    # Fallback for non-chat format
                    answer = result["generated_text"]
                
                answer = answer.strip()
                
                if not answer:
                    if attempt < max_retries - 1:
                        logger.warning("Empty response, retrying...")
                        await asyncio.sleep(2)
                        continue
                    answer = "I could not generate an answer based on the provided context."
                
                elapsed = time.perf_counter() - t0
                logger.info(f"Generated answer ({len(answer)} chars) in {elapsed:.2f}s")
                return (answer, elapsed)
                
            except httpx.TimeoutException:
                logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries}")
                if attempt >= max_retries - 1:
                    raise Exception("Request timed out after all retries")
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"Error on attempt {attempt + 1}/{max_retries}: {e}")
                if attempt >= max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
    
    elapsed = time.perf_counter() - t0
    return ("Failed to generate answer.", elapsed)
