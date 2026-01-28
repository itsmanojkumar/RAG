"""Run the ARQ worker for document ingestion."""

import asyncio
import logging
import sys
import signal

from arq import run_worker

from app.config import get_settings
from app.jobs.worker import WorkerSettings

settings = get_settings()
log_format = (
    '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
    if settings.LOG_JSON
    else "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format=log_format,
    stream=sys.stdout,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down worker gracefully...")
    sys.exit(0)


def main() -> None:
    """Main entry point for the worker."""
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info(f"Starting ARQ worker in {settings.ENVIRONMENT} mode")
    
    try:
        # Test Redis connection before starting
        import redis
        r = redis.from_url(settings.REDIS_URL, socket_connect_timeout=5)
        r.ping()
        logger.info("Redis connection successful, starting worker...")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        if settings.ENVIRONMENT == "production":
            sys.exit(1)
    
    try:
        run_worker(WorkerSettings)
    except KeyboardInterrupt:
        logger.info("Worker stopped by user")
    except Exception as e:
        logger.exception(f"Worker error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
