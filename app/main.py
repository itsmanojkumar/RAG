from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.api.documents import router as documents_router
from app.api.query import router as query_router
from app.config import get_settings

# Configure logging based on environment
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

# Frontend dist (built from frontend/). Default: project root / frontend / dist.
FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager with startup and shutdown logic."""
    s = get_settings()
    
    # Startup
    logger.info(f"Starting RAG application in {s.ENVIRONMENT} mode")
    
    try:
        # Create necessary directories
        s.upload_dir_path().mkdir(parents=True, exist_ok=True)
        if s.VECTOR_STORE == "faiss":
            s.faiss_index_path().parent.mkdir(parents=True, exist_ok=True)
        
        # Validate critical settings
        if not s.HF_TOKEN:
            logger.warning("HF_TOKEN not set - LLM and reranking features will not work")
        
        # Test Redis connection (non-blocking, short timeout)
        try:
            import redis
            r = redis.from_url(s.REDIS_URL, socket_connect_timeout=1, socket_timeout=1)
            r.ping()
            logger.info("Redis connection successful")
            app.state.redis_available = True
        except Exception as e:
            logger.warning(f"Redis connection failed: {e} - background tasks will be unavailable")
            app.state.redis_available = False
            if s.ENVIRONMENT == "production":
                raise
        
        # Production cleanup on startup
        if s.ENVIRONMENT == "production":
            try:
                from app.utils.cleanup import cleanup_old_files
                deleted = cleanup_old_files(s.upload_dir_path(), max_age_days=30)
                if deleted > 0:
                    logger.info(f"Cleaned up {deleted} old files on startup")
            except Exception as e:
                logger.warning(f"Cleanup failed: {e}")
        
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        if s.ENVIRONMENT == "production":
            raise
    
    yield
    
    # Shutdown
    logger.info("Application shutdown initiated")


def create_app() -> FastAPI:
    from app.limiter import limiter

    s = get_settings()
    
    app = FastAPI(
        title="RAG QA API",
        description="Upload documents (PDF/TXT), ask questions. Uses HF Inference API + FAISS/Pinecone.",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if s.ENVIRONMENT == "development" else None,
        redoc_url="/redoc" if s.ENVIRONMENT == "development" else None,
    )
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_origins_list if s.cors_origins_list else ["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        max_age=3600,
    )
    
    # Request timeout middleware
    @app.middleware("http")
    async def timeout_middleware(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Add timeout header
        response.headers["X-Process-Time"] = str(process_time)
        
        # Check timeout
        if process_time > s.REQUEST_TIMEOUT:
            logger.warning(f"Request exceeded timeout: {process_time:.2f}s > {s.REQUEST_TIMEOUT}s")
        
        return response

    app.include_router(documents_router)
    app.include_router(query_router)

    @app.get("/health")
    async def health():
        """Health check endpoint with dependency status."""
        health_status = {
            "status": "ok",
            "environment": s.ENVIRONMENT,
        }
        
        # Check Redis (non-blocking check using cached state)
        redis_available = getattr(app.state, 'redis_available', False)
        if redis_available:
            health_status["redis"] = "ok"
        else:
            health_status["redis"] = "unavailable"
            # Don't return 503 in development without Redis
            if s.ENVIRONMENT == "production":
                return JSONResponse(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    content=health_status
                )
        
        # Check vector store
        try:
            if s.VECTOR_STORE == "faiss":
                idx_path = s.faiss_index_path()
                if idx_path.exists():
                    health_status["vector_store"] = "ok"
                else:
                    health_status["vector_store"] = "no_index"
            else:
                health_status["vector_store"] = "pinecone"
        except Exception as e:
            health_status["vector_store"] = f"error: {str(e)}"
        
        # Check HF token (don't expose in production)
        if s.ENVIRONMENT == "production":
            health_status["hf_token"] = "configured" if s.HF_TOKEN else "not_configured"
        else:
            health_status["hf_token"] = "set" if s.HF_TOKEN else "not_set"
        
        status_code = status.HTTP_200_OK if health_status["redis"] == "ok" else status.HTTP_503_SERVICE_UNAVAILABLE
        return JSONResponse(status_code=status_code, content=health_status)

    # Serve frontend when built (production).
    if (FRONTEND_DIST / "index.html").exists():
        assets = FRONTEND_DIST / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=str(assets)), name="assets")
        favicon = FRONTEND_DIST / "favicon.svg"
        if favicon.exists():

            @app.get("/favicon.svg")
            def favicon_route():
                return FileResponse(favicon, media_type="image/svg+xml")

        @app.get("/")
        def index():
            return FileResponse(FRONTEND_DIST / "index.html")

        logger.info("Serving frontend from %s", FRONTEND_DIST)

    return app


app = create_app()
