# sentry/api/main.py
"""
FastAPI application entry point
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routes import health, query, sources, indexing, chat, stats
from .auth import get_or_create_api_key
from . import __version__

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events
    """
    # === Startup ===
    logger.info("🚀 Starting Sentry-AI API...")
    
    # Generate/retrieve API key
    api_key = get_or_create_api_key()
    logger.info("=" * 60)
    logger.info("🔑 API Key for authentication:")
    logger.info(f"   {api_key}")
    logger.info("=" * 60)
    logger.info("Use this key in the 'X-API-Key' header for protected endpoints")
    
    # Pre-load services (optional, for faster first request)
    try:
        from ..core.database import db
        db.get_source_stats()  # Verify database connection
        logger.info("✅ Database connected")
    except Exception as e:
        logger.warning(f"⚠️ Database check failed: {e}")
    
    yield
    
    # === Shutdown ===
    logger.info("👋 Shutting down Sentry-AI API...")
    
    # Save vector store
    try:
        from ..services.rag import get_rag_service
        rag = get_rag_service()
        rag.save_index()
        logger.info("✅ Vector store saved")
    except Exception as e:
        logger.warning(f"⚠️ Failed to save vector store: {e}")


# Create the FastAPI app
app = FastAPI(
    title="Sentry-AI API",
    description="""
    REST API for Sentry-AI - Local AI-powered IT diagnostics
    
    ## Features
    - **RAG Queries**: Ask questions about your logs
    - **Source Management**: Add/remove log sources
    - **Indexing**: Index log files and folders
    - **Chat History**: Track conversation history
    
    ## Authentication
    All endpoints except `/health` require an API key.
    Include the key in the `X-API-Key` header.
    
    The API key is auto-generated on first startup and displayed in the console.
    It's also stored in `~/.sentry-ai/.api_key`
    """,
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# === CORS Middleware ===
# Configure for Node.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",    # Default Next.js/Vite port
        "http://localhost:5173",    # Vite default
        "http://localhost:8080",    # Common dev port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# === Global Error Handler ===
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all error handler for unhandled exceptions
    """
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred",
            "error_type": type(exc).__name__
        }
    )


# === Mount Routers ===

# Health endpoints (no auth required)
app.include_router(health.router)

# API v1 routes (auth required)
API_PREFIX = "/api/v1"

app.include_router(query.router, prefix=API_PREFIX)
app.include_router(sources.router, prefix=API_PREFIX)
app.include_router(indexing.router, prefix=API_PREFIX)
app.include_router(chat.router, prefix=API_PREFIX)
app.include_router(stats.router, prefix=API_PREFIX)


# === Root Endpoint ===
@app.get("/", tags=["Root"])
async def root():
    """API root - basic info"""
    return {
        "name": "Sentry-AI API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health"
    }
