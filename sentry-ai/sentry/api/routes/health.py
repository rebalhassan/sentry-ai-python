# sentry/api/routes/health.py
"""
Health check endpoints
"""

from datetime import datetime
from fastapi import APIRouter

from ..schemas import HealthResponse, ReadinessResponse
from .. import __version__

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic health check - always returns healthy if server is running
    """
    return HealthResponse(
        status="healthy",
        version=__version__,
        timestamp=datetime.now()
    )


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_check():
    """
    Readiness probe - checks if all dependencies are working
    """
    ollama_ok = False
    db_ok = False
    message = None
    
    # Check Ollama connection
    try:
        from ...services.llm import get_llm_client
        llm = get_llm_client()
        # Try to list models to verify connection
        llm.list_models()
        ollama_ok = True
    except Exception as e:
        message = f"Ollama: {str(e)}"
    
    # Check database
    try:
        from ...core.database import db
        # Simple query to verify connection
        db.get_source_stats()
        db_ok = True
    except Exception as e:
        if message:
            message += f"; Database: {str(e)}"
        else:
            message = f"Database: {str(e)}"
    
    ready = ollama_ok and db_ok
    
    return ReadinessResponse(
        ready=ready,
        ollama_connected=ollama_ok,
        database_connected=db_ok,
        message=message if not ready else None
    )
