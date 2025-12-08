# sentry/api/routes/stats.py
"""
System statistics endpoints
"""

from fastapi import APIRouter, Depends, HTTPException

from ..schemas import StatsResponse, ModelsResponse, ModelInfo
from ..auth import verify_api_key
from ...services.rag import get_rag_service
from ...core.database import db

router = APIRouter(prefix="/stats", tags=["Stats"], dependencies=[Depends(verify_api_key)])


@router.get("", response_model=StatsResponse)
async def get_stats():
    """Get system statistics"""
    try:
        rag = get_rag_service()
        stats = rag.get_stats()
        db_stats = stats.get('database_stats', {})
        
        return StatsResponse(
            total_chunks=stats.get('total_chunks', 0),
            total_sources=db_stats.get('total_sources', 0),
            embedding_model=stats.get('embedding_model', 'unknown'),
            llm_model=stats.get('llm_model', 'unknown'),
            embedding_dimension=stats.get('embedding_dimension', 0),
            database_size_mb=db.get_db_size_mb(),
            vector_store_size=len(rag.vector_store)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/models", response_model=ModelsResponse)
async def list_models():
    """List available LLM models from Ollama"""
    try:
        from ...services.llm import get_llm_client
        llm = get_llm_client()
        model_names = llm.list_models()
        
        return ModelsResponse(
            models=[ModelInfo(name=name) for name in model_names]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")
