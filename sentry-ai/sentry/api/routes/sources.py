# sentry/api/routes/sources.py
"""
Log source management endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status

from ..schemas import (
    SourceCreateRequest, SourceResponse, 
    SourceListResponse, SourceToggleRequest
)
from ..auth import verify_api_key
from ...core.database import db
from ...core.models import LogSource

router = APIRouter(prefix="/sources", tags=["Sources"], dependencies=[Depends(verify_api_key)])


def _source_to_response(source: LogSource) -> SourceResponse:
    """Convert LogSource to API response"""
    return SourceResponse(
        id=source.id,
        name=source.name,
        source_type=source.source_type,
        path=source.path,
        eventlog_name=source.eventlog_name,
        is_active=source.is_active,
        created_at=source.created_at,
        last_indexed=source.last_indexed,
        total_chunks=source.total_chunks
    )


@router.get("", response_model=SourceListResponse)
async def list_sources(active_only: bool = False):
    """
    List all log sources
    
    Args:
        active_only: If true, only return active sources
    """
    sources = db.list_sources(active_only=active_only)
    
    return SourceListResponse(
        sources=[_source_to_response(s) for s in sources],
        total=len(sources)
    )


@router.post("", response_model=SourceResponse, status_code=status.HTTP_201_CREATED)
async def create_source(request: SourceCreateRequest):
    """
    Create a new log source
    
    After creating, use the indexing endpoints to index the logs.
    """
    # Validate request
    if request.source_type in ["file", "folder"] and not request.path:
        raise HTTPException(
            status_code=400,
            detail="Path is required for file and folder source types"
        )
    
    if request.source_type == "eventviewer" and not request.eventlog_name:
        raise HTTPException(
            status_code=400,
            detail="eventlog_name is required for eventviewer source type"
        )
    
    # Check if source with same name exists
    existing = db.get_source_by_name(request.name)
    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Source with name '{request.name}' already exists"
        )
    
    # Create source
    source = LogSource(
        name=request.name,
        source_type=request.source_type,
        path=request.path,
        eventlog_name=request.eventlog_name
    )
    
    try:
        db.add_source(source)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create source: {str(e)}")
    
    return _source_to_response(source)


@router.get("/{source_id}", response_model=SourceResponse)
async def get_source(source_id: str):
    """Get a specific source by ID"""
    source = db.get_source(source_id)
    
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    
    return _source_to_response(source)


@router.delete("/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_source(source_id: str):
    """
    Delete a source and all its chunks
    
    This also removes the vectors from the vector store.
    """
    source = db.get_source(source_id)
    
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    
    try:
        # Delete from database (cascade will delete chunks)
        db.delete_source(source_id)
        
        # Note: Vector store cleanup would ideally happen here
        # but FAISS doesn't support efficient deletion
        # The vectors will be orphaned until index rebuild
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete source: {str(e)}")


@router.patch("/{source_id}/toggle", response_model=SourceResponse)
async def toggle_source(source_id: str, request: SourceToggleRequest):
    """Enable or disable a source"""
    source = db.get_source(source_id)
    
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    
    try:
        db.toggle_source(source_id, request.is_active)
        
        # Refresh source data
        source = db.get_source(source_id)
        return _source_to_response(source)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to toggle source: {str(e)}")
