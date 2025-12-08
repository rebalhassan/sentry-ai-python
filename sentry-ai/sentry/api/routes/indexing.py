# sentry/api/routes/indexing.py
"""
Log indexing endpoints
"""

from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from datetime import datetime

from ..schemas import (
    IndexFileRequest, IndexFolderRequest, 
    IndexChunksRequest, IndexResponse
)
from ..auth import verify_api_key
from ...services.rag import get_rag_service
from ...services.indexer import LogIndexer
from ...core.database import db
from ...core.models import LogChunk, LogLevel

router = APIRouter(prefix="/index", tags=["Indexing"], dependencies=[Depends(verify_api_key)])


@router.post("/file", response_model=IndexResponse)
async def index_file(request: IndexFileRequest):
    """
    Index a log file
    
    Parses the file into chunks and indexes them for RAG queries.
    """
    # Verify source exists
    source = db.get_source(request.source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    
    # Verify file exists
    file_path = Path(request.file_path)
    if not file_path.exists():
        raise HTTPException(status_code=400, detail=f"File not found: {request.file_path}")
    
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail=f"Path is not a file: {request.file_path}")
    
    try:
        # Parse the file
        indexer = LogIndexer()
        chunks = indexer.parse_file(file_path, request.source_id)
        
        if not chunks:
            return IndexResponse(
                success=True,
                chunks_indexed=0,
                message="File parsed but no chunks were created"
            )
        
        # Index the chunks
        rag = get_rag_service()
        indexed = rag.index_chunks_batch(chunks)
        
        # Update source stats
        db.update_source_stats(
            source_id=request.source_id,
            total_chunks=db.get_chunk_count(request.source_id),
            last_indexed=datetime.now()
        )
        
        return IndexResponse(
            success=True,
            chunks_indexed=indexed,
            message=f"Successfully indexed {indexed} chunks from {file_path.name}"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@router.post("/folder", response_model=IndexResponse)
async def index_folder(request: IndexFolderRequest):
    """
    Index all log files in a folder
    
    Recursively parses all supported log files (.log, .txt, .csv)
    """
    # Verify source exists
    source = db.get_source(request.source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    
    # Verify folder exists
    folder_path = Path(request.folder_path)
    if not folder_path.exists():
        raise HTTPException(status_code=400, detail=f"Folder not found: {request.folder_path}")
    
    if not folder_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a folder: {request.folder_path}")
    
    try:
        # Parse the folder
        indexer = LogIndexer()
        chunks = indexer.parse_folder(folder_path, request.source_id)
        
        if not chunks:
            return IndexResponse(
                success=True,
                chunks_indexed=0,
                message="Folder parsed but no chunks were created"
            )
        
        # Index the chunks
        rag = get_rag_service()
        indexed = rag.index_chunks_batch(chunks)
        
        # Update source stats
        db.update_source_stats(
            source_id=request.source_id,
            total_chunks=db.get_chunk_count(request.source_id),
            last_indexed=datetime.now()
        )
        
        return IndexResponse(
            success=True,
            chunks_indexed=indexed,
            message=f"Successfully indexed {indexed} chunks from folder"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@router.post("/chunks", response_model=IndexResponse)
async def index_chunks(request: IndexChunksRequest):
    """
    Index pre-parsed chunks directly
    
    Use this when you've already parsed the logs on the frontend.
    """
    # Verify source exists
    source = db.get_source(request.source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")
    
    if not request.chunks:
        return IndexResponse(
            success=True,
            chunks_indexed=0,
            message="No chunks provided"
        )
    
    try:
        # Convert dict chunks to LogChunk objects
        log_chunks = []
        for chunk_data in request.chunks:
            chunk = LogChunk(
                source_id=request.source_id,
                content=chunk_data.get("content", ""),
                timestamp=chunk_data.get("timestamp", datetime.now()),
                log_level=LogLevel(chunk_data.get("log_level", "unknown")),
                metadata=chunk_data.get("metadata", {})
            )
            log_chunks.append(chunk)
        
        # Index the chunks
        rag = get_rag_service()
        indexed = rag.index_chunks_batch(log_chunks)
        
        # Update source stats
        db.update_source_stats(
            source_id=request.source_id,
            total_chunks=db.get_chunk_count(request.source_id),
            last_indexed=datetime.now()
        )
        
        return IndexResponse(
            success=True,
            chunks_indexed=indexed,
            message=f"Successfully indexed {indexed} chunks"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")
