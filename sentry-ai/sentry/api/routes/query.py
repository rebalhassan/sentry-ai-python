# sentry/api/routes/query.py
"""
RAG Query endpoints - the main AI functionality
"""

import time
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
import json

from ..schemas import (
    QueryRequest, QueryResponse, SourceInfo,
    SearchRequest, SearchResponse, SearchResult
)
from ..auth import verify_api_key
from ...services.rag import get_rag_service
from ...core.database import db
from ...core.models import ChatMessage

router = APIRouter(prefix="/query", tags=["Query"], dependencies=[Depends(verify_api_key)])


@router.post("", response_model=QueryResponse)
async def rag_query(request: QueryRequest):
    """
    Run a RAG query against the indexed logs
    
    This is the main endpoint - it:
    1. Embeds your question
    2. Finds similar log entries
    3. Uses LLM to generate an answer
    4. Returns the answer with source citations
    """
    try:
        rag = get_rag_service()
        
        result = rag.query(
            query_text=request.query,
            top_k=request.top_k,
            use_reranking=request.use_reranking,
            similarity_threshold=request.similarity_threshold
        )
        
        # Save to chat history
        user_msg = ChatMessage(role="user", content=request.query)
        assistant_msg = ChatMessage(
            role="assistant", 
            content=result.answer,
            sources=result.chunk_ids
        )
        db.add_chat_message(user_msg)
        db.add_chat_message(assistant_msg)
        
        # Convert sources to response format
        sources = [
            SourceInfo(
                id=chunk.id,
                content=chunk.content,
                timestamp=chunk.timestamp,
                log_level=chunk.log_level.value,
                metadata=chunk.metadata
            )
            for chunk in result.sources
        ]
        
        return QueryResponse(
            answer=result.answer,
            sources=sources,
            confidence=result.confidence,
            query_time=result.query_time,
            chunk_ids=result.chunk_ids
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/stream")
async def rag_query_stream(request: QueryRequest):
    """
    Streaming RAG query - returns answer token by token via SSE
    
    Use this for real-time UI updates while the LLM generates
    """
    try:
        rag = get_rag_service()
        
        # First, do the retrieval part
        query_vector = rag.embedder.embed_text(request.query)
        chunk_ids, scores = rag.vector_store.search(
            query_vector, 
            k=request.top_k or 10
        )
        chunks = db.get_chunks_by_ids(chunk_ids)
        
        async def generate():
            # Stream metadata first
            yield f"data: {json.dumps({'type': 'metadata', 'sources': len(chunks)})}\n\n"
            
            # Stream the LLM response
            for token in rag.llm.stream_generate(
                prompt=request.query,
                system_prompt=None  # Will use default
            ):
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stream query failed: {str(e)}")


@router.post("/search", response_model=SearchResponse)
async def similarity_search(request: SearchRequest):
    """
    Find similar log entries without using the LLM
    
    Faster than full RAG query - just returns matching logs with scores
    """
    start_time = time.time()
    
    try:
        rag = get_rag_service()
        
        similar = rag.search_similar(
            text=request.text,
            top_k=request.top_k or 5
        )
        
        results = [
            SearchResult(
                chunk=SourceInfo(
                    id=chunk.id,
                    content=chunk.content,
                    timestamp=chunk.timestamp,
                    log_level=chunk.log_level.value,
                    metadata=chunk.metadata
                ),
                score=score
            )
            for chunk, score in similar
        ]
        
        return SearchResponse(
            results=results,
            search_time=time.time() - start_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
