# sentry/api/schemas.py
"""
Request/Response schemas for the API
These extend the core models with API-specific fields
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from ..core.models import LogLevel, SourceType


# ===== Query Endpoints =====

class QueryRequest(BaseModel):
    """Request body for RAG query"""
    query: str = Field(..., description="The question to ask about the logs")
    top_k: Optional[int] = Field(None, ge=1, le=50, description="Number of results to retrieve")
    use_reranking: Optional[bool] = Field(None, description="Use BM25 reranking")
    similarity_threshold: Optional[float] = Field(None, ge=0, le=1, description="Minimum similarity score")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What disk errors occurred today?",
                "top_k": 10
            }
        }


class SourceInfo(BaseModel):
    """Simplified source info for responses"""
    id: str
    content: str
    timestamp: datetime
    log_level: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    """Response for RAG query"""
    answer: str
    sources: List[SourceInfo]
    confidence: float
    query_time: float
    chunk_ids: List[str]


class SearchRequest(BaseModel):
    """Request for similarity search (no LLM)"""
    text: str = Field(..., description="Text to find similar logs for")
    top_k: Optional[int] = Field(5, ge=1, le=50, description="Number of results")


class SearchResult(BaseModel):
    """Single search result"""
    chunk: SourceInfo
    score: float


class SearchResponse(BaseModel):
    """Response for similarity search"""
    results: List[SearchResult]
    search_time: float


# ===== Source Management =====

class SourceCreateRequest(BaseModel):
    """Request to create a new log source"""
    name: str = Field(..., description="Human-readable name for the source")
    source_type: SourceType = Field(..., description="Type of source: file, folder, or eventviewer")
    path: Optional[str] = Field(None, description="File or folder path (for file/folder types)")
    eventlog_name: Optional[str] = Field(None, description="Event log name (for eventviewer type)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "System Logs",
                "source_type": "file",
                "path": "C:\\Logs\\system.log"
            }
        }


class SourceResponse(BaseModel):
    """Response for source operations"""
    id: str
    name: str
    source_type: SourceType
    path: Optional[str]
    eventlog_name: Optional[str]
    is_active: bool
    created_at: datetime
    last_indexed: Optional[datetime]
    total_chunks: int


class SourceListResponse(BaseModel):
    """Response for listing sources"""
    sources: List[SourceResponse]
    total: int


class SourceToggleRequest(BaseModel):
    """Request to toggle source active state"""
    is_active: bool


# ===== Indexing =====

class IndexFileRequest(BaseModel):
    """Request to index a log file"""
    source_id: str = Field(..., description="ID of the source this file belongs to")
    file_path: str = Field(..., description="Path to the log file")


class IndexFolderRequest(BaseModel):
    """Request to index a folder of logs"""
    source_id: str = Field(..., description="ID of the source")
    folder_path: str = Field(..., description="Path to the folder")


class IndexChunksRequest(BaseModel):
    """Request to index pre-parsed chunks"""
    source_id: str
    chunks: List[Dict[str, Any]] = Field(..., description="List of chunk data")


class IndexResponse(BaseModel):
    """Response for indexing operations"""
    success: bool
    chunks_indexed: int
    message: str


# ===== Chat History =====

class ChatMessageResponse(BaseModel):
    """Single chat message"""
    role: str
    content: str
    timestamp: datetime
    sources: List[str] = Field(default_factory=list)


class ChatHistoryResponse(BaseModel):
    """Response for chat history"""
    messages: List[ChatMessageResponse]
    total: int


# ===== Stats =====

class StatsResponse(BaseModel):
    """System statistics"""
    total_chunks: int
    total_sources: int
    embedding_model: str
    llm_model: str
    embedding_dimension: int
    database_size_mb: float
    vector_store_size: int


class ModelInfo(BaseModel):
    """LLM model information"""
    name: str
    size: Optional[str] = None
    modified_at: Optional[str] = None


class ModelsResponse(BaseModel):
    """Response for listing models"""
    models: List[ModelInfo]


# ===== Health =====

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.now)


class ReadinessResponse(BaseModel):
    """Readiness probe response"""
    ready: bool
    ollama_connected: bool
    database_connected: bool
    message: Optional[str] = None


# ===== Error =====

class ErrorResponse(BaseModel):
    """Standard error response"""
    detail: str
    error_code: Optional[str] = None
