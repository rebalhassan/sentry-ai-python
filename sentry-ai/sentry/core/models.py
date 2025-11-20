# sentry/core/models.py
"""
Core data models for Sentry-AI
These are the building blocks that flow through the entire system
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import uuid


class SourceType(str, Enum):
    """Types of log sources we can monitor"""
    FILE = "file"
    FOLDER = "folder"
    EVENTVIEWER = "eventviewer"


class LogLevel(str, Enum):
    """Standard log severity levels"""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    DEBUG = "debug"
    UNKNOWN = "unknown"


class LogSource(BaseModel):
    """
    Represents a source of logs (file, folder, or EventViewer)
    This is what users "add" to the system
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    source_type: SourceType
    path: Optional[str] = None  # For FILE/FOLDER types
    eventlog_name: Optional[str] = None  # For EVENTVIEWER (e.g., "System", "Application")
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    last_indexed: Optional[datetime] = None
    total_chunks: int = 0
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "System Logs",
                "source_type": "eventviewer",
                "eventlog_name": "System",
                "is_active": True
            }
        }


class LogChunk(BaseModel):
    """
    A single chunk of log data that gets embedded and stored
    This is the atomic unit of our RAG system
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str  # References LogSource.id
    content: str  # The actual log text
    timestamp: datetime
    log_level: LogLevel = LogLevel.UNKNOWN
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Extra info (file path, event ID, etc.)
    embedding_id: Optional[int] = None  # Index in FAISS vector store
    
    class Config:
        json_schema_extra = {
            "example": {
                "source_id": "abc-123",
                "content": "ERROR: Disk write failure on D:\\data",
                "timestamp": "2025-11-02T21:30:45",
                "log_level": "error",
                "metadata": {"file": "system.log", "line": 4521}
            }
        }


class ChatMessage(BaseModel):
    """
    A single message in the chat interface
    Can be from user or assistant
    """
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    sources: List[str] = Field(default_factory=list)  # Chunk IDs that were used to generate this
    
    def is_user(self) -> bool:
        return self.role == "user"
    
    def is_assistant(self) -> bool:
        return self.role == "assistant"


class QueryResult(BaseModel):
    """
    The result of a RAG query
    This is what the chat interface displays
    """
    answer: str  # The LLM's natural language response
    sources: List[LogChunk]  # The log chunks that were used as context
    confidence: float  # 0-1 score (based on similarity scores)
    query_time: float  # How long it took (in seconds)
    chunk_ids: List[str] = Field(default_factory=list)  # IDs of chunks used
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Found 3 critical disk errors starting at 9:01 PM...",
                "confidence": 0.92,
                "query_time": 1.23,
                "sources": []
            }
        }


class IndexingStatus(BaseModel):
    """
    Real-time status of indexing operations
    Used to show progress in UI
    """
    source_id: str
    source_name: str
    status: str  # "indexing", "complete", "error"
    progress: float = 0.0  # 0-100
    chunks_processed: int = 0
    total_chunks: int = 0
    error_message: Optional[str] = None
    started_at: datetime = Field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


# Type aliases for clarity
ChunkID = str
SourceID = str
EmbeddingVector = List[float]