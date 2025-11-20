# sentry/core/database.py
"""
Database layer for Sentry-AI
SQLite for metadata + relationships between sources, chunks, and chat history
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

from .models import LogSource, LogChunk, ChatMessage, SourceType, LogLevel
from .config import settings


class Database:
    """
    Handles all SQLite operations
    
    Tables:
    - sources: All log sources (files, folders, EventViewer)
    - chunks: Individual log chunks with embeddings
    - chat_history: Conversation history
    """
    
    # sentry/core/database.py

    def __init__(self, db_path = None):
        """
        Initialize database
        
        Args:
            db_path: Path to database file (or ":memory:" for in-memory DB)
        """
        # Handle both Path objects and strings (for ":memory:")
        if db_path is None:
            self.db_path = settings.db_path
        elif isinstance(db_path, str):
            self.db_path = db_path  # Keep as string for ":memory:"
        else:
            self.db_path = db_path
        
        # ===== THE FIX =====
        # For in-memory databases, keep a persistent connection
        # Otherwise each get_connection() creates a NEW empty database
        self._memory_conn = None
        if self.db_path == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:")
            self._memory_conn.row_factory = sqlite3.Row
            self._memory_conn.execute("PRAGMA foreign_keys = ON")
        
        # Now initialize tables (will use persistent connection if in-memory)
        self._init_db()
    
    def _init_db(self):
        """
        Initialize database schema
        Creates tables if they don't exist
        """
        with self.get_connection() as conn:
            # ===== SOURCES TABLE =====
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    path TEXT,
                    eventlog_name TEXT,
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    last_indexed TEXT,
                    total_chunks INTEGER DEFAULT 0,
                    UNIQUE(name)
                )
            """)
            
            # ===== CHUNKS TABLE =====
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    log_level TEXT DEFAULT 'unknown',
                    metadata TEXT,
                    embedding_id INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE
                )
            """)
            
            # ===== CHAT HISTORY TABLE =====
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    sources TEXT,
                    CHECK(role IN ('user', 'assistant'))
                )
            """)
            
            # ===== INDEXES FOR PERFORMANCE =====
            # These make queries FAST
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_timestamp ON chunks(timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_level ON chunks(log_level)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks(embedding_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_timestamp ON chat_history(timestamp DESC)")
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections
        
        For in-memory DBs: Returns the persistent connection (doesn't close it)
        For file DBs: Creates a new connection each time (and closes it)
        
        Usage:
            with db.get_connection() as conn:
                conn.execute(...)
        """
        if self._memory_conn:
            # ===== IN-MEMORY DATABASE =====
            # Return the persistent connection (DON'T close it)
            yield self._memory_conn
        else:
            # ===== FILE DATABASE =====
            # Create a new connection and close it when done
            db_path_str = str(self.db_path)
            conn = sqlite3.connect(db_path_str)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            try:
                yield conn
            finally:
                conn.close()
    
    # ========================================
    # SOURCE OPERATIONS
    # ========================================
    
    def add_source(self, source: LogSource) -> None:
        """Add a new log source"""
        with self.get_connection() as conn:
            try:
                conn.execute("""
                    INSERT INTO sources (
                        id, name, source_type, path, eventlog_name, 
                        is_active, created_at, last_indexed, total_chunks
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    source.id,
                    source.name,
                    source.source_type.value,
                    source.path,
                    source.eventlog_name,
                    int(source.is_active),
                    source.created_at.isoformat(),
                    source.last_indexed.isoformat() if source.last_indexed else None,
                    source.total_chunks
                ))
                conn.commit()
            except sqlite3.IntegrityError as e:
                raise ValueError(f"Source with name '{source.name}' already exists") from e
    
    def get_source(self, source_id: str) -> Optional[LogSource]:
        """Get a source by ID"""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM sources WHERE id = ?",
                (source_id,)
            ).fetchone()
            
            if row:
                return self._row_to_source(row)
            return None
    
    def get_source_by_name(self, name: str) -> Optional[LogSource]:
        """Get a source by name"""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM sources WHERE name = ?",
                (name,)
            ).fetchone()
            
            if row:
                return self._row_to_source(row)
            return None
    
    def list_sources(self, active_only: bool = False) -> List[LogSource]:
        """
        List all sources
        
        Args:
            active_only: If True, only return active sources
        """
        with self.get_connection() as conn:
            query = "SELECT * FROM sources"
            if active_only:
                query += " WHERE is_active = 1"
            query += " ORDER BY created_at DESC"
            
            rows = conn.execute(query).fetchall()
            return [self._row_to_source(row) for row in rows]
    
    def update_source_stats(
        self,
        source_id: str,
        total_chunks: int,
        last_indexed: datetime
    ) -> None:
        """Update source statistics after indexing"""
        with self.get_connection() as conn:
            conn.execute("""
                UPDATE sources 
                SET total_chunks = ?, last_indexed = ?
                WHERE id = ?
            """, (total_chunks, last_indexed.isoformat(), source_id))
            conn.commit()
    
    def toggle_source(self, source_id: str, is_active: bool) -> None:
        """Enable or disable a source"""
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE sources SET is_active = ? WHERE id = ?",
                (int(is_active), source_id)
            )
            conn.commit()
    
    def delete_source(self, source_id: str) -> None:
        """
        Delete a source and all its chunks
        Foreign key cascade will handle chunk deletion
        """
        with self.get_connection() as conn:
            conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))
            conn.commit()
    
    def get_source_stats(self) -> Dict[str, Any]:
        """Get overall statistics about sources"""
        with self.get_connection() as conn:
            stats = conn.execute("""
                SELECT 
                    COUNT(*) as total_sources,
                    SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END) as active_sources,
                    SUM(total_chunks) as total_chunks,
                    SUM(CASE WHEN source_type = 'file' THEN 1 ELSE 0 END) as file_sources,
                    SUM(CASE WHEN source_type = 'folder' THEN 1 ELSE 0 END) as folder_sources,
                    SUM(CASE WHEN source_type = 'eventviewer' THEN 1 ELSE 0 END) as eventviewer_sources
                FROM sources
            """).fetchone()
            
            return dict(stats) if stats else {}
    
    # ========================================
    # CHUNK OPERATIONS
    # ========================================
    
    def add_chunk(self, chunk: LogChunk) -> None:
        """Add a single chunk"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO chunks (
                    id, source_id, content, timestamp, log_level,
                    metadata, embedding_id, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk.id,
                chunk.source_id,
                chunk.content,
                chunk.timestamp.isoformat(),
                chunk.log_level.value,
                json.dumps(chunk.metadata),
                chunk.embedding_id,
                datetime.now().isoformat()
            ))
            conn.commit()
    
    def add_chunks_batch(self, chunks: List[LogChunk]) -> None:
        """
        Add multiple chunks at once
        MUCH faster than individual inserts
        """
        if not chunks:
            return
        
        with self.get_connection() as conn:
            now = datetime.now().isoformat()
            conn.executemany("""
                INSERT INTO chunks (
                    id, source_id, content, timestamp, log_level,
                    metadata, embedding_id, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    c.id,
                    c.source_id,
                    c.content,
                    c.timestamp.isoformat(),
                    c.log_level.value,
                    json.dumps(c.metadata),
                    c.embedding_id,
                    now
                )
                for c in chunks
            ])
            conn.commit()
    
    def get_chunk(self, chunk_id: str) -> Optional[LogChunk]:
        """Get a single chunk by ID"""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM chunks WHERE id = ?",
                (chunk_id,)
            ).fetchone()
            
            if row:
                return self._row_to_chunk(row)
            return None
    
    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[LogChunk]:
        """Get multiple chunks by IDs (preserves order)"""
        if not chunk_ids:
            return []
        
        with self.get_connection() as conn:
            placeholders = ','.join('?' * len(chunk_ids))
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE id IN ({placeholders})",
                chunk_ids
            ).fetchall()
            
            # Preserve order from chunk_ids
            chunks_dict = {self._row_to_chunk(row).id: self._row_to_chunk(row) for row in rows}
            return [chunks_dict[cid] for cid in chunk_ids if cid in chunks_dict]
    
    def get_chunks_by_embedding_ids(self, embedding_ids: List[int]) -> List[LogChunk]:
        """
        Get chunks by their embedding IDs (FAISS indices)
        This is what we use after vector search
        """
        if not embedding_ids:
            return []
        
        with self.get_connection() as conn:
            placeholders = ','.join('?' * len(embedding_ids))
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE embedding_id IN ({placeholders})",
                embedding_ids
            ).fetchall()
            
            return [self._row_to_chunk(row) for row in rows]
    
    def get_chunks_by_source(
        self,
        source_id: str,
        limit: Optional[int] = None
    ) -> List[LogChunk]:
        """Get all chunks for a source"""
        with self.get_connection() as conn:
            query = "SELECT * FROM chunks WHERE source_id = ? ORDER BY timestamp DESC"
            if limit:
                query += f" LIMIT {limit}"
            
            rows = conn.execute(query, (source_id,)).fetchall()
            return [self._row_to_chunk(row) for row in rows]
    
    def delete_chunks_by_source(self, source_id: str) -> int:
        """Delete all chunks for a source, returns count of deleted chunks"""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM chunks WHERE source_id = ?",
                (source_id,)
            )
            conn.commit()
            return cursor.rowcount
    
    def get_chunk_count(self, source_id: Optional[str] = None) -> int:
        """Get total chunk count, optionally filtered by source"""
        with self.get_connection() as conn:
            if source_id:
                result = conn.execute(
                    "SELECT COUNT(*) as count FROM chunks WHERE source_id = ?",
                    (source_id,)
                ).fetchone()
            else:
                result = conn.execute("SELECT COUNT(*) as count FROM chunks").fetchone()
            
            return result['count'] if result else 0
    
    def search_chunks_by_text(
        self,
        query: str,
        limit: int = 50
    ) -> List[LogChunk]:
        """
        Simple text search in chunks (for fallback/testing)
        This is NOT semantic search - just basic SQL LIKE
        """
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM chunks 
                WHERE content LIKE ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (f"%{query}%", limit)).fetchall()
            
            return [self._row_to_chunk(row) for row in rows]
    
    # ========================================
    # CHAT HISTORY OPERATIONS
    # ========================================
    
    def add_chat_message(self, message: ChatMessage) -> None:
        """Add a message to chat history"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO chat_history (role, content, timestamp, sources)
                VALUES (?, ?, ?, ?)
            """, (
                message.role,
                message.content,
                message.timestamp.isoformat(),
                json.dumps(message.sources)
            ))
            conn.commit()
    
    def get_chat_history(self, limit: int = 50) -> List[ChatMessage]:
        """
        Get recent chat history
        Returns in chronological order (oldest first)
        """
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM chat_history 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,)).fetchall()
            
            # Reverse to get chronological order
            return [self._row_to_message(row) for row in reversed(rows)]
    
    def clear_chat_history(self) -> None:
        """Clear all chat history"""
        with self.get_connection() as conn:
            conn.execute("DELETE FROM chat_history")
            conn.commit()
    
    def get_last_n_messages(self, n: int = 10) -> List[ChatMessage]:
        """Get last N messages for context window"""
        return self.get_chat_history(limit=n)
    
    # ========================================
    # UTILITY / MAINTENANCE
    # ========================================
    
    def vacuum(self) -> None:
        """Optimize database (reclaim space, rebuild indices)"""
        with self.get_connection() as conn:
            conn.execute("VACUUM")
            conn.commit()
    
    def get_db_size(self) -> int:
        """Get database file size in bytes"""
        if isinstance(self.db_path, str) and self.db_path == ":memory:":
            return 0  # In-memory DB has no file size
        return self.db_path.stat().st_size if Path(self.db_path).exists() else 0

    def get_db_size_mb(self) -> float:
        """Get database file size in MB"""
        return self.get_db_size() / (1024 * 1024)

    def backup(self, backup_path: Path) -> None:
        """Create a backup of the database"""
        if isinstance(self.db_path, str) and self.db_path == ":memory:":
            raise ValueError("Cannot backup in-memory database")
        
        import shutil
        shutil.copy2(self.db_path, backup_path)
    
    # ========================================
    # HELPER METHODS (Row -> Model)
    # ========================================
    
    def _row_to_source(self, row) -> LogSource:
        """Convert database row to LogSource model"""
        return LogSource(
            id=row['id'],
            name=row['name'],
            source_type=SourceType(row['source_type']),
            path=row['path'],
            eventlog_name=row['eventlog_name'],
            is_active=bool(row['is_active']),
            created_at=datetime.fromisoformat(row['created_at']),
            last_indexed=datetime.fromisoformat(row['last_indexed']) if row['last_indexed'] else None,
            total_chunks=row['total_chunks']
        )
    
    def _row_to_chunk(self, row) -> LogChunk:
        """Convert database row to LogChunk model"""
        return LogChunk(
            id=row['id'],
            source_id=row['source_id'],
            content=row['content'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            log_level=LogLevel(row['log_level']),
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            embedding_id=row['embedding_id']
        )
    
    def _row_to_message(self, row) -> ChatMessage:
        """Convert database row to ChatMessage model"""
        return ChatMessage(
            role=row['role'],
            content=row['content'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            sources=json.loads(row['sources']) if row['sources'] else []
        )


# ===== SINGLETON INSTANCE =====
# Create a global database instance
db = Database()