# sentry/services/indexer.py
"""
Log Indexer Service
Parses log files into LogChunk objects for AI-powered diagnostics

Strategy (per docs.md lines 761-766):
1. Read file line by line (memory efficient)
2. Detect log format (regex patterns for common formats)
3. Extract: timestamp, level, message
4. Chunk by size (settings.chunk_size characters)
5. Create LogChunk objects
"""

import re
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from dateutil import parser as date_parser

from ..core.models import LogChunk, LogLevel
from ..core.config import settings


class LogIndexer:
    """
    Parse log files and Windows Event Viewer records into LogChunk objects
    
    Supports:
    - .log files (standard log format)
    - .txt files (plain text logs)
    - .csv files (CSV-formatted logs)
    - Windows Event Viewer PyEventLogRecord objects
    """
    
    # Regex patterns for log levels (case-insensitive)
    LEVEL_PATTERNS = {
        LogLevel.CRITICAL: re.compile(r'\b(CRITICAL|CRIT|FATAL|SEVERE)\b', re.I),
        LogLevel.ERROR: re.compile(r'\b(ERROR|ERR|FAIL|FAILED)\b', re.I),
        LogLevel.WARNING: re.compile(r'\b(WARN|WARNING)\b', re.I),
        LogLevel.INFO: re.compile(r'\b(INFO|INFORMATION)\b', re.I),
        LogLevel.DEBUG: re.compile(r'\b(DEBUG|DBG|TRACE)\b', re.I),
    }
    
    # Windows Event Viewer event type mapping
    # EventType values from win32evtlog
    EVENTLOG_TYPE_MAPPING = {
        1: LogLevel.ERROR,      # EVENTLOG_ERROR_TYPE
        2: LogLevel.WARNING,    # EVENTLOG_WARNING_TYPE
        4: LogLevel.INFO,       # EVENTLOG_INFORMATION_TYPE
        8: LogLevel.INFO,       # EVENTLOG_AUDIT_SUCCESS
        16: LogLevel.WARNING,   # EVENTLOG_AUDIT_FAILURE
    }
    
    def parse_file(self, file_path: Path, source_id: str) -> List[LogChunk]:
        """
        Parse a single log file into LogChunk objects
        
        Args:
            file_path: Path to the log file (.log, .txt, .csv)
            source_id: ID of the LogSource this file belongs to
            
        Returns:
            List of LogChunk objects
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is not supported
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Log file not found: {file_path}")
        
        # Check file size
        if settings.is_file_too_large(file_path):
            raise ValueError(
                f"File too large: {file_path.stat().st_size / (1024*1024):.1f}MB "
                f"(max: {settings.max_file_size_mb}MB)"
            )
        
        # Parse based on file extension
        extension = file_path.suffix.lower()
        
        if extension == '.csv':
            return self._parse_csv_file(file_path, source_id)
        elif extension in ['.log', '.txt', '.out', '.err']:
            return self._parse_text_file(file_path, source_id)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")
    
    def parse_folder(self, folder_path: Path, source_id: str) -> List[LogChunk]:
        """
        Recursively parse all supported log files in a folder
        
        Args:
            folder_path: Path to the folder containing log files
            source_id: ID of the LogSource this folder belongs to
            
        Returns:
            List of LogChunk objects from all files
        """
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"Invalid folder path: {folder_path}")
        
        all_chunks = []
        
        # Find all supported files
        for ext in settings.supported_extensions:
            for file_path in folder_path.rglob(f"*{ext}"):
                # Skip files that are too large
                if settings.is_file_too_large(file_path):
                    print(f"⚠️  Skipping large file: {file_path}")
                    continue
                
                try:
                    chunks = self.parse_file(file_path, source_id)
                    all_chunks.extend(chunks)
                except Exception as e:
                    print(f"⚠️  Error parsing {file_path}: {e}")
                    continue
        
        return all_chunks
    
    def parse_file_incremental(
        self, 
        file_path: Path, 
        source_id: str,
        last_position: int = 0
    ) -> tuple[List[LogChunk], int]:
        """
        Parse only new content since last_position (for file watcher)
        
        This is more efficient than re-parsing the entire file when monitoring changes.
        
        Args:
            file_path: Path to the log file
            source_id: ID of the LogSource
            last_position: Byte position where we last read the file
            
        Returns:
            Tuple of (chunks, new_position)
            - chunks: List of LogChunk objects from new content
            - new_position: New byte position to use for next incremental parse
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Log file not found: {file_path}")
        
        chunks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Seek to last position
                f.seek(last_position)
                
                # Read new content
                new_content = f.read()
                
                # Get new position
                new_position = f.tell()
            
            # Parse new content if it exists
            if new_content.strip():
                chunk_texts = self._chunk_content(new_content)
                
                for i, chunk_text in enumerate(chunk_texts):
                    chunk = LogChunk(
                        source_id=source_id,
                        content=chunk_text,
                        timestamp=self._extract_timestamp(chunk_text) or datetime.now(),
                        log_level=self._detect_log_level(chunk_text),
                        metadata={
                            'file': str(file_path),
                            'position': last_position,
                            'chunk_index': i,
                            'incremental': True
                        }
                    )
                    chunks.append(chunk)
            else:
                # No new content, return same position
                new_position = last_position
                
        except Exception as e:
            raise ValueError(f"Error reading file {file_path} incrementally: {e}")
        
        return chunks, new_position
    
    def parse_eventviewer_record(self, record: Any, source_id: str) -> LogChunk:
        """
        Parse a Windows Event Viewer PyEventLogRecord into a LogChunk
        
        Args:
            record: PyEventLogRecord from win32evtlog
            source_id: ID of the LogSource (EventViewer)
            
        Returns:
            Single LogChunk object
        """
        # Extract timestamp
        timestamp = datetime.fromtimestamp(record.TimeGenerated)
        
        # Map EventType to LogLevel
        log_level = self.EVENTLOG_TYPE_MAPPING.get(record.EventType, LogLevel.INFO)
        
        # Build content from record
        # StringInserts contains the actual message data
        message_parts = record.StringInserts if record.StringInserts else []
        message = " ".join(str(part) for part in message_parts if part)
        
        # Create formatted content
        content = f"[Event ID: {record.EventID}] [{record.SourceName}]\n{message}"
        
        # Create metadata
        metadata = {
            'event_id': record.EventID,
            'source_name': record.SourceName,
            'event_type': record.EventType,
            'computer': getattr(record, 'ComputerName', 'Unknown'),
            'record_number': record.RecordNumber,
        }
        
        return LogChunk(
            source_id=source_id,
            content=content,
            timestamp=timestamp,
            log_level=log_level,
            metadata=metadata
        )
    
    def _parse_text_file(self, file_path: Path, source_id: str) -> List[LogChunk]:
        """
        Parse a text-based log file (.log, .txt, .out, .err)
        
        Strategy: Read entire file, chunk by size with overlap
        """
        chunks = []
        
        # Read file content (ignore encoding errors)
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            raise ValueError(f"Error reading file {file_path}: {e}")
        
        # Split into chunks
        chunk_texts = self._chunk_content(content)
        
        # Create LogChunk objects
        for i, chunk_text in enumerate(chunk_texts):
            chunk = LogChunk(
                source_id=source_id,
                content=chunk_text,
                timestamp=self._extract_timestamp(chunk_text) or datetime.now(),
                log_level=self._detect_log_level(chunk_text),
                metadata={
                    'file': str(file_path),
                    'chunk_index': i,
                    'total_chunks': len(chunk_texts)
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _parse_csv_file(self, file_path: Path, source_id: str) -> List[LogChunk]:
        """
        Parse a CSV log file
        
        Expects columns like: timestamp, level, message (flexible)
        """
        chunks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                
                # Collect all rows into content
                rows = list(reader)
                
                # Convert rows to text
                all_lines = []
                for row in rows:
                    # Try to format as: timestamp | level | message
                    line_parts = []
                    
                    # Look for common column names
                    timestamp_col = next((k for k in row.keys() if 'time' in k.lower() or 'date' in k.lower()), None)
                    level_col = next((k for k in row.keys() if 'level' in k.lower() or 'severity' in k.lower()), None)
                    message_col = next((k for k in row.keys() if 'message' in k.lower() or 'msg' in k.lower()), None)
                    
                    if timestamp_col:
                        line_parts.append(row[timestamp_col])
                    if level_col:
                        line_parts.append(row[level_col])
                    if message_col:
                        line_parts.append(row[message_col])
                    else:
                        # If no message column, use all remaining columns
                        remaining = {k: v for k, v in row.items() if k not in [timestamp_col, level_col]}
                        line_parts.append(str(remaining))
                    
                    all_lines.append(" | ".join(line_parts))
                
                content = "\n".join(all_lines)
                
        except Exception as e:
            raise ValueError(f"Error parsing CSV file {file_path}: {e}")
        
        # Chunk the content
        chunk_texts = self._chunk_content(content)
        
        # Create LogChunk objects
        for i, chunk_text in enumerate(chunk_texts):
            chunk = LogChunk(
                source_id=source_id,
                content=chunk_text,
                timestamp=self._extract_timestamp(chunk_text) or datetime.now(),
                log_level=self._detect_log_level(chunk_text),
                metadata={
                    'file': str(file_path),
                    'format': 'csv',
                    'chunk_index': i,
                    'total_chunks': len(chunk_texts)
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_content(self, content: str) -> List[str]:
        """
        Split content into overlapping chunks
        
        Uses settings.chunk_size and settings.chunk_overlap
        """
        chunks = []
        size = settings.chunk_size
        overlap = settings.chunk_overlap
        
        # Edge case: empty content
        if not content:
            return []
        
        # Chunk with overlap
        i = 0
        while i < len(content):
            chunk = content[i:i + size]
            
            # Only add if it meets minimum size (unless it's the last chunk)
            if len(chunk) >= settings.min_chunk_size or i + size >= len(content):
                chunks.append(chunk)
            
            # Move forward by (size - overlap)
            i += size - overlap
        
        return chunks
    
    def _detect_log_level(self, text: str) -> LogLevel:
        """
        Detect log level from text using regex patterns
        
        Returns LogLevel.UNKNOWN if no level is detected
        """
        # Check in priority order (most severe first)
        for level in [LogLevel.CRITICAL, LogLevel.ERROR, LogLevel.WARNING, 
                      LogLevel.INFO, LogLevel.DEBUG]:
            pattern = self.LEVEL_PATTERNS.get(level)
            if pattern and pattern.search(text):
                return level
        
        return LogLevel.UNKNOWN
    
    def _extract_timestamp(self, text: str) -> Optional[datetime]:
        """
        Try to extract timestamp from the first line of text
        
        Uses dateutil.parser for flexible parsing
        """
        try:
            # Get first line
            first_line = text.split('\n')[0]
            
            # Try to parse with dateutil (fuzzy mode finds dates in text)
            return date_parser.parse(first_line, fuzzy=True)
        except:
            # If parsing fails, return None
            return None
