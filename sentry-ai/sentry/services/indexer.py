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
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dateutil import parser as date_parser

from ..core.models import LogChunk, LogLevel
from ..core.config import settings
from .helix import get_helix_service, HelixService

logger = logging.getLogger(__name__)


class LogIndexer:
    """
    Parse log files and Windows Event Viewer records into LogChunk objects
    
    Supports:
    - .log files (standard log format)
    - .txt files (plain text logs)
    - .csv files (CSV-formatted logs)
    - Windows Event Viewer PyEventLogRecord objects
    """
    
    # Regex patterns for log levels (case-insensitive) - pre-compiled at class level
    LEVEL_PATTERNS = {
        LogLevel.CRITICAL: re.compile(r'\b(CRITICAL|CRIT|FATAL|SEVERE)\b', re.I),
        LogLevel.ERROR: re.compile(r'\b(ERROR|ERR|FAIL|FAILED)\b', re.I),
        LogLevel.WARNING: re.compile(r'\b(WARN|WARNING)\b', re.I),
        LogLevel.INFO: re.compile(r'\b(INFO|INFORMATION)\b', re.I),
        LogLevel.DEBUG: re.compile(r'\b(DEBUG|DBG|TRACE)\b', re.I),
    }
    
    # Pre-compiled log entry patterns for semantic chunking (compiled once at class level)
    # Detects entry boundaries using timestamps, log levels, and event formats
    _LOG_ENTRY_PATTERNS = [
        # ISO datetime: 2024-01-15 10:30:45 or 2024-01-15T10:30:45
        r'^\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}',
        # Bracket datetime: [2024-01-15 10:30:45]
        r'^\[\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}\]',
        # Time only: [10:30:45] or 10:30:45,123
        r'^\[?\d{2}:\d{2}:\d{2}[,.]?\d*\]?',
        # Month format: Jan 15 10:30:45 or 15/Jan/2024
        r'^[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}',
        r'^\d{1,2}/[A-Z][a-z]{2}/\d{4}',
        # Log level at start: INFO, ERROR, WARN, DEBUG
        r'^(INFO|ERROR|WARN|WARNING|DEBUG|CRITICAL|FATAL|TRACE)\s*[:\|\-\]]',
        # Windows Event format: [Event ID: 1234]
        r'^\[Event\s+ID:\s*\d+\]',
    ]
    _COMPILED_ENTRY_PATTERN = re.compile(
        '|'.join(f'({p})' for p in _LOG_ENTRY_PATTERNS), 
        re.MULTILINE | re.IGNORECASE
    )
    
    # Windows Event Viewer event type mapping
    # EventType values from win32evtlog
    EVENTLOG_TYPE_MAPPING = {
        1: LogLevel.ERROR,      # EVENTLOG_ERROR_TYPE
        2: LogLevel.WARNING,    # EVENTLOG_WARNING_TYPE
        4: LogLevel.INFO,       # EVENTLOG_INFORMATION_TYPE
        8: LogLevel.INFO,       # EVENTLOG_AUDIT_SUCCESS
        16: LogLevel.WARNING,   # EVENTLOG_AUDIT_FAILURE
    }
    
    def parse_file(
        self, 
        file_path: Path, 
        source_id: str,
        skip_helix: bool = False
    ) -> List[LogChunk]:
        """
        Parse a single log file into LogChunk objects.
        
        Chunks are automatically annotated with Helix Vector metadata
        (cluster IDs, anomaly scores, severity weights) unless skip_helix=True.
        
        Args:
            file_path: Path to the log file (.log, .txt, .csv)
            source_id: ID of the LogSource this file belongs to
            skip_helix: If True, skip Helix annotation (used for batch processing)
            
        Returns:
            List of LogChunk objects (with Helix annotations if skip_helix=False)
            
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
            chunks = self._parse_csv_file(file_path, source_id)
        elif extension in ['.log', '.txt', '.out', '.err']:
            chunks = self._parse_text_file(file_path, source_id)
        else:
            raise ValueError(f"Unsupported file extension: {extension}")
        
        # Annotate with Helix Vector (skip if doing batch annotation later)
        if not skip_helix:
            return self._annotate_with_helix(chunks)
        return chunks
    
    def parse_folder(self, folder_path: Path, source_id: str) -> List[LogChunk]:
        """
        Recursively parse all supported log files in a folder.
        
        Uses parallel processing with ThreadPoolExecutor for faster indexing.
        Helix annotation is applied once at the end for the entire batch,
        which provides better Markov chain learning from the full sequence.
        
        Args:
            folder_path: Path to the folder containing log files
            source_id: ID of the LogSource this folder belongs to
            
        Returns:
            List of LogChunk objects from all files with Helix annotations
        """
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"Invalid folder path: {folder_path}")
        
        # Collect all files to process
        files_to_process = []
        for ext in settings.supported_extensions:
            for file_path in folder_path.rglob(f"*{ext}"):
                # Skip files that are too large
                if settings.is_file_too_large(file_path):
                    logger.warning("Skipping large file: %s", file_path)
                    continue
                files_to_process.append(file_path)
        
        if not files_to_process:
            logger.info("No supported files found in %s", folder_path)
            return []
        
        all_chunks = []
        failed_files = []
        
        # Use parallel processing if enabled and multiple files exist
        if settings.parallel_indexing and len(files_to_process) > 1:
            # Determine worker count (max_workers, but at least 1)
            num_workers = max(1, min(settings.max_workers, len(files_to_process)))
            logger.info(
                "Parallel indexing %d files with %d workers", 
                len(files_to_process), num_workers
            )
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all file parsing jobs (skip_helix=True for batch annotation)
                future_to_file = {
                    executor.submit(
                        self.parse_file, file_path, source_id, True
                    ): file_path 
                    for file_path in files_to_process
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        chunks = future.result()
                        all_chunks.extend(chunks)
                    except Exception as e:
                        # Log error but continue processing other files
                        logger.warning("Error parsing %s: %s", file_path, e)
                        failed_files.append((file_path, str(e)))
                        continue
        else:
            # Sequential processing (single file or parallel disabled)
            for file_path in files_to_process:
                try:
                    chunks = self.parse_file(file_path, source_id, skip_helix=True)
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.warning("Error parsing %s: %s", file_path, e)
                    failed_files.append((file_path, str(e)))
                    continue
        
        # Log summary
        if failed_files:
            logger.warning(
                "Completed with %d errors out of %d files", 
                len(failed_files), len(files_to_process)
            )
        else:
            logger.info(
                "Successfully parsed %d files, %d chunks", 
                len(files_to_process), len(all_chunks)
            )
        
        # Apply Helix annotation to ALL chunks at once (better Markov learning)
        return self._annotate_with_helix(all_chunks)
    
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
        
        # Annotate with Helix Vector
        chunks = self._annotate_with_helix(chunks)
        
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
        Split content into semantically meaningful chunks
        
        Strategy:
        1. Detect log entry boundaries (timestamps, log levels)
        2. Split into individual log entries
        3. Group entries into chunks respecting semantic boundaries
        4. Ensure chunks don't exceed max size
        
        This preserves complete log entries and improves retrieval accuracy.
        """
        if not content:
            return []
        
        # Try semantic chunking first
        entries = self._split_into_log_entries(content)
        
        if entries:
            # Group entries into chunks
            return self._group_entries_into_chunks(entries)
        else:
            # Fallback to sentence-based chunking for unstructured text
            return self._chunk_by_sentences(content)
    
    def _split_into_log_entries(self, content: str) -> List[str]:
        """
        Split content into individual log entries based on common patterns.
        
        Uses pre-compiled regex patterns at class level for performance.
        
        Detects entry boundaries using:
        - ISO timestamps (2024-01-15 10:30:45)
        - Bracket timestamps ([2024-01-15], [10:30:45])
        - Log levels at start of line (INFO, ERROR, etc.)
        - Windows Event format
        """
        # Use pre-compiled class-level pattern (self._COMPILED_ENTRY_PATTERN)
        lines = content.split('\n')
        entries = []
        current_entry_lines = []
        
        for line in lines:
            # Check if this line starts a new entry using pre-compiled pattern
            if self._COMPILED_ENTRY_PATTERN.match(line.strip()):
                # Save previous entry if exists
                if current_entry_lines:
                    entry_text = '\n'.join(current_entry_lines).strip()
                    if entry_text:
                        entries.append(entry_text)
                # Start new entry
                current_entry_lines = [line]
            else:
                # Continue current entry (multi-line log entry)
                current_entry_lines.append(line)
        
        # Don't forget the last entry
        if current_entry_lines:
            entry_text = '\n'.join(current_entry_lines).strip()
            if entry_text:
                entries.append(entry_text)
        
        # Only return entries if we found meaningful structure
        # If we only got 1-2 big chunks, the detection probably failed
        if len(entries) >= 3 or (len(entries) > 0 and all(len(e) < settings.chunk_size * 2 for e in entries)):
            return entries
        
        return []  # Fallback to sentence-based chunking
    
    def _group_entries_into_chunks(self, entries: List[str]) -> List[str]:
        """
        Group log entries into chunks respecting semantic boundaries
        
        Groups entries until chunk_size is reached, then starts a new chunk.
        Ensures we don't split individual entries.
        """
        chunks = []
        current_chunk_entries = []
        current_size = 0
        
        for entry in entries:
            entry_size = len(entry)
            
            # If single entry exceeds chunk size, it becomes its own chunk
            if entry_size > settings.chunk_size:
                # Save current chunk if exists
                if current_chunk_entries:
                    chunks.append('\n\n'.join(current_chunk_entries))
                    current_chunk_entries = []
                    current_size = 0
                
                # Add oversized entry as its own chunk (will be truncated in embedding anyway)
                chunks.append(entry)
                continue
            
            # Check if adding this entry exceeds chunk size
            new_size = current_size + entry_size + 2  # +2 for separator
            
            if new_size > settings.chunk_size and current_chunk_entries:
                # Save current chunk and start new one
                chunks.append('\n\n'.join(current_chunk_entries))
                current_chunk_entries = [entry]
                current_size = entry_size
            else:
                # Add to current chunk
                current_chunk_entries.append(entry)
                current_size = new_size
        
        # Don't forget the last chunk
        if current_chunk_entries:
            chunks.append('\n\n'.join(current_chunk_entries))
        
        return chunks
    
    def _chunk_by_sentences(self, content: str) -> List[str]:
        """
        Fallback: Split by sentences for unstructured text
        
        Uses common sentence-ending patterns and respects word boundaries.
        """
        # Sentence-ending patterns
        sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        
        # Split into sentences
        sentences = sentence_endings.split(content)
        
        if not sentences:
            return [content] if content.strip() else []
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_size = len(sentence)
            
            if current_size + sentence_size + 1 > settings.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size + 1
        
        # Don't forget last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Filter out tiny chunks
        chunks = [c for c in chunks if len(c) >= settings.min_chunk_size]
        
        return chunks if chunks else [content[:settings.chunk_size]]
    
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
    
    def _annotate_with_helix(self, chunks: List[LogChunk]) -> List[LogChunk]:
        """
        Annotate chunks with Helix Vector metadata.
        
        Uses HelixService to add:
        - cluster_id: DNA cluster from Drain3 pattern mining
        - cluster_template: The pattern template for this cluster
        - is_anomaly: Whether this chunk was flagged as anomalous
        - anomaly_type: Classification (e.g., "database_timeout")
        - anomaly_score: How anomalous (0.0 = normal, 1.0 = very rare)
        - severity_weight: Severity penalty from template keywords
        - transition_prob: Markov chain transition probability
        
        Args:
            chunks: List of LogChunk objects to annotate
            
        Returns:
            Same list of chunks with Helix fields populated
        """
        if not chunks:
            return chunks
        
        # Check if Helix is enabled in config
        if not settings.helix_enabled:
            logger.debug("Helix annotation disabled in config")
            return chunks
        
        try:
            helix = get_helix_service()
            return helix.annotate_chunks(chunks)
        except Exception as e:
            logger.error("Helix annotation failed: %s", e)
            # Return chunks without Helix annotation on error
            return chunks

