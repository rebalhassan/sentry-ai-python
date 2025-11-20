# sentry/services/log_watcher.py
"""
Log Watcher Service
Monitors files/folders for changes and auto-indexes new content

Uses watchdog library to detect file system events
Implements debouncing to avoid rapid re-indexing
Supports incremental parsing to only index new content
"""

import time
import logging
from pathlib import Path
from typing import Callable, Dict, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from ..core.models import LogSource, SourceType
from ..core.config import settings
from .indexer import LogIndexer

# Set up logger
logger = logging.getLogger(__name__)


class LogFileHandler(FileSystemEventHandler):
    """
    Handles file system events for log files
    
    Features:
    - Filters to only process supported log file extensions
    - Debounces rapid changes (avoids re-indexing during writes)
    - Triggers callback when files are created or modified
    """
    
    def __init__(self, callback: Callable[[Path], None]):
        """
        Initialize the handler
        
        Args:
            callback: Function to call when a file changes (receives file path)
        """
        super().__init__()
        self.callback = callback
        self.last_modified: Dict[Path, float] = {}  # Track last modification times
    
    def on_modified(self, event: FileSystemEvent):
        """
        Called when a file is modified
        
        Args:
            event: File system event from watchdog
        """
        # Ignore directory events
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Only process supported log files
        if not settings.is_supported_file(file_path):
            return
        
        # Debounce: ignore if file was modified recently
        now = time.time()
        last_time = self.last_modified.get(file_path, 0)
        
        if now - last_time < settings.watch_debounce:
            return
        
        # Update last modified time
        self.last_modified[file_path] = now
        
        # Trigger callback
        logger.debug(f"File modified: {file_path}")
        self.callback(file_path)
    
    def on_created(self, event: FileSystemEvent):
        """
        Called when a file is created
        
        Args:
            event: File system event from watchdog
        """
        if not event.is_directory:
            # Treat creation as modification
            self.on_modified(event)


class LogWatcher:
    """
    Watches log files/folders for changes and auto-indexes new content
    
    Usage:
        watcher = LogWatcher(indexer, index_callback)
        watcher.watch_source(source)
        watcher.start()
        # ... keep running ...
        watcher.stop()
    """
    
    def __init__(
        self, 
        indexer: LogIndexer, 
        index_callback: Callable[[list, str], None]
    ):
        """
        Initialize the watcher
        
        Args:
            indexer: LogIndexer instance for parsing files
            index_callback: Function to call with new chunks (receives chunks, source_id)
                           Typically: rag_service.index_chunks_batch
        """
        self.indexer = indexer
        self.index_callback = index_callback
        self.observer = Observer()
        self.watching: Dict[str, any] = {}  # {source_id: watch_handle}
        self.file_positions: Dict[str, int] = {}  # {file_path: last_position}
        
        logger.info("LogWatcher initialized")
    
    def watch_source(self, source: LogSource):
        """
        Start watching a log source
        
        Args:
            source: LogSource to monitor
        """
        # Skip EventViewer sources (handled separately)
        if source.source_type == SourceType.EVENTVIEWER:
            logger.info(f"Skipping EventViewer source: {source.name}")
            return
        
        # Validate path
        if not source.path:
            logger.error(f"Source {source.name} has no path")
            return
        
        path = Path(source.path)
        if not path.exists():
            logger.error(f"Path doesn't exist: {path}")
            return
        
        # Create handler with callback
        handler = LogFileHandler(
            callback=lambda file_path: self._on_file_changed(file_path, source.id)
        )
        
        # Determine watch path
        # If source is a file, watch its parent directory
        # If source is a folder, watch the folder
        if path.is_file():
            watch_path = path.parent
            logger.info(f"👁️  Watching file: {path}")
        else:
            watch_path = path
            logger.info(f"👁️  Watching folder: {path}")
        
        # Schedule the watch
        watch = self.observer.schedule(
            handler,
            str(watch_path),
            recursive=True  # Watch subdirectories too
        )
        
        # Track the watch
        self.watching[source.id] = watch
        
        logger.info(f"✅ Now watching source: {source.name} ({source.id})")
    
    def _on_file_changed(self, file_path: Path, source_id: str):
        """
        Handle file change event
        
        Args:
            file_path: Path to the changed file
            source_id: ID of the source this file belongs to
        """
        logger.info(f"📝 File changed: {file_path}")
        
        try:
            # Check if we have a last position for incremental parsing
            file_key = str(file_path)
            last_position = self.file_positions.get(file_key, 0)
            
            # Parse new content (incremental if possible)
            if last_position > 0 and file_path.stat().st_size >= last_position:
                # Use incremental parsing
                chunks, new_position = self.indexer.parse_file_incremental(
                    file_path, source_id, last_position
                )
                self.file_positions[file_key] = new_position
                logger.info(f"📊 Incremental parse: {last_position} → {new_position} bytes")
            else:
                # Full parse (first time or file was truncated)
                chunks = self.indexer.parse_file(file_path, source_id)
                # Update position to end of file
                self.file_positions[file_key] = file_path.stat().st_size
                logger.info(f"📊 Full parse: {len(chunks)} chunks")
            
            # Index the chunks
            if chunks:
                self.index_callback(chunks, source_id)
                logger.info(f"✅ Indexed {len(chunks)} new chunks from {file_path.name}")
            else:
                logger.debug(f"No new content in {file_path.name}")
                
        except Exception as e:
            logger.error(f"❌ Failed to index {file_path}: {e}", exc_info=True)
    
    def start(self):
        """Start the file watcher observer"""
        if not self.observer.is_alive():
            self.observer.start()
            logger.info("👁️  File watcher started")
        else:
            logger.warning("File watcher already running")
    
    def stop(self):
        """Stop the file watcher observer"""
        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join(timeout=5)
            logger.info("👁️  File watcher stopped")
        else:
            logger.warning("File watcher not running")
    
    def unwatch_source(self, source_id: str):
        """
        Stop watching a specific source
        
        Args:
            source_id: ID of the source to stop watching
        """
        if source_id in self.watching:
            watch = self.watching[source_id]
            try:
                self.observer.unschedule(watch)
            except (KeyError, Exception) as e:
                logger.warning(f"Error unscheduling watch for {source_id}: {e}")
            del self.watching[source_id]
            logger.info(f"✅ Stopped watching source: {source_id}")
        else:
            logger.warning(f"Source not being watched: {source_id}")
    
    def is_watching(self, source_id: str) -> bool:
        """
        Check if a source is being watched
        
        Args:
            source_id: ID of the source to check
            
        Returns:
            True if source is being watched
        """
        return source_id in self.watching
    
    def get_watched_count(self) -> int:
        """
        Get number of sources being watched
        
        Returns:
            Number of active watches
        """
        return len(self.watching)
