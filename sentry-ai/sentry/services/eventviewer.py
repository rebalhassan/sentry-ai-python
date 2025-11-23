# sentry/services/eventviewer.py
"""
EventViewer Integration for Windows Event Logs
Reads Windows Event Logs in real-time using pywin32
"""

try:
    import win32evtlog
    import win32evtlogutil
    import win32con
    WINDOWS_AVAILABLE = True
except ImportError:
    WINDOWS_AVAILABLE = False

from datetime import datetime, timedelta
from typing import List, Optional
import time
import logging

from ..core.models import LogChunk, LogLevel, LogSource, SourceType
from ..core.config import settings

logger = logging.getLogger(__name__)

if not WINDOWS_AVAILABLE:
    logger.warning("pywin32 not available - EventViewer support disabled")


class EventViewerReader:
    """Read Windows Event Logs"""
    
    # Map Windows event types to our LogLevel
    EVENT_TYPE_MAP = {}
    
    def __init__(self):
        if not WINDOWS_AVAILABLE:
            raise ImportError("pywin32 required for EventViewer support")
        
        # Initialize event type mapping after win32evtlog is imported
        self.EVENT_TYPE_MAP = {
            win32evtlog.EVENTLOG_ERROR_TYPE: LogLevel.ERROR,
            win32evtlog.EVENTLOG_WARNING_TYPE: LogLevel.WARNING,
            win32evtlog.EVENTLOG_INFORMATION_TYPE: LogLevel.INFO,
            win32evtlog.EVENTLOG_AUDIT_FAILURE: LogLevel.ERROR,
            win32evtlog.EVENTLOG_AUDIT_SUCCESS: LogLevel.INFO,
        }
        
        self.handles = {}  # {log_name: handle}
    
    def open_log(self, log_name: str):
        """
        Open an event log
        
        Common logs:
        - "System"
        - "Application"
        - "Security"
        """
        if log_name in self.handles:
            return
        
        try:
            handle = win32evtlog.OpenEventLog(None, log_name)
            self.handles[log_name] = handle
            logger.info(f"📖 Opened EventLog: {log_name}")
        except Exception as e:
            logger.error(f"Failed to open {log_name}: {e}")
            raise
    
    def close_log(self, log_name: str):
        """Close an event log"""
        if log_name in self.handles:
            win32evtlog.CloseEventLog(self.handles[log_name])
            del self.handles[log_name]
    
    def read_events(
        self,
        log_name: str,
        source_id: str,
        start_time: Optional[datetime] = None,
        max_events: int = None
    ) -> List[LogChunk]:
        """
        Read events from a log
        
        Args:
            log_name: Name of log ("System", "Application", etc.)
            source_id: Source ID for chunks
            start_time: Only events after this time (None = all)
            max_events: Max events to read (None = unlimited)
        
        Returns:
            List of LogChunks
        """
        if log_name not in self.handles:
            self.open_log(log_name)
        
        handle = self.handles[log_name]
        chunks = []
        
        flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
        
        max_events = max_events or settings.eventviewer_max_events
        
        try:
            events = win32evtlog.ReadEventLog(handle, flags, 0)
            
            for event in events:
                if len(chunks) >= max_events:
                    break
                
                # Get event time
                event_time = datetime.fromtimestamp(int(event.TimeGenerated))
                
                # Filter by start_time
                if start_time and event_time < start_time:
                    continue
                
                # Convert to LogChunk
                chunk = self._event_to_chunk(event, log_name, source_id)
                chunks.append(chunk)
        
        except Exception as e:
            logger.error(f"Failed to read events from {log_name}: {e}")
        
        return chunks
    
    def _event_to_chunk(
        self,
        event,
        log_name: str,
        source_id: str
    ) -> LogChunk:
        """Convert Windows event to LogChunk"""
        
        # Get event type (error, warning, info)
        event_type = event.EventType
        log_level = self.EVENT_TYPE_MAP.get(event_type, LogLevel.UNKNOWN)
        
        # Get event time
        timestamp = datetime.fromtimestamp(int(event.TimeGenerated))
        
        # Build content
        source_name = event.SourceName
        event_id = event.EventID & 0xFFFF  # Mask to get actual ID
        category = event.EventCategory
        
        # Get message (may need formatting)
        try:
            message = win32evtlogutil.SafeFormatMessage(event, log_name)
        except:
            message = f"Event ID {event_id}"
        
        content = f"[{source_name}] Event ID {event_id}: {message}"
        
        # Create chunk
        chunk = LogChunk(
            source_id=source_id,
            content=content,
            timestamp=timestamp,
            log_level=log_level,
            metadata={
                'log_name': log_name,
                'event_id': event_id,
                'source_name': source_name,
                'category': category,
                'computer': event.ComputerName
            }
        )
        
        return chunk
    
    def poll_new_events(
        self,
        log_name: str,
        source_id: str,
        last_check: datetime
    ) -> List[LogChunk]:
        """
        Poll for events since last_check
        
        Use this in a loop for real-time monitoring
        """
        return self.read_events(
            log_name=log_name,
            source_id=source_id,
            start_time=last_check,
            max_events=settings.eventviewer_max_events
        )
    
    def close_all(self):
        """Close all open logs"""
        for log_name in list(self.handles.keys()):
            self.close_log(log_name)


class EventViewerWatcher:
    """Continuously monitor EventViewer logs"""
    
    def __init__(self, reader: EventViewerReader, rag=None):
        self.reader = reader
        self.rag = rag
        self.watching = {}  # {source_id: (log_name, last_check)}
        self.running = False
    
    def watch_source(self, source: LogSource):
        """Start watching an EventViewer source"""
        if source.source_type != SourceType.EVENTVIEWER:
            return
        
        log_name = source.eventlog_name
        
        if not log_name:
            logger.error(f"EventViewer source {source.id} has no log_name")
            return
        
        # Open the log
        self.reader.open_log(log_name)
        
        # Track it
        self.watching[source.id] = (log_name, datetime.now())
        
        logger.info(f"👁️  Watching EventLog: {log_name}")
    
    def start(self):
        """Start polling loop"""
        self.running = True
        
        logger.info("👁️  EventViewer watcher started")
        
        while self.running:
            for source_id, (log_name, last_check) in list(self.watching.items()):
                try:
                    # Poll for new events
                    chunks = self.reader.poll_new_events(
                        log_name=log_name,
                        source_id=source_id,
                        last_check=last_check
                    )
                    
                    if chunks:
                        # Index new events if RAG service is available
                        if self.rag:
                            self.rag.index_chunks_batch(chunks)
                        logger.info(f"📝 Indexed {len(chunks)} events from {log_name}")
                    
                    # Update last check time
                    self.watching[source_id] = (log_name, datetime.now())
                
                except Exception as e:
                    logger.error(f"Error polling {log_name}: {e}")
            
            # Sleep before next poll
            time.sleep(settings.eventviewer_poll_interval)
    
    def stop(self):
        """Stop polling"""
        self.running = False
        self.reader.close_all()
        logger.info("👁️  EventViewer watcher stopped")
    
    def unwatch_source(self, source_id: str):
        """Stop watching a specific source"""
        if source_id in self.watching:
            log_name, _ = self.watching[source_id]
            self.reader.close_log(log_name)
            del self.watching[source_id]
            logger.info(f"Stopped watching source: {source_id}")
