# sentry/services/helix.py
"""
Helix Vector Service
====================

DNA encoding and anomaly detection for log analysis.

Helix Vector transforms raw logs into structured patterns using:
1. Drain3 clustering (pattern mining)
2. Markov chain transition probabilities
3. Severity-based anomaly detection

This service annotates LogChunks with cluster IDs and anomaly metadata,
enabling the RAG system to provide more structured context to the LLM.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from datetime import datetime

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from ..core.models import LogChunk
from ..core.config import settings

logger = logging.getLogger(__name__)


class HelixService:
    """
    Helix Vector DNA encoding and anomaly detection service.
    
    Uses config settings for all parameters (anomaly_threshold, severity levels, etc.)
    
    Usage:
        helix = HelixService()
        annotated_chunks = helix.annotate_chunks(chunks)
    """
    
    # Anomaly type classification keywords (static)
    ANOMALY_TYPES = {
        'database_timeout': ['database', 'db', 'sql', 'query', 'timeout'],
        'database_connection_error': ['database', 'db', 'connection'],
        'database_error': ['database', 'db', 'sql', 'mysql', 'postgres', 'mongo'],
        'timeout_error': ['timeout', 'timed out', 'deadline exceeded'],
        'auth_error': ['auth', 'login', 'permission', 'denied', 'unauthorized', 
                       'forbidden', '401', '403'],
        'memory_error': ['memory', 'oom', 'heap', 'allocation', 'out of memory'],
        'network_error': ['network', 'socket', 'connection refused', 
                          'unreachable', 'dns'],
        'io_error': ['disk', 'storage', 'write', 'read', 'i/o', 'filesystem'],
        'null_reference_error': ['null', 'undefined', 'nil', 'none', 'missing'],
        'server_error': ['500', '502', '503', '504', 'service unavailable'],
        'crash_error': ['terminated', 'killed', 'crashed', 'segfault'],
    }
    
    def __init__(self, anomaly_threshold: float = None):
        """
        Initialize the Helix service using config settings.
        
        Args:
            anomaly_threshold: Override config threshold (mainly for testing)
        """
        # Use config or override
        self.anomaly_threshold = anomaly_threshold or settings.helix_anomaly_threshold
        
        # Build severity levels from config
        self.severity_levels = {
            settings.helix_severity_fatal: [
                'fatal', 'panic', 'segfault', 'segmentation fault', 
                'core dump', 'kernel panic', 'system halt', 'oom killer'
            ],
            settings.helix_severity_critical: [
                'critical', 'crash', 'crashed', 'killed', 'terminated',
                'data loss', 'corruption', 'security breach', 'unauthorized'
            ],
            settings.helix_severity_severe: [
                'severe', 'service unavailable', '503', '502', 
                'connection refused', 'connection lost', 'unreachable'
            ],
            settings.helix_severity_error: [
                'error', 'exception', 'failed', 'failure', '500', 
                'timeout', 'timed out', 'refused', 'rejected'
            ],
            settings.helix_severity_warning: [
                'warn', 'warning', 'slow', 'retry', 'retrying',
                'degraded', 'high load', 'backpressure'
            ],
        }
        
        # Drain3 template miner with config params
        config = TemplateMinerConfig()
        config.drain_sim_th = settings.helix_drain_sim_th
        config.drain_depth = settings.helix_drain_depth
        self._miner = TemplateMiner(config=config)
        
        # State
        self._codebook: Dict[int, Dict[str, Any]] = {}
        self._transition_probs: Dict[int, Dict[int, float]] = {}
        self._fitted = False
        self._context_window = settings.helix_context_window
        
        logger.info("HelixService initialized (threshold=%.2f, window=%d)", 
                    self.anomaly_threshold, self._context_window)
    
    def reset(self) -> None:
        """Reset the service state for new indexing session."""
        config = TemplateMinerConfig()
        config.drain_sim_th = settings.helix_drain_sim_th
        config.drain_depth = settings.helix_drain_depth
        self._miner = TemplateMiner(config=config)
        self._codebook = {}
        self._transition_probs = {}
        self._fitted = False
        logger.debug("HelixService reset")
    
    def create_windowed_chunks(
        self, 
        logs: List[str], 
        source_id: str,
        metadata_base: Dict[str, Any] = None
    ) -> List[LogChunk]:
        """
        Create chunks with context stuffing - each chunk contains a window of logs.
        
        With window=4, each chunk contains:
        - 2 logs before the center
        - 1 center log (the focus)
        - 2 logs after the center
        
        This provides context for anomaly detection.
        
        Args:
            logs: List of raw log lines
            source_id: Source identifier
            metadata_base: Base metadata to include in all chunks
            
        Returns:
            List of LogChunk objects with windowed content
        """
        if not logs:
            return []
        
        from datetime import datetime
        
        chunks = []
        half_window = self._context_window // 2  # 2 for window=4
        
        # Create chunks with sliding window (step = half_window to avoid too much overlap)
        step = max(1, half_window)
        
        for center_idx in range(0, len(logs), step):
            # Calculate window boundaries
            start_idx = max(0, center_idx - half_window)
            end_idx = min(len(logs), center_idx + half_window + 1)
            
            # Get window content
            window_logs = logs[start_idx:end_idx]
            window_content = "\n".join(window_logs)
            
            # Create metadata
            metadata = dict(metadata_base) if metadata_base else {}
            metadata.update({
                "window_start": start_idx,
                "window_end": end_idx - 1,
                "center_index": center_idx,
                "window_size": len(window_logs)
            })
            
            # Create chunk
            chunk = LogChunk(
                source_id=source_id,
                content=window_content,
                timestamp=datetime.now(),
                log_level=self._detect_log_level(window_content),
                metadata=metadata
            )
            chunks.append(chunk)
        
        logger.info("Created %d windowed chunks from %d logs (window=%d, step=%d)",
                    len(chunks), len(logs), self._context_window, step)
        
        return chunks
    
    def _detect_log_level(self, content: str) -> 'LogLevel':
        """Detect log level from content."""
        from ..core.models import LogLevel
        
        c = content.upper()
        if 'FATAL' in c or 'CRITICAL' in c:
            return LogLevel.CRITICAL
        elif 'ERROR' in c or 'FAIL' in c:
            return LogLevel.ERROR
        elif 'WARN' in c:
            return LogLevel.WARNING
        elif 'DEBUG' in c:
            return LogLevel.DEBUG
        else:
            return LogLevel.INFO
    
    def encode_logs(self, logs: List[str]) -> List[int]:
        """
        Encode raw log lines into cluster IDs using Drain3.
        
        Args:
            logs: List of raw log strings
            
        Returns:
            List of cluster IDs (same length as logs)
        """
        cluster_ids = []
        for log in logs:
            result = self._miner.add_log_message(log)
            # Drain3 returns a dictionary, not an object
            cluster_ids.append(result["cluster_id"])
        
        # Build codebook
        self._codebook = {}
        for cluster in self._miner.drain.clusters:
            self._codebook[cluster.cluster_id] = {
                "template": cluster.get_template(),
                "count": cluster.size,
            }
        
        return cluster_ids
    
    def fit(self, cluster_ids: List[int]) -> None:
        """
        Learn transition probabilities from cluster sequence.
        
        Args:
            cluster_ids: Sequence of cluster IDs from encode_logs()
        """
        if len(cluster_ids) < 2:
            logger.warning("Need at least 2 cluster IDs to compute transitions")
            return
        
        # Count transitions
        transition_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        for i in range(len(cluster_ids) - 1):
            from_c = cluster_ids[i]
            to_c = cluster_ids[i + 1]
            transition_counts[from_c][to_c] += 1
        
        # Convert to probabilities
        self._transition_probs = {}
        for from_c, to_counts in transition_counts.items():
            total = sum(to_counts.values())
            self._transition_probs[from_c] = {
                to_c: count / total
                for to_c, count in to_counts.items()
            }
        
        self._fitted = True
        logger.debug("Fitted transition matrix on %d transitions", len(cluster_ids) - 1)
    
    def _get_severity_penalty(self, template: str) -> float:
        """Calculate severity penalty from template keywords."""
        if not template:
            return 0.0
        
        t = template.lower()
        
        for penalty, keywords in sorted(self.severity_levels.items(), reverse=True):
            if any(kw in t for kw in keywords):
                return penalty
        
        return 0.0
    
    def _classify_anomaly_type(self, template: str) -> str:
        """Derive anomaly type from template keywords."""
        if not template:
            return "unknown"
        
        t = template.lower()
        
        # Check for error indicators first
        error_keywords = ['error', 'exception', 'failed', 'fatal', 'critical', 'crash']
        is_error = any(kw in t for kw in error_keywords)
        
        if not is_error:
            warning_keywords = ['warning', 'warn', 'timeout', 'slow', 'retry']
            is_warning = any(kw in t for kw in warning_keywords)
            if not is_warning:
                return "rare_event"
        
        # Classify by domain (order matters - more specific first)
        for anomaly_type, keywords in self.ANOMALY_TYPES.items():
            if all(kw in t for kw in keywords[:2]):  # Match first 2 keywords
                return anomaly_type
            if any(kw in t for kw in keywords):
                return anomaly_type
        
        return "general_error"
    
    def _get_transition_probability(self, from_cluster: int, to_cluster: int) -> float:
        """Get probability of transition from one cluster to another."""
        if from_cluster not in self._transition_probs:
            return 0.0
        return self._transition_probs[from_cluster].get(to_cluster, 0.0)
    
    def annotate_chunks(self, chunks: List[LogChunk]) -> List[LogChunk]:
        """
        Annotate LogChunks with Helix Vector metadata.
        
        This is the main entry point. It:
        1. Encodes all chunk content into cluster IDs
        2. Fits the transition probability model
        3. Detects anomalies using severity-weighted probabilities
        4. Updates each chunk with cluster_id, anomaly fields, etc.
        
        Args:
            chunks: List of LogChunk objects to annotate
            
        Returns:
            Same list of chunks, now with Helix fields populated
        """
        if not chunks:
            return chunks
        
        # Extract log content
        logs = [c.content for c in chunks]
        
        # Encode logs to clusters
        cluster_ids = self.encode_logs(logs)
        
        # Fit transition model
        self.fit(cluster_ids)
        
        # Annotate each chunk
        for i, chunk in enumerate(chunks):
            cluster_id = cluster_ids[i]
            template = self._codebook.get(cluster_id, {}).get("template", "")
            
            # Basic cluster info
            chunk.cluster_id = cluster_id
            chunk.cluster_template = template
            
            # Calculate anomaly metrics
            if i > 0 and self._fitted:
                from_c = cluster_ids[i - 1]
                prob = self._get_transition_probability(from_c, cluster_id)
                severity = self._get_severity_penalty(template)
                effective_prob = prob * (1.0 - severity)
                
                chunk.transition_prob = prob
                chunk.severity_weight = severity
                
                # Determine if anomalous
                if effective_prob < self.anomaly_threshold:
                    chunk.is_anomaly = True
                    chunk.anomaly_type = self._classify_anomaly_type(template)
                    chunk.anomaly_score = 1.0 - effective_prob
                else:
                    chunk.is_anomaly = False
                    chunk.anomaly_type = None
                    chunk.anomaly_score = 0.0
            else:
                # First chunk or not fitted
                chunk.transition_prob = None
                chunk.severity_weight = self._get_severity_penalty(template)
                chunk.is_anomaly = False
                chunk.anomaly_score = 0.0
        
        # Log summary
        anomaly_count = sum(1 for c in chunks if c.is_anomaly)
        logger.info(
            "Annotated %d chunks: %d clusters, %d anomalies",
            len(chunks), len(self._codebook), anomaly_count
        )
        
        return chunks
    
    def get_codebook(self) -> Dict[int, Dict[str, Any]]:
        """Get the cluster ID to template mapping."""
        return self._codebook.copy()
    
    def get_transition_probs(self) -> Dict[int, Dict[int, float]]:
        """Get the transition probability matrix."""
        return self._transition_probs.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            "fitted": self._fitted,
            "cluster_count": len(self._codebook),
            "anomaly_threshold": self.anomaly_threshold,
        }
    
    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return f"<HelixService(clusters={len(self._codebook)}, {status})>"


# Singleton instance
_helix_service: Optional[HelixService] = None


def get_helix_service() -> HelixService:
    """
    Get the global Helix service instance.
    Lazy-loaded on first call.
    """
    global _helix_service
    if _helix_service is None:
        _helix_service = HelixService()
    return _helix_service
