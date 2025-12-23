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
import re
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
    
    # Anomaly type classification patterns (regex-based for better matching)
    # Order matters: more specific patterns should come first
    ANOMALY_PATTERNS = {
        # Database errors (most specific first)
        'database_timeout': [
            r'\b(database|db|sql|query|mysql|postgres|mongo|redis)\b.*\b(timeout|timed?\s*out)\b',
            r'\b(timeout|timed?\s*out)\b.*\b(database|db|sql|query)\b',
        ],
        'database_connection_error': [
            r'\b(database|db|sql|mysql|postgres|mongo|redis)\b.*\b(connection|connect)\b.*\b(fail|error|refused|lost|reset)\b',
            r'\bcannot connect to (database|db|mysql|postgres|mongo)\b',
            r'\b(database|db)\b.*\bunreachable\b',
        ],
        'database_error': [
            r'\b(database|db|sql|mysql|postgres|mongo|mariadb|oracle|sqlite|redis|cassandra|dynamodb)\b.*\b(error|fail|exception)\b',
            r'\b(query|insert|update|delete|select)\b.*\b(fail|error)\b',
            r'\bsqlalchemy\b.*\b(error|exception)\b',
            r'\bdeadlock\b',
            r'\bforeign key constraint\b',
        ],
        
        # HTTP/API errors
        'http_client_error': [
            r'\b4[0-9]{2}\b',  # 4xx status codes
            r'\b(400|bad request)\b',
            r'\b(401|unauthorized)\b',
            r'\b(403|forbidden)\b',
            r'\b(404|not found)\b',
            r'\b(405|method not allowed)\b',
            r'\b(429|too many requests|rate limit)\b',
        ],
        'http_server_error': [
            r'\b5[0-9]{2}\b',  # 5xx status codes
            r'\b(500|internal server error)\b',
            r'\b(502|bad gateway)\b',
            r'\b(503|service unavailable)\b',
            r'\b(504|gateway timeout)\b',
        ],
        
        # Authentication/Authorization
        'auth_error': [
            r'\b(auth|authentication|authorization)\b.*\b(fail|error|denied|invalid)\b',
            r'\b(login|signin|sign-in)\b.*\b(fail|error|invalid)\b',
            r'\b(permission|access)\s*(denied|forbidden)\b',
            r'\bunauthorized\s*(access|request)?\b',
            r'\b(invalid|expired|missing)\s*(token|session|credential|jwt|api.?key)\b',
            r'\b(password|credential)\b.*\b(incorrect|invalid|wrong|mismatch)\b',
        ],
        
        # Memory errors
        'memory_error': [
            r'\b(out of memory|oom|oom.?killer)\b',
            r'\b(memory|heap|stack)\s*(overflow|exhausted|exceeded|limit)\b',
            r'\bmemory\s*(allocation|alloc)\s*(fail|error)\b',
            r'\b(gc|garbage collector)\b.*\b(overhead|pressure|fail)\b',
            r'\bjava\.lang\.OutOfMemoryError\b',
        ],
        
        # Network errors
        'network_error': [
            r'\b(network|socket|tcp|udp|http)\b.*\b(error|fail|timeout|refused|reset)\b',
            r'\bconnection\s*(refused|reset|closed|timed?\s*out)\b',
            r'\b(host|server|endpoint)\b.*\bunreachable\b',
            r'\bdns\s*(resolution|lookup)?\s*(fail|error|timeout)\b',
            r'\bssl\b.*\b(error|handshake|certificate)\b',
            r'\beconnrefused|econnreset|etimedout|ehostunreach\b',
        ],
        
        # Timeout errors (generic)
        'timeout_error': [
            r'\b(request|response|operation|execution)\s*(timeout|timed?\s*out)\b',
            r'\bdeadline\s*(exceeded|expired)\b',
            r'\b(read|write|connect)\s*timeout\b',
            r'\bglobal timeout\b',
        ],
        
        # I/O and disk errors
        'io_error': [
            r'\b(disk|storage|volume|filesystem|file\s*system)\b.*\b(error|fail|full)\b',
            r'\b(read|write)\b.*\b(error|fail|io)\b',
            r'\b(i/o|io)\s*(error|exception)\b',
            r'\bno\s*space\s*left\b',
            r'\b(permission|access)\s*denied\b.*\b(file|directory|path)\b',
            r'\bfile\s*not\s*found\b',
        ],
        
        # Null/Reference errors
        'null_reference_error': [
            r'\b(null|nil|none|undefined)\s*(pointer|reference)?\s*(exception|error|access)?\b',
            r'\bNullPointerException\b',
            r'\bTypeError:.*None\b',
            r'\bAttributeError:.*NoneType\b',
            r'\bundefined is not (a function|an object)\b',
            r'\bcannot read propert(y|ies) of (null|undefined)\b',
        ],
        
        # Crash/Fatal errors
        'crash_error': [
            r'\b(fatal|panic|crash|crashed)\b',
            r'\b(segfault|segmentation fault|sigsegv|sigabrt|sigkill)\b',
            r'\b(terminated|killed|aborted)\b',
            r'\b(core dump|unhandled exception|uncaught error)\b',
            r'\bprocess\s*(exit|died|killed)\b',
        ],
        
        # Configuration errors
        'config_error': [
            r'\b(config|configuration|settings?)\b.*\b(error|invalid|missing|fail)\b',
            r'\b(environment|env)\s*variable\b.*\b(missing|not set|undefined)\b',
            r'\binvalid\s*(config|configuration|setting|parameter)\b',
        ],
        
        # Validation errors
        'validation_error': [
            r'\b(validation|validate|validator)\b.*\b(error|fail|invalid)\b',
            r'\binvalid\s*(input|data|format|type|value|argument|parameter)\b',
            r'\b(schema|json|xml)\b.*\b(validation|parse)\b.*\b(error|fail)\b',
            r'\btype\s*(error|mismatch)\b',
        ],
        
        # Queue/Message errors
        'queue_error': [
            r'\b(queue|message|kafka|rabbitmq|sqs|redis)\b.*\b(error|fail|timeout|overflow)\b',
            r'\b(consumer|producer|subscriber|publisher)\b.*\b(error|fail)\b',
            r'\bmessage\s*(lost|dropped|rejected)\b',
        ],
        
        # Rate limiting/Throttling
        'rate_limit_error': [
            r'\b(rate\s*limit|throttl|too many requests)\b',
            r'\b429\b.*\b(error|response)\b',
            r'\bquota\s*(exceeded|limit)\b',
        ],
        
        # Dependency/Service errors
        'dependency_error': [
            r'\b(service|microservice|api|upstream|downstream)\b.*\b(unavailable|error|fail|timeout)\b',
            r'\b(circuit\s*breaker|fallback)\b.*\b(open|triggered|activated)\b',
            r'\bdependency\b.*\b(fail|error|unavailable)\b',
        ],
    }
    
    # Legacy keyword-based fallback (used when no regex matches)
    ANOMALY_KEYWORDS = {
        'database_error': ['database', 'db', 'sql', 'mysql', 'postgres', 'mongo'],
        'timeout_error': ['timeout', 'timed out', 'deadline exceeded'],
        'auth_error': ['auth', 'login', 'permission', 'denied', 'unauthorized', 'forbidden'],
        'memory_error': ['memory', 'oom', 'heap', 'allocation', 'out of memory'],
        'network_error': ['network', 'socket', 'connection refused', 'unreachable', 'dns'],
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
        """
        Derive anomaly type from template using regex patterns.
        
        Uses a two-phase approach:
        1. Try regex patterns first (more accurate)
        2. Fall back to keyword matching if no regex matches
        
        Returns the most specific anomaly type that matches.
        """
        if not template:
            return "unknown"
        
        t = template.lower()
        
        # Phase 1: Try regex patterns (more accurate matching)
        for anomaly_type, patterns in self.ANOMALY_PATTERNS.items():
            for pattern in patterns:
                try:
                    if re.search(pattern, t, re.IGNORECASE):
                        return anomaly_type
                except re.error:
                    # Skip invalid patterns
                    continue
        
        # Phase 2: Check for general error/warning indicators
        error_indicators = [
            r'\b(error|err|exception|fail|failed|failure)\b',
            r'\b(fatal|critical|severe|panic)\b',
            r'\b(crash|crashed|abort|aborted)\b',
        ]
        is_error = any(re.search(p, t, re.IGNORECASE) for p in error_indicators)
        
        warning_indicators = [
            r'\b(warn|warning)\b',
            r'\b(timeout|timed?\s*out)\b',
            r'\b(slow|degraded|retry|retrying)\b',
        ]
        is_warning = any(re.search(p, t, re.IGNORECASE) for p in warning_indicators)
        
        if not is_error and not is_warning:
            # Check for HTTP status codes that indicate issues
            if re.search(r'\b[45][0-9]{2}\b', t):
                return "http_error"
            return "rare_event"
        
        # Phase 3: Keyword-based fallback
        for anomaly_type, keywords in self.ANOMALY_KEYWORDS.items():
            if any(kw in t for kw in keywords):
                return anomaly_type
        
        return "general_error" if is_error else "warning"
    
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
    
    # ========== QUERY METHODS FOR INTENT-BASED ROUTING ==========
    
    def get_cluster_statistics(self) -> Dict[str, Any]:
        """
        Get detailed cluster statistics for FREQUENCY intent queries.
        
        Returns structured data about:
        - Total clusters and their occurrence counts
        - Most common patterns
        - Transition probabilities summary
        
        This is used by RAG when routing FREQUENCY queries.
        """
        if not self._codebook:
            return {
                "total_clusters": 0,
                "clusters": [],
                "top_patterns": [],
                "total_logs_processed": 0,
                "has_transitions": False
            }
        
        # Build cluster info with counts
        clusters = []
        for cluster_id, info in self._codebook.items():
            template = info.get("template", "")
            count = info.get("count", 0)
            clusters.append({
                "id": cluster_id,
                "template": template,
                "count": count,
                "is_error": any(kw in template.upper() for kw in ["ERROR", "FAIL", "CRITICAL", "FATAL"]),
                "is_warning": "WARN" in template.upper()
            })
        
        # Sort by count descending
        clusters.sort(key=lambda x: x["count"], reverse=True)
        
        # Top 10 patterns
        top_patterns = clusters[:10]
        
        # Count by type
        error_count = sum(c["count"] for c in clusters if c["is_error"])
        warning_count = sum(c["count"] for c in clusters if c["is_warning"])
        info_count = sum(c["count"] for c in clusters) - error_count - warning_count
        
        return {
            "total_clusters": len(self._codebook),
            "total_logs_processed": sum(c["count"] for c in clusters),
            "clusters": clusters,
            "top_patterns": top_patterns,
            "has_transitions": bool(self._transition_probs),
            "error_count": error_count,
            "warning_count": warning_count,
            "info_count": info_count,
            "error_percentage": error_count / max(sum(c["count"] for c in clusters), 1) * 100
        }
    
    def get_anomalies_summary(self, chunks: List[LogChunk] = None) -> Dict[str, Any]:
        """
        Get anomaly summary for ERROR/ANOMALY intent queries.
        
        Args:
            chunks: Optional list of chunks to analyze. If None, returns 
                    template-level anomaly indicators.
        
        Returns structured anomaly data for LLM to explain.
        """
        # Get clusters that are likely error-related
        error_clusters = []
        for cluster_id, info in self._codebook.items():
            template = info.get("template", "")
            severity = self._get_severity_penalty(template)
            if severity > 0.3:  # Moderate to high severity
                anomaly_type = self._classify_anomaly_type(template)
                error_clusters.append({
                    "cluster_id": cluster_id,
                    "template": template,
                    "severity": severity,
                    "type": anomaly_type,
                    "count": info.get("count", 0)
                })
        
        # Sort by severity
        error_clusters.sort(key=lambda x: x["severity"], reverse=True)
        
        return {
            "total_error_clusters": len(error_clusters),
            "error_clusters": error_clusters[:10],  # Top 10
            "severity_distribution": {
                "critical": sum(1 for c in error_clusters if c["severity"] > 0.7),
                "high": sum(1 for c in error_clusters if 0.5 < c["severity"] <= 0.7),
                "moderate": sum(1 for c in error_clusters if 0.3 < c["severity"] <= 0.5),
            },
            "anomaly_threshold": self.anomaly_threshold
        }
    
    def search_clusters_by_pattern(self, pattern: str) -> List[Dict[str, Any]]:
        """
        Search clusters by pattern match for SIMILAR intent queries.
        
        Args:
            pattern: Search string (case-insensitive)
            
        Returns:
            List of matching clusters with their info
        """
        if not pattern:
            return []
        
        pattern_lower = pattern.lower()
        matches = []
        
        for cluster_id, info in self._codebook.items():
            template = info.get("template", "")
            if pattern_lower in template.lower():
                matches.append({
                    "cluster_id": cluster_id,
                    "template": template,
                    "count": info.get("count", 0),
                    "severity": self._get_severity_penalty(template)
                })
        
        # Sort by count (frequency)
        matches.sort(key=lambda x: x["count"], reverse=True)
        return matches
    
    def get_transition_context(self, cluster_id: int) -> Dict[str, Any]:
        """
        Get transition context for a specific cluster (for WHY queries).
        
        Shows what typically happens before and after this cluster.
        """
        if cluster_id not in self._codebook:
            return {"error": f"Cluster {cluster_id} not found"}
        
        # What leads TO this cluster
        incoming = []
        for from_c, transitions in self._transition_probs.items():
            if cluster_id in transitions:
                prob = transitions[cluster_id]
                from_template = self._codebook.get(from_c, {}).get("template", "")
                incoming.append({
                    "from_cluster": from_c,
                    "probability": prob,
                    "template": from_template
                })
        
        # What this cluster leads TO
        outgoing = []
        if cluster_id in self._transition_probs:
            for to_c, prob in self._transition_probs[cluster_id].items():
                to_template = self._codebook.get(to_c, {}).get("template", "")
                outgoing.append({
                    "to_cluster": to_c,
                    "probability": prob,
                    "template": to_template
                })
        
        # Sort by probability
        incoming.sort(key=lambda x: x["probability"], reverse=True)
        outgoing.sort(key=lambda x: x["probability"], reverse=True)
        
        return {
            "cluster_id": cluster_id,
            "template": self._codebook[cluster_id].get("template", ""),
            "count": self._codebook[cluster_id].get("count", 0),
            "incoming_transitions": incoming[:5],  # Top 5
            "outgoing_transitions": outgoing[:5]   # Top 5
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
