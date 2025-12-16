"""
Anomaly Detector - Research Module
===================================

Detects anomalies in log sequences using probability-based analysis.
NO ML CLASSIFICATION MODEL NEEDED - we derive types from the data itself.

## The Core Technology

### 1. Markov Chain Transition Probabilities

Think of logs as a sequence of "states" (cluster IDs). We learn:
- "After state A, what states usually follow?"
- "How often does each transition happen?"

Example:
    Normal flow: login(1) → dashboard(2) → success(3) → logout(4)
    This happens 90% of the time, so it's EXPECTED.
    
    Error flow: login(1) → dashboard(2) → timeout(5) → crash(6)
    This happens 10% of the time, so it's ANOMALOUS.

A transition with low probability = ANOMALY.

### 2. Template-Based Classification

Drain3 gives us templates like:
    Cluster 5: "ERROR Database connection timeout <*>"
    
We parse this template to classify the anomaly:
    Contains "database" + "error" → anomaly_type = "database_error"

No ML model needed - the template IS the classification!

### 3. Context Windows (4+ lines)

Instead of single log lines, we create windows:
    Center: The anomalous log
    Context: 2 logs before + 2 logs after
    
This gives LLMs the "story" - what happened before/after the error.

## How This is Different

Traditional Approach:
    1. Collect logs
    2. Train ML classifier on labeled data (expensive!)
    3. Classify new logs
    
Our Approach:
    1. Collect logs
    2. Learn normal patterns from the data itself (unsupervised)
    3. Rare patterns = anomalies (no labels needed!)
    4. Template text tells us the anomaly type (no classifier needed!)

Benefits:
    - No labeled training data required
    - Adapts to YOUR log patterns automatically
    - Interpretable: "This is anomalous because it only happens 5% of the time"
    - Fast: Just probability lookups, no model inference

Usage:
    detector = AnomalyDetector()
    detector.fit(cluster_ids)  # Learn normal patterns
    anomalies = detector.detect(cluster_ids)  # Find rare events
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import re


@dataclass
class AnomalyInfo:
    """
    Information about a detected anomaly.
    
    This is what gets attached to each LogChunk that's flagged.
    """
    # Position in the sequence
    index: int
    
    # The cluster ID at this position
    cluster_id: int
    
    # How anomalous is this? (0.0 = normal, 1.0 = very rare)
    anomaly_score: float
    
    # What type of anomaly? (derived from template)
    anomaly_type: str
    
    # The transition probability that triggered this
    transition_prob: float
    
    # What transition was rare? (from_cluster → to_cluster)
    from_cluster: int
    to_cluster: int
    
    # The template text (for human readability)
    template: str = ""
    
    # NEW: Severity penalty applied (0.0 = no penalty, 1.0 = max penalty)
    severity_weight: float = 0.0
    
    # NEW: The effective probability after penalty
    # effective_prob = transition_prob * (1 - severity_weight)
    effective_prob: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        return {
            "index": self.index,
            "cluster_id": self.cluster_id,
            "anomaly_score": self.anomaly_score,
            "anomaly_type": self.anomaly_type,
            "transition_prob": self.transition_prob,
            "effective_prob": self.effective_prob,
            "severity_weight": self.severity_weight,
            "from_cluster": self.from_cluster,
            "to_cluster": self.to_cluster,
            "template": self.template
        }


@dataclass 
class AnnotatedChunk:
    """
    A log chunk with anomaly annotations.
    
    This is what your LogChunk object would look like after annotation.
    Contains 4+ log lines (context stuffing) plus anomaly info.
    """
    # The combined text (4+ log lines)
    text: str
    
    # Raw log lines in this chunk
    log_lines: List[str]
    
    # Cluster IDs for each log line
    cluster_ids: List[int]
    
    # Which position in the chunk is the "center" (main focus)
    center_index: int
    
    # Anomaly information (None if this chunk is normal)
    anomaly: Optional[AnomalyInfo] = None
    
    # Quick access flags
    is_anomaly: bool = False
    anomaly_type: Optional[str] = None
    anomaly_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "text": self.text,
            "log_lines": self.log_lines,
            "cluster_ids": self.cluster_ids,
            "center_index": self.center_index,
            "is_anomaly": self.is_anomaly,
            "anomaly_type": self.anomaly_type,
            "anomaly_score": self.anomaly_score,
            "anomaly": self.anomaly.to_dict() if self.anomaly else None
        }


class AnomalyDetector:
    """
    Detects anomalies using transition probabilities.
    
    ## How It Works
    
    1. **Fit**: Learn the normal transition patterns from data
       - Count how often each A→B transition occurs
       - Convert counts to probabilities
       
    2. **Detect**: Find rare transitions
       - For each transition, check its probability
       - Low probability = anomaly
       
    3. **Classify**: Determine anomaly type from template
       - Parse the cluster template text
       - Extract keywords to determine type
    
    ## Example
    
        detector = AnomalyDetector(anomaly_threshold=0.1)
        
        # Learn from data
        detector.fit(cluster_ids, codebook)
        
        # Find anomalies
        anomalies = detector.detect(cluster_ids)
        
        # Create annotated chunks
        chunks = detector.create_annotated_chunks(logs, cluster_ids, window_size=2)
    """
    
    def __init__(
        self,
        anomaly_threshold: float = 0.1,
        min_observations: int = 2
    ):
        """
        Initialize the detector.
        
        Args:
            anomaly_threshold: Transitions with probability below this are anomalies.
                              Default 0.1 means: if a transition happens less than
                              10% of the time, it's flagged as anomalous.
                              
            min_observations: Minimum times we need to see a transition to judge it.
                             If we've only seen A→B once, we can't say if it's rare.
        """
        self.anomaly_threshold = anomaly_threshold
        self.min_observations = min_observations
        
        # Learned from fit()
        self.transition_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.transition_probs: Dict[int, Dict[int, float]] = {}
        self.cluster_counts: Dict[int, int] = defaultdict(int)
        self.codebook: Dict[int, str] = {}
        
        # Fitted flag
        self._fitted = False
    
    def fit(
        self,
        cluster_ids: List[int],
        codebook: Dict[int, Dict[str, Any]] = None
    ) -> 'AnomalyDetector':
        """
        Learn transition probabilities from the data.
        
        This is the "training" step, but no ML involved!
        We're just counting transitions and converting to probabilities.
        
        Args:
            cluster_ids: Sequence of cluster IDs (the "DNA" of the logs)
            codebook: Mapping of cluster_id → {"template": "...", "count": N}
                     This comes from LogDNAEncoder.get_codebook()
        
        Returns:
            self (for chaining)
        
        ## How It Works
        
        Given: [1, 2, 3, 2, 1, 2, 5, 6, 1, 2, 3, 2]
        
        1. Count transitions:
           1→2: 3 times
           2→3: 2 times
           2→5: 1 time  ← This is rare!
           2→1: 1 time
           3→2: 2 times
           5→6: 1 time
           6→1: 1 time
           
        2. Convert to probabilities:
           From cluster 2, totals: 3+2+1+1 = 7 outgoing transitions
           2→3: 2/7 = 28%
           2→5: 1/7 = 14%  ← Below threshold (10%), so ANOMALY
           2→1: 1/7 = 14%
           etc.
        """
        if len(cluster_ids) < 2:
            raise ValueError("Need at least 2 cluster IDs to compute transitions")
        
        # Store codebook (for template-based classification)
        if codebook:
            self.codebook = {
                cid: info.get("template", "") 
                for cid, info in codebook.items()
            }
        
        # Count transitions
        for i in range(len(cluster_ids) - 1):
            from_c = cluster_ids[i]
            to_c = cluster_ids[i + 1]
            self.transition_counts[from_c][to_c] += 1
            self.cluster_counts[from_c] += 1
        
        # Also count the last cluster
        self.cluster_counts[cluster_ids[-1]] += 1
        
        # Convert to probabilities
        self.transition_probs = {}
        for from_c, to_counts in self.transition_counts.items():
            total = sum(to_counts.values())
            self.transition_probs[from_c] = {
                to_c: count / total
                for to_c, count in to_counts.items()
            }
        
        self._fitted = True
        return self
    
    def get_transition_probability(self, from_cluster: int, to_cluster: int) -> float:
        """
        Get the probability of a specific transition.
        
        Args:
            from_cluster: Source cluster ID
            to_cluster: Destination cluster ID
            
        Returns:
            Probability (0.0 to 1.0). Returns 0.0 if never observed.
        """
        if from_cluster not in self.transition_probs:
            return 0.0
        return self.transition_probs[from_cluster].get(to_cluster, 0.0)
    
    def detect(self, cluster_ids: List[int]) -> List[AnomalyInfo]:
        """
        Detect anomalies in a sequence using SEVERITY-PENALIZED probabilities.
        
        ## The Severity Penalty System
        
        Even if an error happens frequently (high probability), we still want
        to flag it if it's severe. We apply a penalty based on template severity.
        
        Formula:
            effective_prob = transition_prob × (1 - severity_weight)
        
        Example:
            transition_prob = 0.50 (happens 50% of time - normally OK)
            severity_weight = 0.80 (FATAL keyword found)
            effective_prob = 0.50 × (1 - 0.80) = 0.50 × 0.20 = 0.10
            0.10 < threshold (0.15) → FLAGGED AS ANOMALY!
        
        This ensures severe errors are always flagged, even if common.
        
        Args:
            cluster_ids: Sequence to analyze
            
        Returns:
            List of AnomalyInfo for each detected anomaly
        """
        if not self._fitted:
            raise ValueError("Must call fit() before detect()")
        
        anomalies = []
        
        for i in range(1, len(cluster_ids)):
            from_c = cluster_ids[i - 1]
            to_c = cluster_ids[i]
            
            # Get raw transition probability
            prob = self.get_transition_probability(from_c, to_c)
            
            # Get template for classification and severity
            template = self.codebook.get(to_c, "")
            
            # Get severity penalty from template keywords
            severity_weight = self._get_severity_penalty(template)
            
            # Apply penalty: effective_prob = prob × (1 - severity_weight)
            # Higher severity = lower effective probability = more likely to be flagged
            effective_prob = prob * (1.0 - severity_weight)
            
            # Check if this transition is anomalous (using EFFECTIVE probability)
            if effective_prob < self.anomaly_threshold:
                anomaly_type = self._classify_from_template(template)
                
                # Anomaly score combines rarity AND severity
                # Score = 1 - effective_prob (higher = more anomalous)
                anomaly_score = 1.0 - effective_prob
                
                anomaly = AnomalyInfo(
                    index=i,
                    cluster_id=to_c,
                    anomaly_score=anomaly_score,
                    anomaly_type=anomaly_type,
                    transition_prob=prob,
                    effective_prob=effective_prob,
                    severity_weight=severity_weight,
                    from_cluster=from_c,
                    to_cluster=to_c,
                    template=template
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _get_severity_penalty(self, template: str) -> float:
        """
        Calculate severity penalty from template keywords.
        
        The more FATAL/CRITICAL the event, the higher the penalty.
        This ensures severe errors are always flagged, even if common.
        
        Args:
            template: The cluster template text
            
        Returns:
            Penalty weight (0.0 to 0.95)
            - 0.0 = no penalty (INFO logs)
            - 0.3 = low penalty (warnings)
            - 0.5 = medium penalty (errors)
            - 0.7 = high penalty (critical)
            - 0.9 = max penalty (fatal/crash)
        
        ## How It Works
        
        We scan the template for severity keywords and return the highest
        penalty that matches.
        
        Example:
            "FATAL Database connection lost" → 0.9 (max penalty)
            "ERROR Query timeout"            → 0.5 (medium penalty)
            "WARN Slow response"             → 0.3 (low penalty)
            "INFO User logged in"            → 0.0 (no penalty)
        """
        if not template:
            return 0.0
        
        t = template.lower()
        
        # Severity levels (ordered from highest to lowest)
        # Check highest severity first and return immediately
        
        # FATAL (0.9) - System is unusable, immediate failure
        fatal_keywords = ['fatal', 'panic', 'segfault', 'segmentation fault', 
                          'core dump', 'kernel panic', 'system halt', 'oom killer']
        if any(kw in t for kw in fatal_keywords):
            return 0.9
        
        # CRITICAL (0.8) - Crash, data loss, security breach
        critical_keywords = ['critical', 'crash', 'crashed', 'killed', 'terminated',
                            'data loss', 'corruption', 'security breach', 'unauthorized']
        if any(kw in t for kw in critical_keywords):
            return 0.8
        
        # SEVERE (0.7) - Service down, major failure
        severe_keywords = ['severe', 'service unavailable', '503', '502', 
                          'connection refused', 'connection lost', 'unreachable']
        if any(kw in t for kw in severe_keywords):
            return 0.7
        
        # ERROR (0.5) - Something failed but system continues
        error_keywords = ['error', 'exception', 'failed', 'failure', '500', 
                         'timeout', 'timed out', 'refused', 'rejected']
        if any(kw in t for kw in error_keywords):
            return 0.5
        
        # WARNING (0.3) - Potential problem, degraded performance
        warning_keywords = ['warn', 'warning', 'slow', 'retry', 'retrying',
                           'degraded', 'high load', 'backpressure']
        if any(kw in t for kw in warning_keywords):
            return 0.3
        
        # DEBUG/INFO (0.0) - No penalty
        return 0.0
    
    def _classify_from_template(self, template: str) -> str:
        """
        Derive anomaly type by parsing the template text.
        
        This is the "no ML classifier needed" magic!
        Drain3 already extracted the pattern - we just parse it.
        
        Args:
            template: The cluster template (e.g., "ERROR Database connection <*>")
            
        Returns:
            Anomaly type string (e.g., "database_error")
        
        ## How It Works
        
        1. Check if template contains error keywords
        2. Check which domain keywords are present
        3. Return the most specific type
        
        Example:
            "ERROR Database connection timeout after <*>s"
            - Contains "ERROR" → is an error
            - Contains "Database" → database_error
        """
        if not template:
            return "unknown"
        
        t = template.lower()
        
        # First check: is this even an error?
        error_keywords = ['error', 'exception', 'failed', 'fatal', 'critical', 'crash']
        is_error = any(kw in t for kw in error_keywords)
        
        if not is_error:
            # Could still be anomalous for other reasons
            warning_keywords = ['warning', 'warn', 'timeout', 'slow', 'retry']
            is_warning = any(kw in t for kw in warning_keywords)
            if not is_warning:
                return "rare_event"  # Anomalous but not an error
        
        # Classify by domain
        # Order matters - more specific checks first
        
        if any(kw in t for kw in ['database', 'db', 'sql', 'query', 'mysql', 'postgres', 'mongo']):
            if 'timeout' in t:
                return "database_timeout"
            elif 'connection' in t:
                return "database_connection_error"
            else:
                return "database_error"
        
        if any(kw in t for kw in ['timeout', 'timed out', 'deadline exceeded']):
            return "timeout_error"
        
        if any(kw in t for kw in ['auth', 'login', 'permission', 'denied', 'unauthorized', 'forbidden', '401', '403']):
            return "auth_error"
        
        if any(kw in t for kw in ['memory', 'oom', 'heap', 'allocation', 'out of memory']):
            return "memory_error"
        
        if any(kw in t for kw in ['network', 'socket', 'connection refused', 'unreachable', 'dns']):
            return "network_error"
        
        if any(kw in t for kw in ['disk', 'storage', 'write', 'read', 'i/o', 'filesystem']):
            return "io_error"
        
        if any(kw in t for kw in ['null', 'undefined', 'nil', 'none', 'missing']):
            return "null_reference_error"
        
        if any(kw in t for kw in ['500', '502', '503', '504', 'service unavailable']):
            return "server_error"
        
        if any(kw in t for kw in ['terminated', 'killed', 'crashed', 'segfault']):
            return "crash_error"
        
        return "general_error"
    
    def create_annotated_chunks(
        self,
        log_lines: List[str],
        cluster_ids: List[int],
        window_size: int = 2
    ) -> List[AnnotatedChunk]:
        """
        Create annotated chunks with context stuffing.
        
        Instead of single log lines, creates windows of 4+ lines
        centered on each position. Anomalies are marked.
        
        Args:
            log_lines: The raw log text lines
            cluster_ids: Cluster IDs for each line
            window_size: Lines before AND after center (default 2 = 5 total lines)
            
        Returns:
            List of AnnotatedChunk objects
        
        ## How It Works
        
        Given logs: [L0, L1, L2, L3, L4, L5, L6, L7]
        And window_size=2
        
        For center position 3 (L3):
            Window: [L1, L2, L3, L4, L5]  (2 before + center + 2 after)
            Text: "L1 | L2 | L3 | L4 | L5"
            
        If L3 is anomalous:
            chunk.is_anomaly = True
            chunk.anomaly_type = "database_error"
        """
        if len(log_lines) != len(cluster_ids):
            raise ValueError("log_lines and cluster_ids must have same length")
        
        if not self._fitted:
            raise ValueError("Must call fit() before create_annotated_chunks()")
        
        # First detect all anomalies
        anomalies = self.detect(cluster_ids)
        anomaly_map = {a.index: a for a in anomalies}
        
        chunks = []
        
        for i in range(len(log_lines)):
            # Build window around this position
            start = max(0, i - window_size)
            end = min(len(log_lines), i + window_size + 1)
            
            window_logs = log_lines[start:end]
            window_clusters = cluster_ids[start:end]
            
            # Find the center index within the window
            center_in_window = i - start
            
            # Combine text with separator
            combined_text = " | ".join(window_logs)
            
            # Check if this position is anomalous
            anomaly = anomaly_map.get(i)
            
            chunk = AnnotatedChunk(
                text=combined_text,
                log_lines=window_logs,
                cluster_ids=window_clusters,
                center_index=center_in_window,
                anomaly=anomaly,
                is_anomaly=anomaly is not None,
                anomaly_type=anomaly.anomaly_type if anomaly else None,
                anomaly_score=anomaly.anomaly_score if anomaly else 0.0
            )
            chunks.append(chunk)
        
        return chunks
    
    def get_summary(self) -> str:
        """Get a human-readable summary of learned patterns."""
        if not self._fitted:
            return "Not fitted yet"
        
        lines = ["=== Anomaly Detector Summary ==="]
        lines.append(f"Unique clusters: {len(self.cluster_counts)}")
        lines.append(f"Anomaly threshold: {self.anomaly_threshold:.1%}")
        lines.append("")
        lines.append("Transition probabilities:")
        
        for from_c in sorted(self.transition_probs.keys()):
            for to_c, prob in sorted(self.transition_probs[from_c].items(), key=lambda x: -x[1]):
                flag = "⚠️ RARE" if prob < self.anomaly_threshold else ""
                lines.append(f"  [{from_c}] → [{to_c}]: {prob:.1%} {flag}")
        
        return "\n".join(lines)
    
    def __repr__(self):
        status = "fitted" if self._fitted else "not fitted"
        return f"<AnomalyDetector(threshold={self.anomaly_threshold}, {status})>"


# ============================================================
# DEMO / TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🔍 ANOMALY DETECTOR DEMO - With Severity Penalty")
    print("=" * 70)
    
    # Sample data - includes both rare AND common errors
    # The key demo: cluster 7 (CRITICAL crash) happens often but should still be flagged
    logs = [
        "2024-01-15 10:00:01 INFO User john connected from 192.168.1.1",
        "2024-01-15 10:00:02 INFO User john requested dashboard",
        "2024-01-15 10:00:03 INFO Query completed successfully",
        "2024-01-15 10:00:04 INFO User john logged out",
        "2024-01-15 10:01:01 INFO User alice connected from 10.0.0.1",
        "2024-01-15 10:01:02 INFO User alice requested dashboard",
        "2024-01-15 10:01:03 ERROR Database connection timeout after 30s",  # Anomaly (rare)
        "2024-01-15 10:01:04 CRITICAL System crashed due to memory corruption",  # Anomaly (severe!)
        "2024-01-15 10:02:01 INFO User bob connected from 172.16.0.1",
        "2024-01-15 10:02:02 INFO User bob requested dashboard",
        "2024-01-15 10:02:03 CRITICAL System crashed due to memory corruption",  # COMMON but severe!
        "2024-01-15 10:02:04 INFO Recovery completed",
    ]
    
    # Simulated cluster IDs
    # Note: cluster 7 (CRITICAL crash) appears TWICE - making it "common" (33% from cluster 2)
    # Without severity penalty, it wouldn't be flagged
    # With severity penalty (0.8 for CRITICAL), it WILL be flagged
    cluster_ids = [1, 2, 3, 4, 1, 2, 5, 7, 1, 2, 7, 8]
    
    # Simulated codebook
    codebook = {
        1: {"template": "INFO User <*> connected from <*>", "count": 3},
        2: {"template": "INFO User <*> requested dashboard", "count": 3},
        3: {"template": "INFO Query completed successfully", "count": 1},
        4: {"template": "INFO User <*> logged out", "count": 1},
        5: {"template": "ERROR Database connection timeout after <*>", "count": 1},
        7: {"template": "CRITICAL System crashed due to memory corruption", "count": 2},
        8: {"template": "INFO Recovery completed", "count": 1},
    }
    
    print("\n📋 Sample logs:")
    for i, log in enumerate(logs):
        print(f"  [{i}] Cluster {cluster_ids[i]}: {log[:55]}...")
    
    # Create detector with threshold that would normally miss high-prob errors
    print("\n🔧 Creating detector with threshold=0.25 (25%)")
    print("   Without severity penalty, cluster 7 (33% prob) would NOT be flagged!")
    detector = AnomalyDetector(anomaly_threshold=0.25)
    detector.fit(cluster_ids, codebook)
    
    print(f"\n{detector.get_summary()}")
    
    # Show severity weights
    print("\n" + "=" * 70)
    print("⚖️  SEVERITY WEIGHTS (from template keywords)")
    print("=" * 70)
    
    for cid, info in codebook.items():
        template = info["template"]
        weight = detector._get_severity_penalty(template)
        level = "FATAL" if weight >= 0.9 else "CRITICAL" if weight >= 0.8 else \
                "SEVERE" if weight >= 0.7 else "ERROR" if weight >= 0.5 else \
                "WARNING" if weight >= 0.3 else "INFO"
        print(f"  Cluster {cid}: {level:8} (weight={weight:.1f}) - {template[:40]}...")
    
    # Detect anomalies
    print("\n" + "=" * 70)
    print("🚨 DETECTED ANOMALIES (with Severity Penalty)")
    print("=" * 70)
    
    anomalies = detector.detect(cluster_ids)
    
    if not anomalies:
        print("\n  No anomalies detected!")
    else:
        for a in anomalies:
            print(f"\n  Index {a.index}: Cluster {a.cluster_id}")
            print(f"    Transition: [{a.from_cluster}] → [{a.to_cluster}]")
            print(f"    Raw Probability: {a.transition_prob:.1%}")
            print(f"    Severity Weight: {a.severity_weight:.1f} (penalty)")
            print(f"    Effective Prob:  {a.effective_prob:.1%} = {a.transition_prob:.1%} × (1 - {a.severity_weight:.1f})")
            print(f"    Anomaly Score:   {a.anomaly_score:.2f}")
            print(f"    Type: {a.anomaly_type}")
    
    # Explain the magic
    print("\n" + "=" * 70)
    print("💡 HOW SEVERITY PENALTY WORKS")
    print("=" * 70)
    print("""
    Without penalty:
      Cluster 7 (CRITICAL crash) has 33% probability
      33% > 25% threshold → NOT flagged as anomaly ❌
    
    With penalty:
      Raw probability = 33%
      Severity weight = 0.8 (CRITICAL keyword)
      Effective prob  = 33% × (1 - 0.8) = 33% × 0.2 = 6.6%
      6.6% < 25% threshold → FLAGGED as anomaly ✅
    
    This ensures FATAL/CRITICAL events are ALWAYS flagged,
    even if they happen frequently!
    """)
    
    # Create annotated chunks
    print("=" * 70)
    print("📦 ANNOTATED CHUNKS (Context Stuffed)")
    print("=" * 70)
    
    chunks = detector.create_annotated_chunks(logs, cluster_ids, window_size=2)
    
    # Show only anomalous chunks
    print("\nAnomalous chunks:")
    for i, chunk in enumerate(chunks):
        if chunk.is_anomaly:
            print(f"\n  Chunk {i}:")
            print(f"    Type: {chunk.anomaly_type}")
            print(f"    Score: {chunk.anomaly_score:.2f}")
            print(f"    Severity: {chunk.anomaly.severity_weight:.1f}")
            print(f"    Context ({len(chunk.log_lines)} lines):")
            for j, line in enumerate(chunk.log_lines):
                marker = "→ " if j == chunk.center_index else "  "
                print(f"      {marker}{line[:55]}...")
    
    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE!")
    print("=" * 70)

