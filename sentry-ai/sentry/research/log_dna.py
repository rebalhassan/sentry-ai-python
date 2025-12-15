"""
Log DNA Encoder - Simple Research Script
=========================================

Turns raw logs into DNA-like strings using Drain3 clustering.

Future: We'll use transition probabilities to detect anomalies.
  - If A -> B happens 90% of the time, but we see A -> X (10%), 
    that's likely an error pattern!

Usage:
    python log_dna.py

Author: Research Module
"""

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from collections import defaultdict
import json
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Sentence Transformers for trace embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️  sentence-transformers not installed. Install with: pip install sentence-transformers")

# PM4Py imports
try:
    import pm4py
    from pm4py.objects.log.obj import EventLog, Event, Trace
    from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
    from pm4py.visualization.petri_net import visualizer as pn_visualizer
    PM4PY_AVAILABLE = True
except ImportError:
    PM4PY_AVAILABLE = False
    print("⚠️  PM4Py not installed. Install with: pip install pm4py")

# Fix Graphviz PATH issue on Windows
import os
import platform
if platform.system() == "Windows":
    graphviz_paths = [
        r"C:\Program Files\Graphviz\bin",
        r"C:\Program Files (x86)\Graphviz\bin",
    ]
    for gv_path in graphviz_paths:
        if os.path.exists(gv_path) and gv_path not in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + gv_path
            print(f"✅ Added Graphviz to PATH: {gv_path}")
            break



class LogDNAEncoder:
    """
    Encodes raw logs into DNA-like cluster ID sequences.
    
    Think of it like:
    - Raw logs = messy book chapters
    - DNA string = "1-1-2-3-1-2" (chapter type sequence)
    - Codebook = {1: "login event", 2: "error", 3: "transaction"}
    """
    
    def __init__(self, persistence_path: str = None):
        """
        Initialize the encoder.
        
        Args:
            persistence_path: Optional path to save/load cluster state
        """
        # Configure Drain3
        config = TemplateMinerConfig()
        config.drain_sim_th = 0.4  # Similarity threshold (lower = more clusters)
        config.drain_depth = 4     # Tree depth
        config.drain_max_clusters = 1000  # Max clusters to create
        
        self.miner = TemplateMiner(config=config)
        self.persistence_path = Path(persistence_path) if persistence_path else None
        
        # Track cluster info
        self.cluster_counts = defaultdict(int)
        
    def add_log(self, log_message: str) -> int:
        """
        Add a single log message and get its cluster ID.
        
        Args:
            log_message: Raw log line
            
        Returns:
            Cluster ID (integer)
        """
        result = self.miner.add_log_message(log_message)
        cluster_id = result["cluster_id"]
        self.cluster_counts[cluster_id] += 1
        return cluster_id
    
    def encode_logs(self, logs: list[str]) -> list[int]:
        """
        Encode a list of logs into cluster IDs.
        
        Args:
            logs: List of raw log messages
            
        Returns:
            List of cluster IDs in order
        """
        return [self.add_log(log) for log in logs]
    
    def to_dna_string(self, cluster_ids: list[int], separator: str = "-") -> str:
        """
        Convert cluster IDs to a DNA-like string.
        
        Args:
            cluster_ids: List of cluster IDs
            separator: Character to join IDs (default: "-")
            
        Returns:
            DNA string like "1-1-2-3-1-2"
        """
        return separator.join(str(cid) for cid in cluster_ids)
    
    def get_codebook(self) -> dict:
        """
        Get the cluster codebook (ID -> template mapping).
        
        Returns:
            Dictionary of {cluster_id: {"template": "...", "count": N}}
        """
        codebook = {}
        for cluster in self.miner.drain.clusters:
            codebook[cluster.cluster_id] = {
                "template": cluster.get_template(),
                "count": self.cluster_counts[cluster.cluster_id]
            }
        return codebook
    
    def get_summary(self) -> str:
        """
        Get a human-readable summary of all clusters.
        
        Returns:
            Formatted string showing all templates
        """
        codebook = self.get_codebook()
        lines = ["=== Cluster Codebook ==="]
        for cid, info in sorted(codebook.items()):
            lines.append(f"  [{cid}] ({info['count']}x) {info['template']}")
        return "\n".join(lines)
    
    def save_state(self, filepath: str = None):
        """Save the current state to a JSON file."""
        path = Path(filepath) if filepath else self.persistence_path
        if not path:
            raise ValueError("No persistence path specified")
        
        state = {
            "codebook": self.get_codebook(),
            "cluster_counts": dict(self.cluster_counts)
        }
        path.write_text(json.dumps(state, indent=2))
        print(f"Saved state to {path}")
    
    def get_transition_matrix(self, cluster_ids: list[int]) -> dict:
        """
        Calculate transition probabilities between clusters.
        
        This is the "heuristics" part - we'll use this to detect anomalies.
        If A -> B is 90% but we see A -> X (10%), that's unusual!
        
        Args:
            cluster_ids: Sequence of cluster IDs
            
        Returns:
            Dictionary of {from_id: {to_id: probability}}
        """
        if len(cluster_ids) < 2:
            return {}
        
        # Count transitions
        transitions = defaultdict(lambda: defaultdict(int))
        for i in range(len(cluster_ids) - 1):
            from_id = cluster_ids[i]
            to_id = cluster_ids[i + 1]
            transitions[from_id][to_id] += 1
        
        # Convert to probabilities
        probabilities = {}
        for from_id, to_counts in transitions.items():
            total = sum(to_counts.values())
            probabilities[from_id] = {
                to_id: round(count / total, 3)
                for to_id, count in to_counts.items()
            }
        
        return probabilities
    
    def format_transitions(self, transitions: dict) -> str:
        """Pretty print transition probabilities."""
        lines = ["=== Transition Probabilities ==="]
        for from_id, to_probs in sorted(transitions.items()):
            for to_id, prob in sorted(to_probs.items(), key=lambda x: -x[1]):
                pct = int(prob * 100)
                bar = "█" * (pct // 5)  # Visual bar
                lines.append(f"  [{from_id}] -> [{to_id}]: {pct:3d}% {bar}")
        return "\n".join(lines)
    
    def to_event_log(self, cluster_ids: list[int], case_id: str = "case_1") -> 'EventLog':
        """
        Convert cluster IDs to PM4Py EventLog format.
        
        This creates a process mining event log where each cluster ID
        becomes an activity in the process.
        
        Args:
            cluster_ids: List of cluster IDs
            case_id: Case identifier (default: "case_1")
            
        Returns:
            PM4Py EventLog object
        """
        if not PM4PY_AVAILABLE:
            raise ImportError("PM4Py is not installed. Run: pip install pm4py")
        
        # Get codebook for activity names
        codebook = self.get_codebook()
        
        # Create a trace (sequence of events for one case)
        trace = Trace()
        trace.attributes["concept:name"] = case_id
        
        # Add events with timestamps
        base_time = datetime.now()
        for i, cluster_id in enumerate(cluster_ids):
            event = Event()
            event["concept:name"] = f"Cluster_{cluster_id}"
            event["cluster_id"] = cluster_id
            event["template"] = codebook.get(cluster_id, {}).get("template", "Unknown")
            event["time:timestamp"] = base_time + timedelta(seconds=i)
            trace.append(event)
        
        # Create event log and add the trace
        event_log = EventLog()
        event_log.append(trace)
        
        return event_log
    
    def discover_process_model(self, cluster_ids: list[int]):
        """
        Use PM4Py's heuristics miner to discover the process model.
        
        This creates a Petri net showing:
        - Normal flow paths (high probability transitions)
        - Anomalous paths (low probability transitions)
        
        Args:
            cluster_ids: List of cluster IDs
            
        Returns:
            Tuple of (net, initial_marking, final_marking)
        """
        if not PM4PY_AVAILABLE:
            raise ImportError("PM4Py is not installed. Run: pip install pm4py")
        
        event_log = self.to_event_log(cluster_ids)
        
        # Apply heuristics miner
        net, initial_marking, final_marking = heuristics_miner.apply(
            event_log,
            parameters={
                "dependency_threshold": 0.5,  # Lower = more edges shown
                "and_threshold": 0.65,
                "loop_two_threshold": 0.5
            }
        )
        
        return net, initial_marking, final_marking
    
    def visualize_petri_net(self, cluster_ids: list[int], output_path: str = None):
        """
        Visualize the process as a Petri net using PM4Py.
        
        This shows the flow of log events as a graph:
        - Circles = Places (states)
        - Rectangles = Transitions (activities/cluster IDs)
        - Arrows = Flow direction
        
        Args:
            cluster_ids: List of cluster IDs
            output_path: Optional path to save the visualization (PNG/SVG)
        """
        if not PM4PY_AVAILABLE:
            raise ImportError("PM4Py is not installed. Run: pip install pm4py")
        
        print("\n🔍 Discovering process model with Heuristics Miner...")
        net, im, fm = self.discover_process_model(cluster_ids)
        
        # Visualize
        gviz = pn_visualizer.apply(
            net, im, fm,
            parameters={
                "format": "png",
                "debug": False
            }
        )
        
        if output_path:
            pn_visualizer.save(gviz, output_path)
            print(f"✅ Petri net saved to: {output_path}")
        else:
            # Display in viewer
            pn_visualizer.view(gviz)
            print("✅ Petri net visualization opened!")

    # ============================================================
    # SENTENCE TRANSFORMER TRACE VECTORS
    # ============================================================
    
    def init_embeddings(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the sentence transformer model for trace embeddings.
        
        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2, fast & good)
        """
        if not EMBEDDINGS_AVAILABLE:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        
        self.embed_model = SentenceTransformer(model_name)
        print(f"✅ Loaded embedding model: {model_name}")
    
    def _trace_to_text(self, cluster_ids: list[int], window_size: int = 5) -> str:
        """
        Convert a trace of cluster IDs to a text representation for embedding.
        
        Uses the codebook templates to create meaningful text.
        
        Args:
            cluster_ids: Sequence of cluster IDs
            window_size: Context window (5 before + 5 after = 11 total)
            
        Returns:
            Text representation of the trace
        """
        codebook = self.get_codebook()
        
        # Convert cluster IDs to their template descriptions
        parts = []
        for cid in cluster_ids:
            if cid in codebook:
                template = codebook[cid]["template"]
                parts.append(f"[{cid}:{template}]")
            else:
                parts.append(f"[{cid}:unknown]")
        
        return " -> ".join(parts)
    
    def get_trace_vector(self, cluster_ids: list[int]) -> np.ndarray:
        """
        Get a vector representation for a trace/sequence.
        
        This creates a fixed-size vector for any length sequence,
        perfect for vector search.
        
        Args:
            cluster_ids: Sequence of cluster IDs
            
        Returns:
            Numpy array of shape (384,) for MiniLM model
        """
        if not hasattr(self, 'embed_model'):
            self.init_embeddings()
        
        # Convert trace to text representation
        trace_text = self._trace_to_text(cluster_ids)
        
        # Get embedding
        return self.embed_model.encode(trace_text)
    
    def get_window_vectors(
        self, 
        cluster_ids: list[int], 
        window_size: int = 5
    ) -> list[dict]:
        """
        Create vectors for sliding windows of events.
        
        This is useful for finding similar "moments" in different traces.
        Each window captures the context: 5 events around a center event.
        
        Args:
            cluster_ids: Full sequence of cluster IDs
            window_size: How many events each side (default: 5, so 11 total)
            
        Returns:
            List of dicts with:
            - "center_idx": Position in original sequence
            - "center_cluster": The center event cluster ID
            - "window": The cluster IDs in this window
            - "vector": The embedding for this window
        """
        if not hasattr(self, 'embed_model'):
            self.init_embeddings()
        
        results = []
        
        for i, center_cluster in enumerate(cluster_ids):
            # Extract window around center
            start = max(0, i - window_size)
            end = min(len(cluster_ids), i + window_size + 1)
            window = cluster_ids[start:end]
            
            # Get vector for this window
            vector = self.get_trace_vector(window)
            
            results.append({
                "center_idx": i,
                "center_cluster": center_cluster,
                "window": window,
                "vector": vector
            })
        
        return results
    
    def find_similar_traces(
        self, 
        query_trace: list[int], 
        all_traces: list[list[int]], 
        topn: int = 5
    ) -> list[tuple[int, float, list[int]]]:
        """
        Find traces most similar to a query trace.
        
        Great for: "Show me other error chains similar to this one"
        
        Args:
            query_trace: The trace to find similar ones for
            all_traces: List of all traces to search through
            topn: How many similar traces to return
            
        Returns:
            List of (trace_idx, similarity_score, trace) tuples
        """
        if not hasattr(self, 'embed_model'):
            self.init_embeddings()
        
        # Get query vector
        query_vec = self.get_trace_vector(query_trace)
        
        # Calculate similarity to all traces
        similarities = []
        for idx, trace in enumerate(all_traces):
            trace_vec = self.get_trace_vector(trace)
            # Cosine similarity
            sim = np.dot(query_vec, trace_vec) / (
                np.linalg.norm(query_vec) * np.linalg.norm(trace_vec) + 1e-8
            )
            similarities.append((idx, sim, trace))
        
        # Sort by similarity
        similarities.sort(key=lambda x: -x[1])
        return similarities[:topn]


# ============================================================
# TEST IT OUT!
# ============================================================

if __name__ == "__main__":
    # Sample logs (simulating a typical session with an error)
    sample_logs = [
        # Normal flow
        "2024-01-15 10:00:01 INFO User john_doe logged in from 192.168.1.100",
        "2024-01-15 10:00:02 INFO User john_doe requested dashboard",
        "2024-01-15 10:00:03 INFO Query executed successfully in 45ms",
        "2024-01-15 10:00:05 INFO User john_doe logged out",
        
        # Another normal session
        "2024-01-15 10:01:01 INFO User alice_smith logged in from 10.0.0.50",
        "2024-01-15 10:01:02 INFO User alice_smith requested dashboard", 
        "2024-01-15 10:01:03 INFO Query executed successfully in 32ms",
        "2024-01-15 10:01:04 INFO User alice_smith logged out",
        
        # Session with error!
        "2024-01-15 10:02:01 INFO User bob_jones logged in from 172.16.0.1",
        "2024-01-15 10:02:02 INFO User bob_jones requested dashboard",
        "2024-01-15 10:02:03 ERROR Database connection failed: timeout after 30s",  # ← The anomaly!
        "2024-01-15 10:02:04 ERROR User bob_jones session terminated unexpectedly",
        
        # More normal sessions
        "2024-01-15 10:03:01 INFO User charlie logged in from 192.168.1.200",
        "2024-01-15 10:03:02 INFO User charlie requested dashboard",
        "2024-01-15 10:03:03 INFO Query executed successfully in 28ms",
        "2024-01-15 10:03:04 INFO User charlie logged out",
    ]
    
    print("=" * 60)
    print("🧬 LOG DNA ENCODER - Research Demo")
    print("=" * 60)
    print()
    
    # Create encoder
    encoder = LogDNAEncoder()
    
    # Encode all logs
    print("📥 Processing logs...")
    cluster_ids = encoder.encode_logs(sample_logs)
    
    # Show DNA string
    dna = encoder.to_dna_string(cluster_ids)
    print(f"\n🧬 DNA String: {dna}")
    
    # Show codebook
    print(f"\n{encoder.get_summary()}")
    
    # Show transition probabilities
    transitions = encoder.get_transition_matrix(cluster_ids)
    print(f"\n{encoder.format_transitions(transitions)}")
    
    # Analysis hint
    print("\n" + "=" * 60)
    print("💡 INSIGHT:")
    print("=" * 60)
    print("Look at the transition probabilities above.")
    print("Normal flow: login -> dashboard -> query success -> logout")
    print("Error flow:  login -> dashboard -> DB ERROR -> session terminated")
    print()
    print("The 'DB ERROR' transition is rare (anomaly signal!)")
    print("An LLM can use this DNA + codebook to understand the pattern!")
    
    # PM4Py Visualization
    if PM4PY_AVAILABLE:
        print("\n" + "=" * 60)
        print("🎨 PM4PY HEURISTICS MINER VISUALIZATION")
        print("=" * 60)
        try:
            encoder.visualize_petri_net(
                cluster_ids,
                output_path="log_dna_petri_net.png"
            )
            print("\n📊 The Petri net shows:")
            print("  - Rectangles = Log event types (cluster IDs)")
            print("  - Circles = Process states")
            print("  - Arrows = Flow direction")
            print("  - Thick arrows = High probability transitions")
            print("  - Thin arrows = Low probability (anomalies!)")
        except Exception as e:
            print(f"⚠️  Could not generate Petri net: {e}")
    else:
        print("\n" + "=" * 60)
        print("⚠️  Install PM4Py to see Petri net visualization:")
        print("    pip install pm4py")
        print("=" * 60)
    
    # Sentence Transformer Trace Vectors
    if EMBEDDINGS_AVAILABLE:
        print("\n" + "=" * 60)
        print("🧠 SENTENCE TRANSFORMER TRACE VECTORS")
        print("=" * 60)
        
        # Create traces for comparison
        traces = [
            # Normal traces
            [1, 2, 3, 2],  # login -> dashboard -> success -> dashboard
            [1, 2, 3, 2],  # same pattern
            [1, 2, 3, 2],  # same pattern
            # Error trace
            [1, 2, 4, 5],  # login -> dashboard -> DB error -> terminated
            # Original full trace
            cluster_ids,
        ]
        
        try:
            # Initialize embeddings (auto-loads on first use)
            print("\n� Loading embedding model...")
            encoder.init_embeddings()
            
            # Get trace vectors
            print("\n📐 Trace vectors (for vector search):")
            normal_trace = [1, 2, 3, 2]
            error_trace = [1, 2, 4, 5]
            
            normal_vec = encoder.get_trace_vector(normal_trace)
            error_vec = encoder.get_trace_vector(error_trace)
            
            print(f"  Normal trace {normal_trace}: vector shape {normal_vec.shape}")
            print(f"  Error trace {error_trace}: vector shape {error_vec.shape}")
            
            # Show the text representation
            print("\n📝 Trace text representations:")
            print(f"  Normal: {encoder._trace_to_text(normal_trace)[:80]}...")
            print(f"  Error:  {encoder._trace_to_text(error_trace)[:80]}...")
            
            # Calculate similarity between traces
            from numpy.linalg import norm
            similarity = np.dot(normal_vec, error_vec) / (norm(normal_vec) * norm(error_vec) + 1e-8)
            print(f"\n  Similarity between normal and error trace: {similarity:.3f}")
            print("  (Lower = more different, which is good for anomaly detection!)")
            
            # Find similar traces
            print("\n🔍 Finding traces similar to the error trace:")
            similar_traces = encoder.find_similar_traces(error_trace, traces, topn=3)
            for idx, score, trace in similar_traces:
                print(f"  Trace {idx}: {trace} (similarity: {score:.3f})")
            
        except Exception as e:
            import traceback
            print(f"⚠️  Trace vector error: {e}")
            traceback.print_exc()
    else:
        print("\n" + "=" * 60)
        print("⚠️  Install sentence-transformers to use trace vectors:")
        print("    pip install sentence-transformers")
        print("=" * 60)
    
    print("\n" + "=" * 60)
    print("✅ DEMO COMPLETE!")
    print("=" * 60)
