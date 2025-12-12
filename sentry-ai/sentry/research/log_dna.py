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
