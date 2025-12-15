"""
Trace Vector Store - Research Module
=====================================

A FAISS-based vector store specifically designed for log trace embeddings.
This stores trace embeddings WITH rich metadata (PIDs, IPs, timestamps, etc.)

Key difference from main vectorstore:
- Optimized for trace sequences (not text chunks)
- Stores structured metadata alongside embeddings
- Supports filtering by metadata fields

Think of it like this:
- Regular DB: Store rows with columns
- Vector DB: Store embeddings that we can search by similarity
- This: Store embeddings + structured metadata = best of both worlds

Usage:
    store = TraceVectorStore()
    
    # Add a trace with metadata
    store.add_trace(
        trace_id="trace_001",
        embedding=encoder.get_trace_vector([1, 2, 3, 4]),
        metadata={
            "cluster_ids": [1, 2, 3, 4],
            "pids": ["1234", "1234", "1234", "1234"],
            "ips": ["192.168.1.1"],
            "has_error": True,
            "error_type": "database_timeout",
            "timestamp": "2024-01-15T10:00:00"
        }
    )
    
    # Search for similar traces
    results = store.search(query_embedding, k=5)
    for trace_id, score, metadata in results:
        print(f"{trace_id}: {score:.3f} - Error: {metadata['error_type']}")
"""

import faiss
import numpy as np
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import logging
import re

logger = logging.getLogger(__name__)


class TraceVectorStore:
    """
    Vector store for log trace embeddings with metadata.
    
    This is like a searchable database where:
    - You can find similar traces by their "meaning" (embedding similarity)
    - Each trace carries structured metadata (PIDs, IPs, error types, etc.)
    
    Perfect for: "Find me other error traces similar to this one"
    """
    
    def __init__(
        self,
        dimension: int = 384,  # MiniLM default
        index_path: Path = None
    ):
        """
        Initialize the trace vector store.
        
        Args:
            dimension: Embedding dimension (384 for MiniLM, 768 for larger models)
            index_path: Where to save/load the index (None = in-memory only)
        """
        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else None
        
        # FAISS index for fast similarity search
        self.index = None
        
        # Metadata store: trace_id -> {metadata dict}
        # This is where we keep PIDs, IPs, timestamps, etc.
        self.metadata_store: Dict[str, Dict[str, Any]] = {}
        
        # ID mappings (FAISS uses integers, we use string trace_ids)
        self.id_to_trace: Dict[int, str] = {}  # faiss_id -> trace_id
        self.trace_to_id: Dict[str, int] = {}  # trace_id -> faiss_id
        self.next_id = 0
        
        # Try to load existing index
        if self.index_path and self.index_path.exists():
            self.load()
        else:
            self._create_index()
    
    def _create_index(self):
        """Create a new FAISS index."""
        # IndexFlatIP = exact search using inner product (= cosine similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(self.dimension)
        logger.info(f"Created FAISS index (dim={self.dimension})")
    
    def add_trace(
        self,
        trace_id: str,
        embedding: np.ndarray,
        metadata: Dict[str, Any]
    ) -> int:
        """
        Add a single trace to the store.
        
        Args:
            trace_id: Unique identifier for this trace (e.g., "trace_001")
            embedding: Vector representation of the trace
            metadata: Structured data about the trace:
                - cluster_ids: List[int] - The DNA sequence
                - raw_logs: List[str] - Original log lines (optional)
                - pids: List[str] - Process IDs found in logs
                - ips: List[str] - IP addresses found
                - memory_addresses: List[str] - Memory refs found
                - has_error: bool - Whether this trace contains errors
                - error_type: str - Type of error (if any)
                - timestamp: str - When this trace started
                - source: str - Where logs came from (datadog, vercel, etc.)
                
        Returns:
            FAISS ID assigned to this trace
        """
        # Validate embedding
        if embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
        
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)
        
        if embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embedding.shape[1]} doesn't match "
                f"index dimension {self.dimension}"
            )
        
        # Assign FAISS ID
        faiss_id = self.next_id
        self.next_id += 1
        
        # Store mappings
        self.id_to_trace[faiss_id] = trace_id
        self.trace_to_id[trace_id] = faiss_id
        
        # Store metadata (with some auto-extraction if not provided)
        enriched_metadata = self._enrich_metadata(metadata)
        self.metadata_store[trace_id] = enriched_metadata
        
        # Add to FAISS index
        self.index.add(embedding)
        
        logger.debug(f"Added trace {trace_id} (faiss_id={faiss_id})")
        return faiss_id
    
    def add_traces_batch(
        self,
        trace_ids: List[str],
        embeddings: np.ndarray,
        metadata_list: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Add multiple traces at once (faster than calling add_trace in a loop).
        
        Args:
            trace_ids: List of unique trace identifiers
            embeddings: Array of shape (n_traces, dimension)
            metadata_list: List of metadata dicts (one per trace)
            
        Returns:
            List of FAISS IDs
        """
        if len(trace_ids) != len(metadata_list):
            raise ValueError("trace_ids and metadata_list must have same length")
        
        if embeddings.shape[0] != len(trace_ids):
            raise ValueError("embeddings count doesn't match trace_ids")
        
        faiss_ids = []
        for trace_id, embedding, metadata in zip(trace_ids, embeddings, metadata_list):
            fid = self.add_trace(trace_id, embedding, metadata)
            faiss_ids.append(fid)
        
        return faiss_ids
    
    def _enrich_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich metadata with extracted patterns and defaults.
        
        This auto-extracts PIDs, IPs, memory addresses from raw logs
        if they weren't provided explicitly.
        """
        enriched = metadata.copy()
        
        # Add timestamp if not present
        if "indexed_at" not in enriched:
            enriched["indexed_at"] = datetime.now().isoformat()
        
        # If we have raw_logs, try to extract structured data
        raw_logs = enriched.get("raw_logs", [])
        if raw_logs and isinstance(raw_logs, list):
            # Extract PIDs if not provided
            if "pids" not in enriched:
                enriched["pids"] = self._extract_pids(raw_logs)
            
            # Extract IPs if not provided
            if "ips" not in enriched:
                enriched["ips"] = self._extract_ips(raw_logs)
            
            # Extract memory addresses if not provided
            if "memory_addresses" not in enriched:
                enriched["memory_addresses"] = self._extract_memory_addresses(raw_logs)
            
            # Detect if error trace
            if "has_error" not in enriched:
                enriched["has_error"] = self._detect_error(raw_logs)
        
        return enriched
    
    def _extract_pids(self, logs: List[str]) -> List[str]:
        """Extract process IDs from log lines."""
        pids = set()
        # Common PID patterns: PID=1234, pid:1234, [1234], process 1234
        patterns = [
            r'[Pp][Ii][Dd][=:\s](\d+)',
            r'\[(\d{4,6})\]',
            r'process[:\s]+(\d+)',
        ]
        for log in logs:
            for pattern in patterns:
                matches = re.findall(pattern, log)
                pids.update(matches)
        return list(pids)
    
    def _extract_ips(self, logs: List[str]) -> List[str]:
        """Extract IP addresses from log lines."""
        ips = set()
        # IPv4 pattern
        pattern = r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\b'
        for log in logs:
            matches = re.findall(pattern, log)
            ips.update(matches)
        return list(ips)
    
    def _extract_memory_addresses(self, logs: List[str]) -> List[str]:
        """Extract memory addresses (hex) from log lines."""
        addresses = set()
        # Hex address pattern: 0x followed by hex digits
        pattern = r'(0x[0-9a-fA-F]{4,16})'
        for log in logs:
            matches = re.findall(pattern, log)
            addresses.update(matches)
        return list(addresses)
    
    def _detect_error(self, logs: List[str]) -> bool:
        """Detect if logs contain error indicators."""
        error_keywords = ['error', 'exception', 'failed', 'failure', 'fatal', 'critical']
        for log in logs:
            if any(kw in log.lower() for kw in error_keywords):
                return True
        return False
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        filter_fn: callable = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar traces.
        
        Args:
            query_embedding: The trace embedding to search for
            k: Number of results to return
            filter_fn: Optional filter function that takes metadata and returns bool
                       Example: lambda m: m.get("has_error") == True
                       
        Returns:
            List of (trace_id, similarity_score, metadata) tuples
            Sorted by similarity (highest first)
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty")
            return []
        
        # Prepare query
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Search for more than k if we have a filter (we'll filter down)
        search_k = min(k * 3 if filter_fn else k, self.index.ntotal)
        
        # FAISS search
        scores, faiss_ids = self.index.search(query_embedding, search_k)
        scores = scores[0]  # Flatten
        faiss_ids = faiss_ids[0]
        
        # Build results with metadata
        results = []
        for faiss_id, score in zip(faiss_ids, scores):
            if faiss_id == -1:
                continue
            
            trace_id = self.id_to_trace.get(faiss_id)
            if not trace_id:
                continue
            
            metadata = self.metadata_store.get(trace_id, {})
            
            # Apply filter if provided
            if filter_fn and not filter_fn(metadata):
                continue
            
            results.append((trace_id, float(score), metadata))
            
            if len(results) >= k:
                break
        
        return results
    
    def search_error_traces(
        self,
        query_embedding: np.ndarray,
        k: int = 5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Convenience method: Search only for error traces.
        
        This is the key use case: "Find similar errors to this one"
        """
        return self.search(
            query_embedding,
            k=k,
            filter_fn=lambda m: m.get("has_error", False)
        )
    
    def search_by_ip(
        self,
        query_embedding: np.ndarray,
        ip: str,
        k: int = 5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar traces that involve a specific IP.
        
        Useful for: "What other issues happened with this server?"
        """
        return self.search(
            query_embedding,
            k=k,
            filter_fn=lambda m: ip in m.get("ips", [])
        )
    
    def search_by_pid(
        self,
        query_embedding: np.ndarray,
        pid: str,
        k: int = 5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar traces involving a specific process.
        
        Useful for: "What happened in this process?"
        """
        return self.search(
            query_embedding,
            k=k,
            filter_fn=lambda m: pid in m.get("pids", [])
        )
    
    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific trace."""
        return self.metadata_store.get(trace_id)
    
    def get_all_traces(self) -> Dict[str, Dict[str, Any]]:
        """Get all traces and their metadata."""
        return self.metadata_store.copy()
    
    def save(self, path: Path = None):
        """Save the index and metadata to disk."""
        path = path or self.index_path
        if not path:
            raise ValueError("No save path specified")
        
        path = Path(path)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path))
        
        # Save metadata
        meta = {
            "metadata_store": self.metadata_store,
            "id_to_trace": self.id_to_trace,
            "trace_to_id": self.trace_to_id,
            "next_id": self.next_id,
            "dimension": self.dimension
        }
        
        meta_path = Path(str(path) + ".meta")
        with open(meta_path, "wb") as f:
            pickle.dump(meta, f)
        
        logger.info(f"Saved {len(self.metadata_store)} traces to {path}")
    
    def load(self, path: Path = None):
        """Load the index and metadata from disk."""
        path = path or self.index_path
        if not path:
            raise ValueError("No load path specified")
        
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Index not found: {path}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(path))
        
        # Load metadata
        meta_path = Path(str(path) + ".meta")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        
        self.metadata_store = meta["metadata_store"]
        self.id_to_trace = meta["id_to_trace"]
        self.trace_to_id = meta["trace_to_id"]
        self.next_id = meta["next_id"]
        
        logger.info(f"Loaded {len(self.metadata_store)} traces from {path}")
    
    def clear(self):
        """Clear all traces from the store."""
        self._create_index()
        self.metadata_store.clear()
        self.id_to_trace.clear()
        self.trace_to_id.clear()
        self.next_id = 0
        logger.info("Cleared trace store")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the store."""
        error_count = sum(
            1 for m in self.metadata_store.values() 
            if m.get("has_error", False)
        )
        
        return {
            "total_traces": len(self.metadata_store),
            "error_traces": error_count,
            "normal_traces": len(self.metadata_store) - error_count,
            "dimension": self.dimension,
            "index_size": self.index.ntotal if self.index else 0
        }
    
    def __len__(self):
        return len(self.metadata_store)
    
    def __repr__(self):
        stats = self.get_stats()
        return (
            f"<TraceVectorStore(traces={stats['total_traces']}, "
            f"errors={stats['error_traces']}, dim={self.dimension})>"
        )
