# sentry/services/vectorstore.py
"""
Vector store using FAISS for fast similarity search
This is the "database" for embeddings
"""

import faiss
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
import logging

from ..core.config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-based vector store for semantic search
    
    This stores embeddings and provides fast similarity search.
    Think of it as a "database" where the primary key is semantic meaning.
    """
    
    def __init__(
        self,
        dimension: int = None,
        index_path: Path = None
    ):
        """
        Initialize vector store
        
        Args:
            dimension: Vector dimension (defaults to settings)
            index_path: Path to save/load index (defaults to settings)
        """
        self.dimension = dimension or settings.embedding_dimension
        self.index_path = index_path or settings.vector_index_path
        
        # FAISS index (the actual vector database)
        self.index = None
        
        # Metadata mapping: FAISS ID → Chunk ID
        # FAISS uses integer IDs (0, 1, 2, ...) but we need to map back to chunk IDs
        self.id_mapping = {}  # {faiss_id: chunk_id}
        self.reverse_mapping = {}  # {chunk_id: faiss_id}
        
        # Counter for next FAISS ID
        self.next_id = 0
        
        # Try to load existing index
        if self.index_path.exists():
            logger.info(f"Loading existing index from {self.index_path}")
            self.load()
        else:
            logger.info("Creating new FAISS index")
            self._create_index()
    
    def _create_index(self):
        """
        Create a new FAISS index
        
        We use IndexFlatIP (Inner Product) because:
        - Our vectors are normalized (unit length)
        - Inner product = cosine similarity for normalized vectors
        - It's exact search (no approximation)
        - Fast enough for desktop use (<1M vectors)
        """
        logger.info(f"Creating FAISS index with dimension {self.dimension}")
        
        # IndexFlatIP = Flat (exact) search using Inner Product
        # For larger datasets, you could use IndexIVFFlat or IndexHNSWFlat
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Sanity check
        assert self.index.is_trained
        logger.info(f"✅ FAISS index created (trained: {self.index.is_trained})")
    
    def add(
        self,
        vectors: np.ndarray,
        chunk_ids: List[str]
    ) -> List[int]:
        """
        Add vectors to the index
        
        Args:
            vectors: Array of shape (n, dimension)
            chunk_ids: List of chunk IDs corresponding to vectors
            
        Returns:
            List of FAISS IDs assigned to these vectors
            
        Example:
            >>> vectors = embedder.embed_batch(["text1", "text2"])
            >>> chunk_ids = ["chunk-1", "chunk-2"]
            >>> faiss_ids = store.add(vectors, chunk_ids)
            >>> faiss_ids
            [0, 1]
        """
        if len(vectors) != len(chunk_ids):
            raise ValueError(f"Mismatch: {len(vectors)} vectors but {len(chunk_ids)} chunk_ids")
        
        if len(vectors) == 0:
            return []
        
        # Ensure vectors are float32 (FAISS requirement)
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        
        # Ensure vectors are 2D
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        # Check dimension
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} doesn't match "
                f"index dimension {self.dimension}"
            )
        
        # Assign FAISS IDs
        faiss_ids = list(range(self.next_id, self.next_id + len(vectors)))
        
        # Update mappings
        for faiss_id, chunk_id in zip(faiss_ids, chunk_ids):
            self.id_mapping[faiss_id] = chunk_id
            self.reverse_mapping[chunk_id] = faiss_id
        
        # Add to FAISS index
        self.index.add(vectors)
        
        self.next_id += len(vectors)
        
        logger.debug(f"Added {len(vectors)} vectors. Total: {self.index.ntotal}")
        
        return faiss_ids
    
    def add_single(
        self,
        vector: np.ndarray,
        chunk_id: str
    ) -> int:
        """
        Add a single vector (convenience method)
        
        Args:
            vector: Single vector of shape (dimension,)
            chunk_id: Chunk ID for this vector
            
        Returns:
            FAISS ID assigned
        """
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        
        faiss_ids = self.add(vector, [chunk_id])
        return faiss_ids[0]
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = None
    ) -> Tuple[List[str], List[float]]:
        """
        Search for most similar vectors
        
        Args:
            query_vector: Query vector of shape (dimension,)
            k: Number of results to return (defaults to settings.top_k_results)
            
        Returns:
            Tuple of (chunk_ids, scores)
            - chunk_ids: List of matching chunk IDs
            - scores: List of similarity scores (higher = more similar)
            
        Example:
            >>> query_vec = embedder.embed_text("disk error")
            >>> chunk_ids, scores = store.search(query_vec, k=5)
            >>> for cid, score in zip(chunk_ids, scores):
            ...     print(f"{cid}: {score:.3f}")
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty, returning no results")
            return [], []
        
        k = k or settings.top_k_results
        
        # Don't search for more results than we have
        k = min(k, self.index.ntotal)
        
        # Ensure query is float32 and 2D
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)
        
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Search in FAISS
        # Returns: distances (scores), indices (FAISS IDs)
        scores, faiss_ids = self.index.search(query_vector, k)
        
        # Flatten arrays (search returns 2D even for single query)
        scores = scores[0]
        faiss_ids = faiss_ids[0]
        
        # Map FAISS IDs back to chunk IDs
        chunk_ids = []
        valid_scores = []
        
        for faiss_id, score in zip(faiss_ids, scores):
            # FAISS returns -1 for missing results (shouldn't happen with exact search)
            if faiss_id == -1:
                continue
            
            # Map to chunk ID
            chunk_id = self.id_mapping.get(faiss_id)
            if chunk_id:
                chunk_ids.append(chunk_id)
                valid_scores.append(float(score))
        
        logger.debug(f"Search returned {len(chunk_ids)} results (k={k})")
        
        return chunk_ids, valid_scores
    
    def search_with_threshold(
        self,
        query_vector: np.ndarray,
        k: int = None,
        threshold: float = None
    ) -> Tuple[List[str], List[float]]:
        """
        Search and filter by similarity threshold
        
        Args:
            query_vector: Query vector
            k: Max number of results
            threshold: Minimum similarity score (defaults to settings)
            
        Returns:
            Tuple of (chunk_ids, scores) where score >= threshold
        """
        threshold = threshold or settings.similarity_threshold
        
        # Search for more results than k (we'll filter)
        chunk_ids, scores = self.search(query_vector, k=k*2)
        
        # Filter by threshold
        filtered = [
            (cid, score)
            for cid, score in zip(chunk_ids, scores)
            if score >= threshold
        ]
        
        # Limit to k results
        filtered = filtered[:k]
        
        if filtered:
            chunk_ids, scores = zip(*filtered)
            return list(chunk_ids), list(scores)
        else:
            return [], []
    
    def remove(self, chunk_id: str) -> bool:
        """
        Remove a vector by chunk ID
        
        Note: FAISS doesn't support efficient removal, so we:
        1. Mark the ID as deleted in our mapping
        2. Periodically rebuild the index (see rebuild())
        
        Args:
            chunk_id: Chunk ID to remove
            
        Returns:
            True if removed, False if not found
        """
        faiss_id = self.reverse_mapping.get(chunk_id)
        if faiss_id is None:
            return False
        
        # Remove from mappings
        del self.id_mapping[faiss_id]
        del self.reverse_mapping[chunk_id]
        
        logger.debug(f"Marked chunk {chunk_id} for deletion (rebuild index to apply)")
        
        return True
    
    def rebuild(self, vectors: np.ndarray, chunk_ids: List[str]):
        """
        Rebuild index from scratch (for cleaning up deleted items)
        
        Args:
            vectors: All vectors to keep
            chunk_ids: Corresponding chunk IDs
        """
        logger.info(f"Rebuilding index with {len(vectors)} vectors")
        
        # Clear everything
        self._create_index()
        self.id_mapping.clear()
        self.reverse_mapping.clear()
        self.next_id = 0
        
        # Re-add all vectors
        if len(vectors) > 0:
            self.add(vectors, chunk_ids)
        
        logger.info(f"✅ Index rebuilt. Total vectors: {self.index.ntotal}")
    
    def save(self, path: Path = None):
        """
        Save index and mappings to disk
        
        Saves two files:
        - {path}: FAISS index (binary)
        - {path}.meta: Metadata (mappings, next_id)
        """
        path = path or self.index_path
        
        logger.info(f"Saving index to {path}")
        
        # Save FAISS index
        faiss.write_index(self.index, str(path))
        
        # Save metadata
        metadata = {
            'id_mapping': self.id_mapping,
            'reverse_mapping': self.reverse_mapping,
            'next_id': self.next_id,
            'dimension': self.dimension
        }
        
        meta_path = Path(str(path) + '.meta')
        with open(meta_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"✅ Saved {self.index.ntotal} vectors")
    
    def load(self, path: Path = None):
        """
        Load index and mappings from disk
        """
        path = path or self.index_path
        
        if not path.exists():
            raise FileNotFoundError(f"Index file not found: {path}")
        
        logger.info(f"Loading index from {path}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(path))
        
        # Load metadata
        meta_path = Path(str(path) + '.meta')
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        
        with open(meta_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.id_mapping = metadata['id_mapping']
        self.reverse_mapping = metadata['reverse_mapping']
        self.next_id = metadata['next_id']
        
        # Verify dimension - CRITICAL CHECK
        loaded_dim = metadata.get('dimension', self.dimension)
        if loaded_dim != self.dimension:
            error_msg = (
                f"❌ Dimension mismatch!\n"
                f"   Loaded index: {loaded_dim} dimensions\n"
                f"   Current model: {self.dimension} dimensions\n"
                f"   This happens when you change the embedding model.\n"
                f"\n"
                f"   Solution: Delete the old index files:\n"
                f"   - {self.index_path}\n"
                f"   - {self.index_path}.meta\n"
                f"\n"
                f"   Then restart and re-index your content."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"✅ Loaded {self.index.ntotal} vectors")
    
    def get_stats(self) -> dict:
        """Get statistics about the index"""
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'is_trained': self.index.is_trained if self.index else False,
            'next_id': self.next_id,
            'mappings_count': len(self.id_mapping)
        }
    
    def clear(self):
        """Clear all vectors from the index"""
        logger.info("Clearing vector store")
        self._create_index()
        self.id_mapping.clear()
        self.reverse_mapping.clear()
        self.next_id = 0
    
    def __len__(self):
        """Return number of vectors in index"""
        return self.index.ntotal if self.index else 0
    
    def __repr__(self):
        return f"<VectorStore(vectors={len(self)}, dim={self.dimension})>"


# ===== SINGLETON INSTANCE =====
_vector_store = None


def get_vector_store() -> VectorStore:
    """
    Get the global vector store instance
    Lazy-loads on first call
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store