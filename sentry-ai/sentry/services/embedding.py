# sentry/services/embedding.py
"""
Embedding service for converting text to vectors
Uses sentence-transformers with all-MiniLM-L6-v2
"""

import numpy as np
from typing import List, Union
from pathlib import Path
import logging

from sentence_transformers import SentenceTransformer

from ..core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Handles text-to-vector conversion using sentence transformers
    
    This is the "search engine" part of RAG - it converts text into
    numerical vectors that capture semantic meaning.
    """
    
    def __init__(self, model_name: str = None, cache_dir: Path = None):
        """
        Initialize the embedding service
        
        Args:
            model_name: HuggingFace model name (defaults to settings)
            cache_dir: Where to cache the model (defaults to settings)
        """
        self.model_name = model_name or settings.embedding_model
        self.cache_dir = cache_dir or settings.get_model_cache_path()
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading embedding model: {self.model_name}")
        logger.info(f"Cache directory: {self.cache_dir}")
        
        # Load the model
        # This downloads it on first run (~80MB), then caches it
        try:
            self.model = SentenceTransformer(
                self.model_name,
                cache_folder=str(self.cache_dir)
            )
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"✅ Model loaded. Embedding dimension: {self.dimension}")
            
            # Verify it matches our config
            if self.dimension != settings.embedding_dimension:
                logger.warning(
                    f"Model dimension ({self.dimension}) doesn't match "
                    f"settings ({settings.embedding_dimension}). Updating settings."
                )
                settings.embedding_dimension = self.dimension
                
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Convert a single piece of text to a vector
        
        Args:
            text: The text to embed
            
        Returns:
            numpy array of shape (dimension,)
            
        Example:
            >>> embedder = EmbeddingService()
            >>> vector = embedder.embed_text("ERROR: Disk failure")
            >>> vector.shape
            (384,)
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return np.zeros(self.dimension, dtype=np.float32)
        
        # Convert to vector
        # normalize_embeddings=True makes vectors unit length (good for cosine similarity)
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        return embedding
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Convert multiple texts to vectors in one batch
        Much faster than calling embed_text() in a loop
        
        Args:
            texts: List of texts to embed
            
        Returns:
            numpy array of shape (num_texts, dimension)
            
        Example:
            >>> texts = ["Error 1", "Error 2", "Error 3"]
            >>> vectors = embedder.embed_batch(texts)
            >>> vectors.shape
            (3, 384)
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.dimension)
        
        # Filter out empty strings
        valid_texts = [t if t and t.strip() else " " for t in texts]
        
        # Batch encoding is MUCH faster
        embeddings = self.model.encode(
            valid_texts,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100  # Show progress for large batches
        )
        
        return embeddings
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1, higher = more similar)
            
        Example:
            >>> embedder.similarity("disk error", "drive failure")
            0.87  # High similarity
            >>> embedder.similarity("disk error", "network timeout")
            0.23  # Low similarity
        """
        vec1 = self.embed_text(text1)
        vec2 = self.embed_text(text2)
        
        # Cosine similarity (since vectors are normalized, this is just dot product)
        similarity = np.dot(vec1, vec2)
        
        return float(similarity)
    
    def batch_similarity(
        self,
        query: str,
        texts: List[str]
    ) -> List[float]:
        """
        Calculate similarity between a query and multiple texts
        
        Args:
            query: The query text
            texts: List of texts to compare against
            
        Returns:
            List of similarity scores (same order as texts)
        """
        query_vec = self.embed_text(query)
        text_vecs = self.embed_batch(texts)
        
        # Vectorized dot product (fast!)
        similarities = np.dot(text_vecs, query_vec)
        
        return similarities.tolist()
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "cache_dir": str(self.cache_dir),
            "max_seq_length": self.model.max_seq_length,
        }
    
    def __repr__(self):
        return f"<EmbeddingService(model={self.model_name}, dim={self.dimension})>"


# ===== SINGLETON INSTANCE =====
# Create a global instance for easy access
# This loads the model once and reuses it
_embedding_service = None


def get_embedder() -> EmbeddingService:
    """
    Get the global embedding service instance
    Lazy-loads on first call
    
    Usage:
        embedder = get_embedder()
        vector = embedder.embed_text("some text")
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def embed_text(text: str) -> np.ndarray:
    """Convenience function for quick embedding"""
    return get_embedder().embed_text(text)


def embed_batch(texts: List[str]) -> np.ndarray:
    """Convenience function for batch embedding"""
    return get_embedder().embed_batch(texts)