# sentry/services/rag.py
"""
RAG (Retrieval Augmented Generation) service
This is the brain that connects all AI components together
"""

import time
import logging
from typing import List, Tuple, Optional
from datetime import datetime

from ..core.models import LogChunk, QueryResult
from ..core.database import db
from ..core.config import settings

from .embedding import get_embedder
from .vectorstore import get_vector_store
from .llm import get_llm_client

logger = logging.getLogger(__name__)


class RAGService:
    """
    RAG service - the orchestrator
    
    Flow:
    1. User asks: "What disk errors happened?"
    2. Embed the query → vector
    3. Search vector store → find similar log chunks
    4. Retrieve full chunks from database
    5. Format chunks as context
    6. Ask LLM to answer based on context
    7. Return answer with sources
    """
    
    def __init__(self):
        """Initialize RAG service with all components"""
        logger.info("Initializing RAG service...")
        
        # Load components (lazy-loaded singletons)
        self.embedder = get_embedder()
        self.vector_store = get_vector_store()
        self.llm = get_llm_client()
        
        logger.info("✅ RAG service ready")
        logger.info(f"   Embedder: {self.embedder.model_name}")
        logger.info(f"   Vector store: {len(self.vector_store)} vectors")
        logger.info(f"   LLM: {self.llm.model}")
    
    def query(
        self,
        query_text: str,
        top_k: int = None,
        similarity_threshold: float = None,
        use_reranking: bool = None
    ) -> QueryResult:
        """
        Answer a query using RAG
        
        This is the main method - everything flows through here.
        
        Args:
            query_text: User's question
            top_k: Number of results to retrieve
            similarity_threshold: Minimum similarity score
            use_reranking: Whether to use BM25 reranking
            
        Returns:
            QueryResult with answer and sources
            
        Example:
            >>> rag = RAGService()
            >>> result = rag.query("What disk errors happened?")
            >>> print(result.answer)
            >>> for chunk in result.sources:
            ...     print(chunk.content)
        """
        start_time = time.time()
        
        # Use defaults from settings if not provided
        top_k = top_k or settings.top_k_results
        similarity_threshold = similarity_threshold or settings.similarity_threshold
        use_reranking = use_reranking if use_reranking is not None else settings.use_reranking
        
        logger.info(f"RAG Query: '{query_text}'")
        logger.info(f"  top_k={top_k}, threshold={similarity_threshold}")
        
        try:
            # Step 1: Embed the query
            logger.debug("Step 1: Embedding query...")
            query_vector = self.embedder.embed_text(query_text)
            
            # Step 2: Search vector store
            logger.debug("Step 2: Searching vector store...")
            chunk_ids, scores = self.vector_store.search_with_threshold(
                query_vector,
                k=top_k,
                threshold=similarity_threshold
            )
            
            if not chunk_ids:
                logger.warning("No relevant chunks found")
                return QueryResult(
                    answer="I couldn't find any relevant log entries for your query. Try rephrasing or check if logs are indexed.",
                    sources=[],
                    confidence=0.0,
                    query_time=time.time() - start_time,
                    chunk_ids=[]
                )
            
            logger.info(f"  Found {len(chunk_ids)} relevant chunks")
            
            # Step 3: Retrieve full chunks from database
            logger.debug("Step 3: Retrieving chunks from database...")
            chunks = db.get_chunks_by_ids(chunk_ids)
            
            # Step 4: Optional reranking (BM25 for exact keyword matches)
            if use_reranking:
                logger.debug("Step 4: Reranking with BM25...")
                chunks, scores = self._rerank_bm25(query_text, chunks, scores)
            
            # Step 5: Generate answer with LLM
            logger.debug("Step 5: Generating answer with LLM...")
            answer = self.llm.generate_with_context(query_text, chunks)
            
            # Calculate confidence (average similarity score)
            confidence = sum(scores) / len(scores) if scores else 0.0
            
            query_time = time.time() - start_time
            
            logger.info(f"✅ Query completed in {query_time:.2f}s")
            logger.info(f"   Confidence: {confidence:.3f}")
            
            return QueryResult(
                answer=answer,
                sources=chunks,
                confidence=confidence,
                query_time=query_time,
                chunk_ids=chunk_ids
            )
            
        except Exception as e:
            logger.error(f"RAG query failed: {e}", exc_info=True)
            
            return QueryResult(
                answer=f"An error occurred while processing your query: {str(e)}",
                sources=[],
                confidence=0.0,
                query_time=time.time() - start_time,
                chunk_ids=[]
            )
    
    def _rerank_bm25(
        self,
        query: str,
        chunks: List[LogChunk],
        vector_scores: List[float]
    ) -> Tuple[List[LogChunk], List[float]]:
        """
        Rerank results using BM25 (keyword-based)
        
        Why this matters:
        - Vector search is semantic (understands meaning)
        - BM25 is lexical (exact keyword matches)
        - Combining both gives best results
        
        Example:
        - Query: "EventID 7"
        - Vector search might miss exact event IDs
        - BM25 catches exact matches
        
        Args:
            query: User query
            chunks: Retrieved chunks
            vector_scores: Original similarity scores
            
        Returns:
            Reranked chunks and new scores
        """
        try:
            from rank_bm25 import BM25Okapi
            
            # Tokenize documents
            tokenized_docs = [chunk.content.lower().split() for chunk in chunks]
            
            # Create BM25 index
            bm25 = BM25Okapi(tokenized_docs)
            
            # Score against query
            tokenized_query = query.lower().split()
            bm25_scores = bm25.get_scores(tokenized_query)
            
            # Normalize BM25 scores to 0-1 range
            max_bm25 = max(bm25_scores) if bm25_scores.max() > 0 else 1.0
            bm25_scores_norm = bm25_scores / max_bm25
            
            # Combine scores (70% vector, 30% BM25)
            combined_scores = [
                0.7 * vec_score + 0.3 * bm25_score
                for vec_score, bm25_score in zip(vector_scores, bm25_scores_norm)
            ]
            
            # Sort by combined score
            sorted_pairs = sorted(
                zip(chunks, combined_scores),
                key=lambda x: x[1],
                reverse=True
            )
            
            reranked_chunks, reranked_scores = zip(*sorted_pairs)
            
            logger.debug(f"  Reranked {len(chunks)} chunks")
            
            return list(reranked_chunks), list(reranked_scores)
            
        except ImportError:
            logger.warning("rank_bm25 not installed, skipping reranking")
            return chunks, vector_scores
        except Exception as e:
            logger.warning(f"Reranking failed: {e}, using original order")
            return chunks, vector_scores
    
    def index_chunk(self, chunk: LogChunk) -> bool:
        """
        Index a single chunk (add to vector store + database)
        
        Args:
            chunk: LogChunk to index
            
        Returns:
            True if successful
        """
        try:
            # Step 1: Add to database
            db.add_chunk(chunk)
            
            # Step 2: Embed the content
            vector = self.embedder.embed_text(chunk.content)
            
            # Step 3: Add to vector store
            faiss_id = self.vector_store.add_single(vector, chunk.id)
            
            # Step 4: Update chunk with embedding ID
            chunk.embedding_id = faiss_id
            
            logger.debug(f"Indexed chunk {chunk.id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to index chunk: {e}")
            return False
    
    def index_chunks_batch(self, chunks: List[LogChunk]) -> int:
        """
        Index multiple chunks at once (much faster)
        
        Args:
            chunks: List of chunks to index
            
        Returns:
            Number of successfully indexed chunks
        """
        if not chunks:
            return 0
        
        try:
            logger.info(f"Batch indexing {len(chunks)} chunks...")
            
            # Step 1: Add to database
            db.add_chunks_batch(chunks)
            
            # Step 2: Embed all contents
            contents = [chunk.content for chunk in chunks]
            vectors = self.embedder.embed_batch(contents)
            
            # Step 3: Add to vector store
            chunk_ids = [chunk.id for chunk in chunks]
            faiss_ids = self.vector_store.add(vectors, chunk_ids)
            
            # Step 4: Update chunks with embedding IDs
            for chunk, faiss_id in zip(chunks, faiss_ids):
                chunk.embedding_id = faiss_id
            
            logger.info(f"✅ Indexed {len(chunks)} chunks")
            
            return len(chunks)
            
        except Exception as e:
            logger.error(f"Batch indexing failed: {e}")
            return 0
    
    def search_similar(
        self,
        text: str,
        top_k: int = 5
    ) -> List[Tuple[LogChunk, float]]:
        """
        Find similar log entries to given text
        
        This is like query() but simpler - just returns chunks, no LLM.
        
        Args:
            text: Text to find similar logs for
            top_k: Number of results
            
        Returns:
            List of (chunk, score) tuples
        """
        # Embed the text
        vector = self.embedder.embed_text(text)
        
        # Search
        chunk_ids, scores = self.vector_store.search(vector, k=top_k)
        
        # Retrieve chunks
        chunks = db.get_chunks_by_ids(chunk_ids)
        
        return list(zip(chunks, scores))
    
    def get_stats(self) -> dict:
        """Get statistics about the RAG system"""
        return {
            'total_chunks': len(self.vector_store),
            'embedding_model': self.embedder.model_name,
            'embedding_dimension': self.embedder.dimension,
            'llm_model': self.llm.model,
            'vector_store_stats': self.vector_store.get_stats(),
            'database_stats': db.get_source_stats()
        }
    
    def save_index(self):
        """Save vector store to disk"""
        logger.info("Saving vector index...")
        self.vector_store.save()
        logger.info("✅ Index saved")
    
    def __repr__(self):
        return f"<RAGService(vectors={len(self.vector_store)}, llm={self.llm.model})>"


# ===== SINGLETON INSTANCE =====
_rag_service = None


def get_rag_service() -> RAGService:
    """
    Get the global RAG service instance
    Lazy-loads on first call
    """
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service