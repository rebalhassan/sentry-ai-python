# sentry/services/rag.py
"""
RAG (Retrieval Augmented Generation) service
This is the brain that connects all AI components together
"""

import time
import logging
from typing import List, Tuple, Optional
from datetime import datetime

from ..core.models import LogChunk, LogSource, SourceType, QueryResult
from ..core.database import db
from ..core.config import settings
from ..core.security import sanitize_query
            
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
        chat_context: str = None,  # Task 2: Chat history context
        top_k: int = None,
        similarity_threshold: float = None,
        use_reranking: bool = None
    ) -> QueryResult:
        """
        Answer a query using RAG
        
        This is the main method - everything flows through here.
        
        Args:
            query_text: User's question
            chat_context: Previous conversation context for continuity
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
        
        # Step 0: Sanitize input for security
        query_text = sanitize_query(query_text)
        
        if not query_text:
            return QueryResult(
                answer="Please provide a valid query.",
                sources=[],
                confidence=0.0,
                query_time=time.time() - start_time,
                chunk_ids=[]
            )
        
        # Use defaults from settings if not provided
        top_k = top_k or settings.top_k_results
        similarity_threshold = similarity_threshold or settings.similarity_threshold
        use_reranking = use_reranking if use_reranking is not None else settings.use_reranking
        
        logger.info(f"RAG Query: '{query_text}'")
        logger.info(f"  top_k={top_k}, threshold={similarity_threshold}")
        if chat_context:
            logger.info(f"  with chat context ({len(chat_context)} chars)")
        
        try:
            # Step 1: Expand the query (if it's a natural language question)
            # We keep the original query for the final LLM answer, but use expanded for search
            expanded_query = self.llm.expand_query(query_text)
            
            # Step 2: Embed the query (using cached embedding for repeated queries)
            logger.debug("Step 2: Embedding query...")
            query_vector = self.embedder.embed_text_cached(expanded_query)
            
            # Step 3: Search vector store
            logger.debug("Step 3: Searching vector store...")
            chunk_ids, scores = self.vector_store.search_with_threshold(
                query_vector,
                k=top_k,
                threshold=similarity_threshold
            )
            
            # Task 3: If no chunks found, fall back to LLM inference without context
            if not chunk_ids:
                logger.warning("No relevant chunks found - falling back to LLM inference")
                
                # Build prompt with chat context if available
                fallback_prompt = query_text
                if chat_context:
                    fallback_prompt = f"{chat_context}\n\nCurrent question: {query_text}"
                
                # Generate answer without log context
                fallback_system_prompt = settings.main_prompt

                answer = self.llm.generate(
                    prompt=fallback_prompt,
                    system_prompt=fallback_system_prompt
                )
                
                return QueryResult(
                    answer=answer,
                    sources=[],
                    confidence=0.0,  # Low confidence since no logs used
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
            
            # Task 2: Include chat context in the generation
            answer = self.llm.generate_with_context(
                query_text, 
                chunks,
                chat_context=chat_context  # Pass chat context
            )
            
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

            # Step 0: Ensure all sources exist (fix for foreign key constraint)
            # Extract unique source IDs from chunks
            unique_source_ids = set(chunk.source_id for chunk in chunks)
            
            for source_id in unique_source_ids:
                # Check if source exists
                existing_source = db.get_source(source_id)
                
                if not existing_source:
                    # Create the source automatically
                    logger.info(f"Creating missing source: {source_id}")
                    
                    # Determine source type and name from source_id pattern
                    if source_id.startswith("evt_"):
                        source_type = SourceType.EVENTVIEWER
                        # Extract log name (e.g., "evt_System" -> "System")
                        log_name = source_id[4:]  # Remove "evt_" prefix
                        name = f"{log_name} Event Log"
                        eventlog_name = log_name
                        path = None
                    elif source_id.startswith("file_"):
                        source_type = SourceType.FILE
                        # Extract file name (e.g., "file_mylog.txt" -> "mylog.txt")
                        file_name = source_id[5:]  # Remove "file_" prefix
                        name = file_name
                        eventlog_name = None
                        path = None  # We don't have the actual path here
                    else:
                        # Default to FILE type
                        source_type = SourceType.FILE
                        name = source_id
                        eventlog_name = None
                        path = None
                    
                    # Create the source
                    source = LogSource(
                        id=source_id,
                        name=name,
                        source_type=source_type,
                        path=path,
                        eventlog_name=eventlog_name
                    )
                    
                    try:
                        db.add_source(source)
                        logger.info(f"✅ Created source: {name}")
                    except Exception as e:
                        logger.warning(f"Failed to create source {source_id}: {e}")
                        # Continue anyway - might be a race condition
            
            
            # Step 1: Add to database (store original content)
            db.add_chunks_batch(chunks)

            # Step 2: Helix Vector annotation
            # Annotate chunks with cluster_id, anomaly_type, severity_weight
            # This is needed when chunks bypass the LogIndexer (e.g., from Streamlit)
            if settings.helix_enabled:
                try:
                    from .helix import get_helix_service
                    helix = get_helix_service()
                    chunks = helix.annotate_chunks(chunks)
                    logger.info(f"Helix: {helix.get_stats()['cluster_count']} clusters, "
                               f"{sum(1 for c in chunks if c.is_anomaly)} anomalies")
                except Exception as e:
                    logger.warning(f"Helix annotation failed: {e}")
            
            # Step 3: Prepare contents for embedding
            contents_to_embed = []
            for chunk in chunks:
                # Use raw content for embedding
                # Helix annotations are stored in chunk fields, not embedded
                contents_to_embed.append(chunk.content)

            # Step 3: Embed all contents
            vectors = self.embedder.embed_batch(contents_to_embed)

            # Step 4: Add to vector store
            chunk_ids = [chunk.id for chunk in chunks]
            faiss_ids = self.vector_store.add(vectors, chunk_ids)

            # Step 5: Update chunks with embedding IDs
            for chunk, faiss_id in zip(chunks, faiss_ids):
                chunk.embedding_id = faiss_id
            
            logger.info(f"Indexed {len(chunks)} chunks")
            
            return len(chunks)
            
        except ValueError as e:
            # Specific handling for dimension mismatch
            if "dimension" in str(e).lower():
                logger.error(f"❌ Dimension mismatch detected!")
                logger.error(f"   This usually means you changed the embedding model.")
                logger.error(f"   Solution: Delete the old vector index and re-index:")
                logger.error(f"   rm ~/.sentry-ai/vectors.faiss*")
                logger.error(f"   Full error: {e}")
            else:
                logger.error(f"Batch indexing failed: {e}", exc_info=True)
            return 0
        except Exception as e:
            logger.error(f"Batch indexing failed: {e}", exc_info=True)
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
    
    def search_anomalies(
        self,
        text: str = None,
        top_k: int = 10,
        min_severity: float = 0.0
    ) -> List[Tuple[LogChunk, float]]:
        """
        Search for anomalous log entries.
        
        Filters results to only return chunks flagged as anomalies by
        the Helix Vector system.
        
        Args:
            text: Optional search text (if None, returns all anomalies)
            top_k: Maximum number of results
            min_severity: Minimum severity weight to include
            
        Returns:
            List of (chunk, score) tuples, only anomalous entries
        """
        try:
            anomalies = []
            
            if text:
                # Search with text, then filter anomalies
                results = self.search_similar(text, top_k=top_k * 3)
                for chunk, score in results:
                    # Safe attribute access for chunks that may not have Helix annotations
                    if getattr(chunk, 'is_anomaly', False):
                        severity = getattr(chunk, 'severity_weight', 0.0)
                        if severity >= min_severity:
                            anomalies.append((chunk, score))
            else:
                # Get all chunks from vector store and filter
                # Use search_similar with empty query as fallback
                try:
                    all_chunks = db.get_recent_chunks(limit=top_k * 3)
                except AttributeError:
                    # db.get_recent_chunks may not exist - use empty list
                    logger.warning("db.get_recent_chunks not available, using empty results")
                    all_chunks = []
                
                for chunk in all_chunks:
                    if getattr(chunk, 'is_anomaly', False):
                        severity = getattr(chunk, 'severity_weight', 0.0)
                        if severity >= min_severity:
                            score = getattr(chunk, 'anomaly_score', 0.0)
                            anomalies.append((chunk, score))
            
            # Sort by anomaly score (most anomalous first)
            anomalies.sort(key=lambda x: getattr(x[0], 'anomaly_score', 0.0), reverse=True)
            
            return anomalies[:top_k]
            
        except Exception as e:
            logger.error("Error in search_anomalies: %s", e)
            return []
    
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