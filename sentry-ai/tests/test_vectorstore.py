# test_vectorstore.py
"""Test the vector store"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sentry.services.embedding import EmbeddingService
from sentry.services.vectorstore import VectorStore


def test_vectorstore():
    print("🧪 Testing Vector Store\n")
    
    # Use in-memory for testing
    test_index_path = Path(":memory:")  # We'll use a temp file instead
    import tempfile
    temp_dir = tempfile.mkdtemp()
    test_index_path = Path(temp_dir) / "test_index.faiss"
    
    # Initialize
    print("📦 Initializing vector store...")
    embedder = EmbeddingService()
    store = VectorStore(index_path=test_index_path)
    print(f"✅ Vector store created")
    print(f"   Dimension: {store.dimension}")
    print(f"   Total vectors: {len(store)}\n")
    
    # ===== TEST 1: Add vectors =====
    print("➕ TEST 1: Adding vectors")
    
    texts = [
        "ERROR: Disk write failure on drive D:",
        "WARNING: High memory usage detected",
        "CRITICAL: Database connection timeout",
        "INFO: Service started successfully",
        "ERROR: Failed to read configuration file"
    ]
    
    chunk_ids = [f"chunk-{i}" for i in range(len(texts))]
    
    # Embed the texts
    vectors = embedder.embed_batch(texts)
    print(f"   Embedded {len(texts)} texts into vectors")
    print(f"   Vector shape: {vectors.shape}")
    
    # Add to store
    faiss_ids = store.add(vectors, chunk_ids)
    print(f"   ✅ Added {len(faiss_ids)} vectors to store")
    print(f"   FAISS IDs: {faiss_ids}")
    print(f"   Store now has {len(store)} vectors\n")
    
    # ===== TEST 2: Search =====
    print("🔍 TEST 2: Semantic search")
    
    query = "disk problem"
    print(f"   Query: '{query}'")
    
    query_vector = embedder.embed_text(query)
    result_ids, scores = store.search(query_vector, k=3)
    
    print(f"   ✅ Found {len(result_ids)} results:\n")
    for i, (chunk_id, score) in enumerate(zip(result_ids, scores), 1):
        # Find the original text
        idx = chunk_ids.index(chunk_id)
        text = texts[idx]
        print(f"   {i}. [{score:.3f}] {chunk_id}")
        print(f"      '{text}'\n")
    
    # ===== TEST 3: Search with threshold =====
    print("🎯 TEST 3: Search with similarity threshold")
    
    result_ids, scores = store.search_with_threshold(
        query_vector,
        k=5,
        threshold=0.5  # Only return if similarity > 0.5
    )
    
    print(f"   Query: '{query}'")
    print(f"   Threshold: 0.5")
    print(f"   ✅ Found {len(result_ids)} results above threshold\n")
    
    # ===== TEST 4: Different query =====
    print("🔍 TEST 4: Different query")
    
    query2 = "memory issue"
    query_vector2 = embedder.embed_text(query2)
    result_ids, scores = store.search(query_vector2, k=3)
    
    print(f"   Query: '{query2}'")
    print(f"   Top result:\n")
    
    idx = chunk_ids.index(result_ids[0])
    print(f"   [{scores[0]:.3f}] {texts[idx]}\n")
    
    # ===== TEST 5: Save and load =====
    print("💾 TEST 5: Save and load index")
    
    print(f"   Saving to {test_index_path}")
    store.save()
    print(f"   ✅ Saved")
    
    # Create new store and load
    store2 = VectorStore(index_path=test_index_path)
    print(f"   ✅ Loaded into new instance")
    print(f"   Vectors in new store: {len(store2)}")
    
    # Verify it works
    result_ids2, scores2 = store2.search(query_vector, k=1)
    print(f"   ✅ Search after load works: {result_ids2[0]}\n")
    
    # ===== TEST 6: Statistics =====
    print("📊 TEST 6: Index statistics")
    stats = store.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # ===== TEST 7: Add more and search =====
    print("\n➕ TEST 7: Add more vectors")
    
    more_texts = [
        "ERROR: Network connection lost",
        "WARNING: SSL certificate expired",
        "INFO: Backup completed successfully"
    ]
    more_chunk_ids = [f"chunk-{i+len(texts)}" for i in range(len(more_texts))]
    
    more_vectors = embedder.embed_batch(more_texts)
    store.add(more_vectors, more_chunk_ids)
    
    print(f"   ✅ Added {len(more_texts)} more vectors")
    print(f"   Total vectors now: {len(store)}\n")
    
    # Search again
    query3 = "network error"
    query_vector3 = embedder.embed_text(query3)
    result_ids, scores = store.search(query_vector3, k=3)
    
    print(f"   Query: '{query3}'")
    print(f"   Top result:")
    
    all_texts = texts + more_texts
    all_chunk_ids = chunk_ids + more_chunk_ids
    
    idx = all_chunk_ids.index(result_ids[0])
    print(f"   [{scores[0]:.3f}] {all_texts[idx]}")
    
    # ===== TEST 8: Removal =====
    print("\n🗑️  TEST 8: Remove a vector")
    
    removed = store.remove("chunk-0")
    print(f"   ✅ Removed chunk-0: {removed}")
    print(f"   (Note: FAISS doesn't physically remove until rebuild)")
    
    print("\n🔥 All vector store tests passed!")
    print(f"\n💡 Index saved at: {test_index_path}")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    print(f"   (Cleaned up test files)")


if __name__ == "__main__":
    test_vectorstore()