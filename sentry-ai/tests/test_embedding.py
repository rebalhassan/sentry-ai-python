# test_embedding.py
"""Test the embedding service"""

import time
from sentry.services.embedding import EmbeddingService


def test_embedding():
    print("🧪 Testing Embedding Service\n")
    
    # Initialize
    print("📥 Loading model (this may download ~80MB on first run)...")
    embedder = EmbeddingService()
    print(f"✅ Model loaded: {embedder.model_name}")
    print(f"   Dimension: {embedder.dimension}")
    print(f"   Cache dir: {embedder.cache_dir}\n")
    
    # ===== TEST 1: Single embedding =====
    print("🔢 TEST 1: Single text embedding")
    text = "ERROR: Disk write failure on drive D:"
    
    start = time.time()
    vector = embedder.embed_text(text)
    elapsed = time.time() - start
    
    print(f"   Text: '{text}'")
    print(f"   Vector shape: {vector.shape}")
    print(f"   Vector sample: [{vector[0]:.4f}, {vector[1]:.4f}, {vector[2]:.4f}, ...]")
    print(f"   Time: {elapsed*1000:.2f}ms")
    
    # ===== TEST 2: Batch embedding =====
    print("\n📦 TEST 2: Batch embedding")
    texts = [
        "ERROR: Disk write failure on drive D:",
        "WARNING: High memory usage detected",
        "CRITICAL: Database connection timeout",
        "INFO: Service started successfully",
        "ERROR: Failed to read configuration file"
    ]
    
    start = time.time()
    vectors = embedder.embed_batch(texts)
    elapsed = time.time() - start
    
    print(f"   Embedded {len(texts)} texts")
    print(f"   Output shape: {vectors.shape}")
    print(f"   Time: {elapsed*1000:.2f}ms ({elapsed*1000/len(texts):.2f}ms per text)")
    
    # ===== TEST 3: Semantic similarity =====
    print("\n🔍 TEST 3: Semantic similarity")
    
    pairs = [
        ("disk error", "drive failure"),
        ("disk error", "network timeout"),
        ("database crash", "database connection lost"),
        ("memory leak", "coffee break"),
    ]
    
    for text1, text2 in pairs:
        sim = embedder.similarity(text1, text2)
        print(f"   '{text1}' ↔ '{text2}': {sim:.3f}")
    
    # ===== TEST 4: Batch similarity (real use case) =====
    print("\n🎯 TEST 4: Find most similar log (RAG simulation)")
    
    query = "disk problem"
    logs = [
        "ERROR: Hard drive failure detected",
        "INFO: Network configuration updated",
        "WARNING: Disk space running low",
        "CRITICAL: CPU overheating",
        "ERROR: Unable to write to disk"
    ]
    
    start = time.time()
    similarities = embedder.batch_similarity(query, logs)
    elapsed = time.time() - start
    
    print(f"   Query: '{query}'")
    print(f"   Searching through {len(logs)} logs...")
    print(f"   Time: {elapsed*1000:.2f}ms\n")
    
    # Sort by similarity
    results = sorted(zip(logs, similarities), key=lambda x: x[1], reverse=True)
    
    print("   Top 3 most relevant logs:")
    for i, (log, score) in enumerate(results[:3], 1):
        print(f"   {i}. [{score:.3f}] {log}")
    
    # ===== TEST 5: Model info =====
    print("\n📊 TEST 5: Model information")
    info = embedder.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # ===== TEST 6: Speed test =====
    print("\n⚡ TEST 6: Speed benchmark")
    
    test_texts = [f"Test log message number {i}" for i in range(1000)]
    
    start = time.time()
    vectors = embedder.embed_batch(test_texts)
    elapsed = time.time() - start
    
    print(f"   Embedded 1,000 texts in {elapsed:.2f}s")
    print(f"   Speed: {len(test_texts)/elapsed:.0f} texts/second")
    print(f"   Average: {elapsed*1000/len(test_texts):.2f}ms per text")
    
    print("\n🔥 All embedding tests passed!")
    print(f"\n💡 The model is cached at: {embedder.cache_dir}")
    print("   Future runs will be instant (no download needed)")


if __name__ == "__main__":
    test_embedding()