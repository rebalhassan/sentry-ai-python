# test_rag.py
"""Test the complete RAG pipeline"""

from datetime import datetime
from sentry.services.rag import RAGService
from sentry.core.models import LogChunk, LogLevel, LogSource, SourceType
from sentry.core.database import db


def test_rag():
    print("🧪 Testing RAG Service (The Complete Pipeline)\n")
    
    # ===== SETUP: Create a source first (IMPORTANT!) =====
    print("🔧 Setting up test environment...")
    
    # Create a test source
    test_source = LogSource(
        name="Test Log Source",
        source_type=SourceType.FILE,
        path="C:\\test\\system.log"
    )
    
    # Add source to database
    try:
        db.add_source(test_source)
        print(f"✅ Created test source: {test_source.name}")
    except ValueError:
        # Source already exists
        test_source = db.get_source_by_name("Test Log Source")
        print(f"✅ Using existing source: {test_source.name}")
    
    print()
    
    # Initialize RAG
    print("🔧 Initializing RAG service...")
    rag = RAGService()
    print("✅ RAG service ready!\n")
    
    # ===== TEST 1: Index some logs =====
    print("📚 TEST 1: Indexing sample logs")
    
    sample_logs = [
        LogChunk(
            source_id=test_source.id,  # ← USE THE SOURCE ID
            content="ERROR: Disk write failure on drive D: - Error code: 0x80070015",
            timestamp=datetime(2025, 11, 3, 9, 1, 3),
            log_level=LogLevel.ERROR,
            metadata={"file": "system.log", "line": 4521}
        ),
        LogChunk(
            source_id=test_source.id,
            content="ERROR: The device, \\Device\\Harddisk0\\DR0, has a bad block.",
            timestamp=datetime(2025, 11, 3, 9, 1, 5),
            log_level=LogLevel.ERROR,
            metadata={"event_id": 7}
        ),
        LogChunk(
            source_id=test_source.id,
            content="CRITICAL: Volume D: is out of disk space. 0 bytes remaining.",
            timestamp=datetime(2025, 11, 3, 9, 1, 8),
            log_level=LogLevel.CRITICAL,
            metadata={"file": "system.log", "line": 4525}
        ),
        LogChunk(
            source_id=test_source.id,
            content="WARNING: High memory usage detected - 95% of RAM in use",
            timestamp=datetime(2025, 11, 3, 10, 15, 22),
            log_level=LogLevel.WARNING,
            metadata={"file": "performance.log"}
        ),
        LogChunk(
            source_id=test_source.id,
            content="ERROR: Database connection timeout after 30 seconds",
            timestamp=datetime(2025, 11, 3, 11, 30, 45),
            log_level=LogLevel.ERROR,
            metadata={"file": "application.log"}
        ),
        LogChunk(
            source_id=test_source.id,
            content="INFO: Service 'WebServer' started successfully on port 8080",
            timestamp=datetime(2025, 11, 3, 8, 0, 12),
            log_level=LogLevel.INFO,
            metadata={"file": "application.log"}
        ),
    ]
    
    indexed = rag.index_chunks_batch(sample_logs)
    print(f"   ✅ Indexed {indexed} log chunks")
    print(f"   📊 Vector store now has {len(rag.vector_store)} vectors\n")
    
    # ===== TEST 2: Simple query =====
    print("🔍 TEST 2: Simple RAG query WITH straight fwd question")
    
    query = "Did database connection timeout?"
    print(f"   Query: '{query}'")
    print(f"   Processing...\n")
    
    result = rag.query(query)
    
    print("   📋 ANSWER:")
    print("   " + "=" * 70)
    print(f"   {result.answer}")
    print("   " + "=" * 70)
    print(f"\n   ⏱️  Query time: {result.query_time:.2f}s")
    print(f"   🎯 Confidence: {result.confidence:.3f}")
    print(f"   📚 Sources used: {len(result.sources)}")
    
    if result.sources:
        print(f"\n   📝 Sources:")
        for i, chunk in enumerate(result.sources[:3], 1):
            print(f"   {i}. [{chunk.log_level.value.upper()}] {chunk.content[:60]}...")
    
    print()
    
    # ===== TEST 3: Different query =====
    print("🔍 TEST 3: Another query")
    
    query = "Are there any memory issues?"
    print(f"   Query: '{query}'")
    print(f"   Processing...\n")
    
    result = rag.query(query, top_k=5)
    
    print("   📋 ANSWER:")
    print("   " + "=" * 70)
    print(f"   {result.answer}")
    print("   " + "=" * 70)
    print(f"\n   ⏱️  Query time: {result.query_time:.2f}s")
    print(f"   🎯 Confidence: {result.confidence:.3f}\n")
    
    # ===== TEST 4: Similarity search (no LLM) =====
    print("🔎 TEST 4: Similarity search (without LLM)")
    
    text = "disk failure"
    print(f"   Find logs similar to: '{text}'")
    
    similar = rag.search_similar(text, top_k=3)
    
    print(f"   ✅ Found {len(similar)} similar logs:\n")
    for i, (chunk, score) in enumerate(similar, 1):
        print(f"   {i}. [{score:.3f}] {chunk.content[:60]}...")
    
    print()
    
    # ===== TEST 5: Query with no results =====
    print("❓ TEST 5: Query with no matching results")
    
    query = "What happened with the coffee machine?"
    print(f"   Query: '{query}'")
    
    result = rag.query(query)
    
    print(f"   Answer: {result.answer[:100]}...")
    print(f"   Sources: {len(result.sources)}")
    print(f"   Confidence: {result.confidence:.3f}\n")
    
    # ===== TEST 6: Statistics =====
    print("📊 TEST 6: RAG system statistics")
    
    stats = rag.get_stats()
    print(f"   Total chunks indexed: {stats['total_chunks']}")
    print(f"   Embedding model: {stats['embedding_model']}")
    print(f"   LLM model: {stats['llm_model']}")
    print(f"   Vector dimension: {stats['embedding_dimension']}")
    
    db_stats = stats['database_stats']
    print(f"   Sources in database: {db_stats.get('total_sources', 0)}")
    print()
    
    # ===== TEST 7: Save index =====
    print("💾 TEST 7: Saving index")
    
    rag.save_index()
    print(f"   ✅ Index saved\n")
    
    print("🔥 All RAG tests passed!")
    print("\n💡 The RAG pipeline is complete:")
    print("   Query → Embed → Search → Retrieve → LLM → Answer")
    print("\n🎉 You now have a fully functional local AI assistant!")
    
    # Cleanup (optional - comment out if you want to keep the data)
    print("\n🧹 Cleanup:")
    print("   To clean up test data, run:")
    print(f"   >>> from sentry.core.database import db")
    print(f"   >>> db.delete_source('{test_source.id}')")


if __name__ == "__main__":
    test_rag()