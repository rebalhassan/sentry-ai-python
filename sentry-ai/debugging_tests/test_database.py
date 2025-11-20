# test_database.py
"""Test the database layer"""

from datetime import datetime
from sentry.core import LogSource, LogChunk, ChatMessage, SourceType, LogLevel
from sentry.core.database import Database


def test_database():
    print("🧪 Testing Database Layer\n")
    
    # Create a test database in memory
    db = Database(db_path=":memory:")  # In-memory DB for testing
    
    # ===== TEST 1: Add a source =====
    print("📁 TEST 1: Adding a log source...")
    source = LogSource(
        name="System Logs",
        source_type=SourceType.EVENTVIEWER,
        eventlog_name="System"
    )
    db.add_source(source)
    print(f"   ✅ Added source: {source.name} (ID: {source.id[:8]}...)")
    
    # Retrieve it
    retrieved = db.get_source(source.id)
    assert retrieved.name == "System Logs"
    print(f"   ✅ Retrieved source: {retrieved.name}")
    
    # ===== TEST 2: Add chunks =====
    print("\n📝 TEST 2: Adding log chunks...")
    chunks = [
        LogChunk(
            source_id=source.id,
            content=f"ERROR: Test error #{i}",
            timestamp=datetime.now(),
            log_level=LogLevel.ERROR,
            metadata={"test": True},
            embedding_id=i
        )
        for i in range(10)
    ]
    db.add_chunks_batch(chunks)
    print(f"   ✅ Added {len(chunks)} chunks in batch")
    
    # Retrieve them
    chunk_ids = [c.id for c in chunks[:3]]
    retrieved_chunks = db.get_chunks_by_ids(chunk_ids)
    print(f"   ✅ Retrieved {len(retrieved_chunks)} chunks by ID")
    
    # Get by embedding IDs
    embedding_chunks = db.get_chunks_by_embedding_ids([0, 1, 2])
    print(f"   ✅ Retrieved {len(embedding_chunks)} chunks by embedding ID")
    
    # ===== TEST 3: Chat history =====
    print("\n💬 TEST 3: Chat history...")
    messages = [
        ChatMessage(role="user", content="What are the errors?"),
        ChatMessage(role="assistant", content="Found 10 errors", sources=chunk_ids)
    ]
    for msg in messages:
        db.add_chat_message(msg)
    print(f"   ✅ Added {len(messages)} messages")
    
    history = db.get_chat_history(limit=10)
    print(f"   ✅ Retrieved {len(history)} messages from history")
    
    # ===== TEST 4: Statistics =====
    print("\n📊 TEST 4: Database statistics...")
    stats = db.get_source_stats()
    print(f"   Total sources: {stats.get('total_sources', 0)}")
    print(f"   Active sources: {stats.get('active_sources', 0)}")
    print(f"   Total chunks: {stats.get('total_chunks', 0)}")
    print(f"   EventViewer sources: {stats.get('eventviewer_sources', 0)}")
    
    chunk_count = db.get_chunk_count()
    print(f"   ✅ Total chunks in DB: {chunk_count}")
    
    # ===== TEST 5: Search =====
    print("\n🔍 TEST 5: Text search...")
    results = db.search_chunks_by_text("error #3")
    print(f"   ✅ Found {len(results)} matching chunks")
    if results:
        print(f"   Sample: {results[0].content}")
    
    # ===== TEST 6: Update source stats =====
    print("\n📈 TEST 6: Update source stats...")
    db.update_source_stats(source.id, 10, datetime.now())
    updated_source = db.get_source(source.id)
    print(f"   ✅ Updated source stats: {updated_source.total_chunks} chunks")
    
    print("\n🔥 All database tests passed!")


if __name__ == "__main__":
    test_database()