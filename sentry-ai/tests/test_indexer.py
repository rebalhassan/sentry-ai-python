# test_indexer.py
"""
Test the Log Indexer Service
Tests parsing of .log, .txt, .csv files and chunking logic
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import tempfile
from datetime import datetime

from sentry.core import LogLevel, settings
from sentry.services.indexer import LogIndexer


def test_log_level_detection():
    """Test that log levels are correctly detected from text"""
    print("\n🧪 Testing Log Level Detection\n")
    
    indexer = LogIndexer()
    
    test_cases = [
        ("ERROR: Database connection failed", LogLevel.ERROR),
        ("Warning: Low disk space", LogLevel.WARNING),
        ("INFO: Server started successfully", LogLevel.INFO),
        ("CRITICAL: System failure imminent", LogLevel.CRITICAL),
        ("DEBUG: Variable x = 42", LogLevel.DEBUG),
        ("This is just normal text", LogLevel.UNKNOWN),
        ("[2025-11-20 00:00:00] FATAL error occurred", LogLevel.CRITICAL),
        ("Server responded with FAIL code", LogLevel.ERROR),
    ]
    
    for text, expected_level in test_cases:
        detected_level = indexer._detect_log_level(text)
        status = "✅" if detected_level == expected_level else "❌"
        print(f"{status} '{text[:50]}...' → {detected_level} (expected: {expected_level})")
        assert detected_level == expected_level, f"Expected {expected_level}, got {detected_level}"
    
    print("\n✅ All log level detection tests passed!")


def test_timestamp_extraction():
    """Test that timestamps are correctly extracted from text"""
    print("\n🧪 Testing Timestamp Extraction\n")
    
    indexer = LogIndexer()
    
    test_cases = [
        "2025-11-20 00:00:00 ERROR: Something failed",
        "[2025/11/20 00:00:00] Warning message",
        "Nov 20 00:00:00 Server started",
        "2025-11-20T00:00:00.123Z Request received",
    ]
    
    for text in test_cases:
        timestamp = indexer._extract_timestamp(text)
        status = "✅" if timestamp else "❌"
        print(f"{status} '{text[:50]}...' → {timestamp}")
        assert timestamp is not None, f"Failed to extract timestamp from: {text}"
    
    # Test case with no timestamp
    no_timestamp_text = "This text has no timestamp at all"
    timestamp = indexer._extract_timestamp(no_timestamp_text)
    assert timestamp is None, "Should return None for text without timestamp"
    print(f"✅ No timestamp text → None (correct)")
    
    print("\n✅ All timestamp extraction tests passed!")


def test_content_chunking():
    """Test that content is correctly chunked with overlap"""
    print("\n🧪 Testing Content Chunking\n")
    
    indexer = LogIndexer()
    
    # Create test content (longer than chunk_size)
    content = "A" * 1000  # 1000 characters
    chunks = indexer._chunk_content(content)
    
    print(f"📊 Content length: {len(content)} chars")
    print(f"📊 Chunk size: {settings.chunk_size} chars")
    print(f"📊 Chunk overlap: {settings.chunk_overlap} chars")
    print(f"📊 Number of chunks created: {len(chunks)}")
    
    # Verify chunks
    assert len(chunks) > 1, "Should create multiple chunks for long content"
    
    # Check first chunk size
    assert len(chunks[0]) == settings.chunk_size, f"First chunk should be {settings.chunk_size} chars"
    
    # Check overlap (last chars of chunk[0] should match first chars of chunk[1])
    if len(chunks) > 1:
        overlap_content = chunks[0][-settings.chunk_overlap:]
        next_start = chunks[1][:settings.chunk_overlap]
        assert overlap_content == next_start, "Chunks should overlap correctly"
        print("✅ Overlap verification: Last 50 chars of chunk[0] == First 50 chars of chunk[1]")
    
    # Test empty content
    empty_chunks = indexer._chunk_content("")
    assert empty_chunks == [], "Empty content should return empty list"
    print("✅ Empty content handling: Returns empty list")
    
    # Test short content (less than min_chunk_size)
    short_content = "A" * (settings.min_chunk_size - 10)
    short_chunks = indexer._chunk_content(short_content)
    assert len(short_chunks) == 1, "Short content should create 1 chunk"
    print(f"✅ Short content ({len(short_content)} chars): Creates 1 chunk")
    
    print("\n✅ All content chunking tests passed!")


def test_text_file_parsing():
    """Test parsing of .log and .txt files"""
    print("\n🧪 Testing Text File Parsing\n")
    
    indexer = LogIndexer()
    
    # Create a temporary log file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False, encoding='utf-8') as f:
        log_content = """2025-11-20 00:00:01 ERROR: Database connection timeout
2025-11-20 00:00:02 WARNING: Retry attempt 1 of 3
2025-11-20 00:00:03 ERROR: Database connection timeout
2025-11-20 00:00:04 WARNING: Retry attempt 2 of 3
2025-11-20 00:00:05 INFO: Connection established successfully
2025-11-20 00:00:06 INFO: Application started
"""
        f.write(log_content)
        temp_file = Path(f.name)
    
    try:
        # Parse the file
        chunks = indexer.parse_file(temp_file, source_id="test-source-123")
        
        print(f"📄 Parsed file: {temp_file.name}")
        print(f"📊 File size: {len(log_content)} chars")
        print(f"📊 Chunks created: {len(chunks)}")
        
        # Verify chunks
        assert len(chunks) > 0, "Should create at least one chunk"
        
        # Check first chunk
        first_chunk = chunks[0]
        print(f"\n📦 First Chunk:")
        print(f"   ID: {first_chunk.id[:8]}...")
        print(f"   Source ID: {first_chunk.source_id}")
        print(f"   Log Level: {first_chunk.log_level}")
        print(f"   Timestamp: {first_chunk.timestamp}")
        print(f"   Content length: {len(first_chunk.content)} chars")
        print(f"   Content preview: {first_chunk.content[:100]}...")
        
        # Verify structure
        assert first_chunk.source_id == "test-source-123", "Source ID should match"
        assert first_chunk.log_level in [LogLevel.ERROR, LogLevel.WARNING, LogLevel.INFO], "Should detect log level"
        assert 'file' in first_chunk.metadata, "Metadata should include file path"
        assert 'chunk_index' in first_chunk.metadata, "Metadata should include chunk index"
        
        print("\n✅ Text file parsing test passed!")
        
    finally:
        # Clean up
        temp_file.unlink()


def test_csv_file_parsing():
    """Test parsing of .csv files"""
    print("\n🧪 Testing CSV File Parsing\n")
    
    indexer = LogIndexer()
    
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        csv_content = """timestamp,level,message
2025-11-20 00:00:01,ERROR,Database connection timeout
2025-11-20 00:00:02,WARNING,Retry attempt 1 of 3
2025-11-20 00:00:03,ERROR,Database connection timeout
2025-11-20 00:00:04,INFO,Connection established
"""
        f.write(csv_content)
        temp_file = Path(f.name)
    
    try:
        # Parse the CSV file
        chunks = indexer.parse_file(temp_file, source_id="test-csv-source")
        
        print(f"📄 Parsed CSV file: {temp_file.name}")
        print(f"📊 Chunks created: {len(chunks)}")
        
        # Verify chunks
        assert len(chunks) > 0, "Should create at least one chunk"
        
        # Check first chunk
        first_chunk = chunks[0]
        print(f"\n📦 First Chunk:")
        print(f"   Log Level: {first_chunk.log_level}")
        print(f"   Content preview: {first_chunk.content[:150]}...")
        
        # Verify structure
        assert first_chunk.source_id == "test-csv-source", "Source ID should match"
        assert 'format' in first_chunk.metadata, "Metadata should indicate CSV format"
        assert first_chunk.metadata['format'] == 'csv', "Format should be 'csv'"
        
        print("\n✅ CSV file parsing test passed!")
        
    finally:
        # Clean up
        temp_file.unlink()


def test_folder_parsing():
    """Test recursive folder parsing"""
    print("\n🧪 Testing Folder Parsing\n")
    
    indexer = LogIndexer()
    
    # Create a temporary directory with multiple log files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create multiple log files
        log_files = []
        for i in range(3):
            log_file = temp_path / f"test_{i}.log"
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"2025-11-20 00:00:0{i} ERROR: Test error {i}\n")
                f.write(f"2025-11-20 00:00:0{i} INFO: Test info {i}\n")
            log_files.append(log_file)
        
        # Create a subdirectory with another log file
        sub_dir = temp_path / "subdir"
        sub_dir.mkdir()
        sub_log = sub_dir / "sub_test.log"
        with open(sub_log, 'w', encoding='utf-8') as f:
            f.write("2025-11-20 00:00:10 WARNING: Subdirectory warning\n")
        log_files.append(sub_log)
        
        # Parse the folder
        chunks = indexer.parse_folder(temp_path, source_id="test-folder-source")
        
        print(f"📁 Parsed folder: {temp_path}")
        print(f"📊 Total files: {len(log_files)}")
        print(f"📊 Total chunks created: {len(chunks)}")
        
        # Verify chunks were created from multiple files
        assert len(chunks) >= len(log_files), "Should create chunks from all files"
        
        # Verify different files are represented
        file_paths = set()
        for chunk in chunks:
            if 'file' in chunk.metadata:
                file_paths.add(chunk.metadata['file'])
        
        print(f"📊 Unique files processed: {len(file_paths)}")
        assert len(file_paths) == len(log_files), "Should process all log files"
        
        print("\n✅ Folder parsing test passed!")


def test_windows_eventviewer_format():
    """Test parsing Windows Event Viewer log format (simulated)"""
    print("\n🧪 Testing Windows Event Viewer Format\n")
    
    indexer = LogIndexer()
    
    # Simulate a PyEventLogRecord (mock object)
    class MockEventLogRecord:
        def __init__(self):
            self.EventID = 1001
            self.TimeGenerated = datetime(2025, 11, 20, 0, 0, 0).timestamp()
            self.SourceName = "TestSource"
            self.EventType = 1  # ERROR type
            self.StringInserts = ["Test error message", "Additional details"]
            self.RecordNumber = 12345
            self.ComputerName = "TestPC"
    
    mock_record = MockEventLogRecord()
    
    # Parse the mock record
    chunk = indexer.parse_eventviewer_record(mock_record, source_id="eventviewer-system")
    
    print(f"📦 Event Viewer Chunk:")
    print(f"   Event ID: {chunk.metadata.get('event_id')}")
    print(f"   Source Name: {chunk.metadata.get('source_name')}")
    print(f"   Log Level: {chunk.log_level}")
    print(f"   Computer: {chunk.metadata.get('computer')}")
    print(f"   Content: {chunk.content[:100]}...")
    
    # Verify structure
    assert chunk.source_id == "eventviewer-system", "Source ID should match"
    assert chunk.log_level == LogLevel.ERROR, "Event Type 1 should map to ERROR"
    assert chunk.metadata['event_id'] == 1001, "Event ID should be preserved"
    assert chunk.metadata['source_name'] == "TestSource", "Source name should be preserved"
    assert "Test error message" in chunk.content, "Content should include message"
    
    print("\n✅ Windows Event Viewer format test passed!")


def test_settings_integration():
    """Test that indexer respects settings"""
    print("\n🧪 Testing Settings Integration\n")
    
    print(f"⚙️  Settings:")
    print(f"   Chunk size: {settings.chunk_size}")
    print(f"   Chunk overlap: {settings.chunk_overlap}")
    print(f"   Min chunk size: {settings.min_chunk_size}")
    print(f"   Max file size: {settings.max_file_size_mb} MB")
    print(f"   Supported extensions: {settings.supported_extensions}")
    
    indexer = LogIndexer()
    
    # Verify chunking uses settings
    content = "A" * (settings.chunk_size * 2)
    chunks = indexer._chunk_content(content)
    
    assert len(chunks[0]) == settings.chunk_size, "Should use settings.chunk_size"
    print(f"✅ Chunk size respects settings: {len(chunks[0])} == {settings.chunk_size}")
    
    print("\n✅ Settings integration test passed!")


def main():
    """Run all tests"""
    print("=" * 70)
    print("🚀 SENTRY-AI LOG INDEXER TEST SUITE")
    print("=" * 70)
    
    try:
        test_log_level_detection()
        test_timestamp_extraction()
        test_content_chunking()
        test_text_file_parsing()
        test_csv_file_parsing()
        test_folder_parsing()
        test_windows_eventviewer_format()
        test_settings_integration()
        
        print("\n" + "=" * 70)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 70)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        raise


if __name__ == "__main__":
    main()
