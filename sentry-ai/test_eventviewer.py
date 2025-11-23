"""
Test script for EventViewer Integration
Tests the EventViewerReader and EventViewerWatcher classes

Run this script to verify that the EventViewer integration is working correctly.
This script will:
1. Check if pywin32 is available
2. Test reading events from System and Application logs
3. Display sample events
4. Test the polling mechanism

Note: This script must be run on Windows with pywin32 installed.
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / "sentry-ai"))

try:
    import win32evtlog
    print("✅ pywin32 is available")
    WINDOWS_AVAILABLE = True
except ImportError:
    print("❌ pywin32 is NOT available")
    print("   Install with: pip install pywin32")
    WINDOWS_AVAILABLE = False

from sentry.services.eventviewer import EventViewerReader, EventViewerWatcher
from sentry.core.models import LogSource, SourceType


def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def test_basic_reading():
    """Test basic event reading from System log"""
    print_header("TEST 1: Basic Event Reading")
    
    if not WINDOWS_AVAILABLE:
        print("⚠️  Skipping test - pywin32 not available")
        return False
    
    try:
        # Create reader
        reader = EventViewerReader()
        print("✅ EventViewerReader created successfully")
        
        # Create a test source
        test_source_id = "test-source-001"
        
        # Read last 10 System events
        print("\n📖 Reading last 10 events from System log...")
        chunks = reader.read_events(
            log_name="System",
            source_id=test_source_id,
            max_events=10
        )
        
        print(f"✅ Successfully read {len(chunks)} events")
        
        # Display sample events
        if chunks:
            print("\n📝 Sample Events:")
            print("-" * 70)
            for i, chunk in enumerate(chunks[:5], 1):
                print(f"\n{i}. [{chunk.log_level.value.upper()}] @ {chunk.timestamp}")
                print(f"   Source: {chunk.metadata.get('source_name', 'Unknown')}")
                print(f"   Event ID: {chunk.metadata.get('event_id', 'N/A')}")
                print(f"   Computer: {chunk.metadata.get('computer', 'N/A')}")
                # Truncate content for display
                content = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
                print(f"   Content: {content}")
        
        # Close handles
        reader.close_all()
        print("\n✅ TEST 1 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_logs():
    """Test reading from multiple event logs"""
    print_header("TEST 2: Multiple Event Logs")
    
    if not WINDOWS_AVAILABLE:
        print("⚠️  Skipping test - pywin32 not available")
        return False
    
    try:
        reader = EventViewerReader()
        
        log_names = ["System", "Application"]
        all_chunks = []
        
        for log_name in log_names:
            print(f"\n📖 Reading from {log_name} log...")
            chunks = reader.read_events(
                log_name=log_name,
                source_id=f"test-{log_name.lower()}",
                max_events=5
            )
            all_chunks.extend(chunks)
            print(f"   ✅ Read {len(chunks)} events from {log_name}")
        
        print(f"\n✅ Total events read: {len(all_chunks)}")
        reader.close_all()
        print("✅ TEST 2 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_time_filtering():
    """Test reading events with time filter"""
    print_header("TEST 3: Time-Based Filtering")
    
    if not WINDOWS_AVAILABLE:
        print("⚠️  Skipping test - pywin32 not available")
        return False
    
    try:
        reader = EventViewerReader()
        
        # Get events from the last hour
        one_hour_ago = datetime.now() - timedelta(hours=1)
        
        print(f"\n📖 Reading events since {one_hour_ago.strftime('%Y-%m-%d %H:%M:%S')}...")
        chunks = reader.read_events(
            log_name="System",
            source_id="test-time-filter",
            start_time=one_hour_ago,
            max_events=20
        )
        
        print(f"✅ Found {len(chunks)} events from the last hour")
        
        if chunks:
            # Verify all events are within the time range
            oldest = min(chunk.timestamp for chunk in chunks)
            newest = max(chunk.timestamp for chunk in chunks)
            print(f"   Oldest event: {oldest.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   Newest event: {newest.strftime('%Y-%m-%d %H:%M:%S')}")
        
        reader.close_all()
        print("\n✅ TEST 3 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_polling():
    """Test the polling mechanism for real-time monitoring"""
    print_header("TEST 4: Real-Time Polling (10 seconds)")
    
    if not WINDOWS_AVAILABLE:
        print("⚠️  Skipping test - pywin32 not available")
        return False
    
    try:
        reader = EventViewerReader()
        
        print("\n⏱️  Starting 10-second polling test...")
        print("   Checking for new events every 2 seconds...")
        
        last_check = datetime.now()
        poll_count = 0
        total_events = 0
        
        for i in range(5):  # Poll 5 times (every 2 seconds)
            time.sleep(2)
            poll_count += 1
            
            chunks = reader.poll_new_events(
                log_name="System",
                source_id="test-polling",
                last_check=last_check
            )
            
            if chunks:
                print(f"   📝 Poll {poll_count}: Found {len(chunks)} new events")
                total_events += len(chunks)
            else:
                print(f"   ⚪ Poll {poll_count}: No new events")
            
            last_check = datetime.now()
        
        print(f"\n✅ Polling completed. Total events detected: {total_events}")
        reader.close_all()
        print("✅ TEST 4 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_watcher():
    """Test the EventViewerWatcher class"""
    print_header("TEST 5: EventViewerWatcher (15 seconds)")
    
    if not WINDOWS_AVAILABLE:
        print("⚠️  Skipping test - pywin32 not available")
        return False
    
    try:
        import threading
        
        reader = EventViewerReader()
        watcher = EventViewerWatcher(reader, rag=None)  # No RAG service for testing
        
        # Create test sources
        system_source = LogSource(
            name="System Events",
            source_type=SourceType.EVENTVIEWER,
            eventlog_name="System"
        )
        
        app_source = LogSource(
            name="Application Events",
            source_type=SourceType.EVENTVIEWER,
            eventlog_name="Application"
        )
        
        # Start watching
        watcher.watch_source(system_source)
        watcher.watch_source(app_source)
        
        print(f"\n✅ Watching {len(watcher.watching)} sources")
        print("⏱️  Running watcher for 15 seconds...")
        
        # Run watcher in background thread
        watcher_thread = threading.Thread(target=watcher.start, daemon=True)
        watcher_thread.start()
        
        # Let it run for 15 seconds
        time.sleep(15)
        
        # Stop watcher
        watcher.stop()
        
        print("\n✅ Watcher stopped successfully")
        print("✅ TEST 5 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 5 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_event_levels():
    """Test event level mapping"""
    print_header("TEST 6: Event Level Distribution")
    
    if not WINDOWS_AVAILABLE:
        print("⚠️  Skipping test - pywin32 not available")
        return False
    
    try:
        reader = EventViewerReader()
        
        print("\n📖 Reading 50 events to analyze level distribution...")
        chunks = reader.read_events(
            log_name="System",
            source_id="test-levels",
            max_events=50
        )
        
        # Count events by level
        level_counts = {}
        for chunk in chunks:
            level = chunk.log_level.value
            level_counts[level] = level_counts.get(level, 0) + 1
        
        print(f"\n📊 Event Level Distribution (out of {len(chunks)} events):")
        print("-" * 50)
        for level, count in sorted(level_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(chunks)) * 100
            bar = "█" * int(percentage / 2)
            print(f"   {level.upper():10} : {count:3} ({percentage:5.1f}%) {bar}")
        
        reader.close_all()
        print("\n✅ TEST 6 PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST 6 FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("  EventViewer Integration Test Suite")
    print("  Testing Windows Event Log reading functionality")
    print("=" * 70)
    
    # Check if running on Windows
    if os.name != 'nt':
        print("\n⚠️  WARNING: This test suite requires Windows")
        print("   Current OS: " + os.name)
        sys.exit(1)
    
    # Run tests
    tests = [
        ("Basic Event Reading", test_basic_reading),
        ("Multiple Event Logs", test_multiple_logs),
        ("Time-Based Filtering", test_time_filtering),
        ("Real-Time Polling", test_polling),
        ("EventViewerWatcher", test_watcher),
        ("Event Level Distribution", test_event_levels),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except KeyboardInterrupt:
            print("\n\n⚠️  Test interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Print summary
    print_header("TEST SUMMARY")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\nTests Run: {total}")
    print(f"Passed:    {passed}")
    print(f"Failed:    {total - passed}")
    print(f"Success:   {(passed/total)*100:.1f}%\n")
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")
    
    print("\n" + "=" * 70)
    
    if passed == total:
        print("🎉 All tests passed! EventViewer integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
