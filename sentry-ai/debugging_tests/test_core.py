# test_core.py
"""Quick test to verify core models work"""

from sentry.core import LogSource, LogChunk, SourceType, LogLevel, settings
from datetime import datetime


def test_models():
    print("🧪 Testing Core Models\n")
    
    # Test LogSource
    source = LogSource(
        name="Test System Logs",
        source_type=SourceType.EVENTVIEWER,
        eventlog_name="System"
    )
    print(f"✅ Created LogSource: {source.name} (ID: {source.id})")
    
    # Test LogChunk
    chunk = LogChunk(
        source_id=source.id,
        content="ERROR: Database connection failed",
        timestamp=datetime.now(),
        log_level=LogLevel.ERROR,
        metadata={"server": "db-01"}
    )
    print(f"✅ Created LogChunk: {chunk.content[:50]}... (ID: {chunk.id})")
    
    # Test Settings
    print(f"\n⚙️  Settings:")
    print(f"   App: {settings.app_name} v{settings.version}")
    print(f"   Data Dir: {settings.data_dir}")
    print(f"   Embedding Model: {settings.embedding_model}")
    print(f"   LLM Model: {settings.llm_model}")
    print(f"   Supported Extensions: {settings.supported_extensions}")
    
    print("\n🔥 All core models working!")


if __name__ == "__main__":
    test_models()