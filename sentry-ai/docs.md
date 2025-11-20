# Sentry-AI Development Documentation
**Version:** 0.1.0 "Cyrus"  
**Status:** Core AI Backend Complete  
**Last Updated:** November 7, 2025  

---

## 🎯 Project Overview

**Sentry-AI** is a local-first, privacy-by-design desktop application that acts as an AI "co-pilot" for IT diagnostics. It aggregates local log sources (files, folders, Windows EventViewer) into a single AI-powered interface to reduce diagnostic time from hours to seconds.

### Key Philosophy
- **100% Local** - No cloud, no data leaving the machine
- **Privacy-First** - All processing happens on-premises
- **Fast & Practical** - 2-3 second response times with good-enough accuracy
- **Real-time** - Live monitoring of EventViewer and file changes

---

## 🏗️ Architecture Overview
```
┌─────────────────────────────────────────────────────────┐
│                     USER INTERFACE                       │
│                    (To Be Built)                         │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                    RAG SERVICE                           │
│              (Query Orchestrator)                        │
└─┬───────────┬───────────┬──────────┬────────────────────┘
  │           │           │          │
  │           │           │          │
┌─▼────┐  ┌──▼─────┐  ┌──▼────┐  ┌─▼─────┐
│Embed │  │Vector  │  │ LLM   │  │  DB   │
│ding  │  │ Store  │  │Client │  │SQLite │
└──────┘  └────────┘  └───────┘  └───────┘
```

### Data Flow (RAG Pipeline)
```
User Query
    ↓
1. Embed Query (text → vector)
    ↓
2. Search Vector Store (find similar logs)
    ↓
3. Retrieve Chunks (from SQLite)
    ↓
4. Optional Reranking (BM25)
    ↓
5. LLM Generation (with context)
    ↓
Natural Language Answer + Citations
```

---

## 🛠️ Technology Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Language** | Python | 3.9+ | Core application |
| **Data Models** | Pydantic | 2.5+ | Type-safe data structures |
| **Config** | pydantic-settings | 2.1+ | Configuration management |
| **Database** | SQLite | 3.x | Metadata storage |
| **Vector DB** | FAISS | 1.7+ | Vector similarity search |
| **Embeddings** | sentence-transformers | 2.2+ | Text-to-vector conversion |
| **LLM** | Ollama | Latest | Local LLM serving |
| **Model (Embedding)** | all-MiniLM-L6-v2 | - | 80MB, 384-dim vectors |
| **Model (LLM)** | DeepSeek R1 1.5B | - | Fast, 2GB RAM, good quality |

### Why These Choices?

**Pydantic**: Type safety prevents bugs, provides auto-validation, and serves as documentation.

**SQLite**: Zero-config, embedded, perfect for desktop apps. Stores metadata (sources, chunks, chat history).

**FAISS**: Facebook's vector search library. Blazing fast (<1ms searches), runs in-process, no server needed.

**sentence-transformers**: HuggingFace's embedding library. Pre-trained models, easy to use.

**all-MiniLM-L6-v2**: Small (80MB), fast (2000 texts/sec), good semantic understanding. Perfect balance for desktop use.

**Ollama**: Local LLM server. Easy to install, manages models, provides API.

**DeepSeek R1 1.5B**: Smaller than Llama 3 8B, faster responses (2s vs 8s), uses only 2GB RAM. Good enough for log analysis.

---

## 📁 Project Structure
```
sentry-ai/
├── sentry/
│   ├── __init__.py                 # Package init (version info)
│   ├── cli.py                      # Main CLI interface (TODO)
│   │
│   ├── core/                       # Core data structures & config
│   │   ├── __init__.py
│   │   ├── models.py               # ✅ Pydantic models
│   │   ├── config.py               # ✅ Settings management
│   │   └── database.py             # ✅ SQLite operations
│   │
│   ├── services/                   # AI & business logic services
│   │   ├── __init__.py
│   │   ├── embedding.py            # ✅ Text → Vector conversion
│   │   ├── vectorstore.py          # ✅ FAISS vector search
│   │   ├── llm.py                  # ✅ Ollama LLM client
│   │   ├── rag.py                  # ✅ RAG orchestration
│   │   ├── indexer.py              # TODO: Parse & chunk logs
│   │   ├── log_watcher.py          # TODO: File monitoring
│   │   └── eventviewer.py          # TODO: Windows EventViewer
│   │
│   └── ui/                         # User interface (TODO)
│       ├── __init__.py
│       ├── chat.py                 # TODO: Chat interface
│       └── settings.py             # TODO: Settings UI
│
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
├── README.md                       # Project readme
├── docs.md                         # This file
├── .gitignore                      # Git ignore rules
│
└── tests/                          # Test files (currently in root)
    ├── test_core.py                # ✅ Models & config tests
    ├── test_database.py            # ✅ Database tests
    ├── test_embedding.py           # ✅ Embedding tests
    ├── test_vectorstore.py         # ✅ Vector store tests
    ├── test_llm.py                 # ✅ LLM tests
    └── test_rag.py                 # ✅ RAG pipeline tests
```

---

## 🧩 Core Components (Completed)

### 1. Data Models (`sentry/core/models.py`)

**Purpose**: Define the "shape" of all data in the system.

**Key Models**:
```python
LogSource      # Represents a source (file, folder, EventViewer)
LogChunk       # A piece of log data (512 chars, embedded)
ChatMessage    # User/assistant messages
QueryResult    # RAG query response (answer + sources)
IndexingStatus # Progress tracking for indexing
```

**Why Pydantic?**
- Runtime validation (prevents bad data from entering system)
- Auto-serialization (`.model_dump()`, `.model_dump_json()`)
- IDE autocomplete (knows what fields exist)
- Self-documenting (models ARE the schema)

**Example**:
```python
chunk = LogChunk(
    source_id="abc-123",
    content="ERROR: Disk failure",
    timestamp=datetime.now(),
    log_level=LogLevel.ERROR
)
# If any field is wrong type, Pydantic raises ValidationError
```

---

### 2. Configuration (`sentry/core/config.py`)

**Purpose**: Single source of truth for all settings.

**Key Settings**:
- Storage paths (`data_dir`, `db_path`, `vector_index_path`)
- Embedding config (`embedding_model`, `chunk_size`, `chunk_overlap`)
- LLM config (`llm_model`, `temperature`, `context_window`)
- Performance tuning (`batch_size`, `max_workers`)

**Environment Variable Overrides**:
```bash
export SENTRY_LLM_MODEL="llama3:8b"
export SENTRY_CHUNK_SIZE=1024
```

**Usage**:
```python
from sentry.core import settings

print(settings.llm_model)  # "deepseek-r1:1.5b"
print(settings.data_dir)   # Path("C:/Users/.../.sentry-ai")
```

**Key Pattern**: Centralized config prevents "magic numbers" scattered throughout code.

---

### 3. Database (`sentry/core/database.py`)

**Purpose**: Persist metadata to SQLite.

**Schema**:
```sql
-- Log sources (files, folders, EventViewer configs)
CREATE TABLE sources (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source_type TEXT,           -- 'file', 'folder', 'eventviewer'
    path TEXT,                  -- For files/folders
    eventlog_name TEXT,         -- For EventViewer
    is_active INTEGER,
    created_at TEXT,
    last_indexed TEXT,
    total_chunks INTEGER
);

-- Log chunks (the indexed data)
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    source_id TEXT,             -- FK to sources
    content TEXT,               -- The actual log text
    timestamp TEXT,
    log_level TEXT,             -- 'error', 'warning', etc.
    metadata TEXT,              -- JSON blob for extra data
    embedding_id INTEGER,       -- Link to FAISS vector
    created_at TEXT,
    FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE
);

-- Chat history
CREATE TABLE chat_history (
    id INTEGER PRIMARY KEY,
    role TEXT,                  -- 'user' or 'assistant'
    content TEXT,
    timestamp TEXT,
    sources TEXT                -- JSON array of chunk IDs used
);
```

**Key Methods**:
- `add_source()` / `list_sources()` / `delete_source()`
- `add_chunk()` / `add_chunks_batch()` / `get_chunks_by_ids()`
- `add_chat_message()` / `get_chat_history()`

**Critical Pattern - In-Memory DB Handling**:
```python
# For testing, we use :memory: databases
# Problem: Each connection creates a NEW empty database
# Solution: Keep a persistent connection for :memory:

self._memory_conn = None
if self.db_path == ":memory:":
    self._memory_conn = sqlite3.connect(":memory:")
    # This connection stays alive for the object's lifetime
```

**Why SQLite?**
- Zero config (no server to run)
- Fast (millions of reads/sec)
- Reliable (ACID transactions)
- Portable (single file)

---

### 4. Embedding Service (`sentry/services/embedding.py`)

**Purpose**: Convert text to numerical vectors that capture semantic meaning.

**Model**: `all-MiniLM-L6-v2`
- **Size**: 80MB
- **Output**: 384-dimensional vectors
- **Speed**: ~2000 texts/second on CPU
- **Quality**: Good semantic understanding

**Key Methods**:
```python
embedder = EmbeddingService()

# Single text
vector = embedder.embed_text("ERROR: Disk failure")
# Returns: np.array([0.23, -0.45, 0.89, ...])  (384 numbers)

# Batch (20x faster than loop)
vectors = embedder.embed_batch(["text1", "text2", "text3"])
# Returns: np.array([[...], [...], [...]])  (3x384)

# Similarity
score = embedder.similarity("disk error", "drive failure")
# Returns: 0.87 (high similarity)
```

**How It Works**:
1. Loads pre-trained model from HuggingFace (downloads once, caches locally)
2. Uses sentence-transformers library (PyTorch under the hood)
3. Normalizes vectors (makes them unit length for cosine similarity)
4. Runs on CPU (no GPU needed)

**Why This Model?**
- **Small**: Fits in memory easily
- **Fast**: No GPU needed, runs on any PC
- **Local**: Downloads once, runs offline forever
- **Good enough**: Understands "disk error" ≈ "drive failure"

**Vector Properties**:
```python
# Semantically similar texts have similar vectors
embed("disk failure")    # [0.23, -0.45, 0.89, ...]
embed("drive error")     # [0.25, -0.43, 0.91, ...]  ← Close!
embed("coffee break")    # [-0.78, 0.12, -0.34, ...] ← Far!

# Similarity = dot product (since vectors are normalized)
similarity = np.dot(vec1, vec2)  # Returns 0.0-1.0
```

---

### 5. Vector Store (`sentry/services/vectorstore.py`)

**Purpose**: Store vectors and find similar ones fast.

**Technology**: FAISS (Facebook AI Similarity Search)
- In-memory index
- Sub-millisecond search (<1ms for 10k vectors)
- No server needed

**Key Methods**:
```python
store = VectorStore()

# Add vectors
vectors = embedder.embed_batch(["log1", "log2", "log3"])
chunk_ids = ["chunk-1", "chunk-2", "chunk-3"]
faiss_ids = store.add(vectors, chunk_ids)

# Search
query_vector = embedder.embed_text("disk error")
chunk_ids, scores = store.search(query_vector, k=10)
# Returns: (["chunk-1", "chunk-5", ...], [0.89, 0.76, ...])

# Save/Load
store.save()  # Persists to disk
store.load()  # Loads back
```

**Index Type**: `IndexFlatIP` (Inner Product)
- "Flat" = exact search (no approximation)
- "IP" = Inner Product (equivalent to cosine similarity for normalized vectors)
- Fast enough for desktop use (<1M vectors)

**Mapping System**:
```python
# FAISS uses integer IDs (0, 1, 2, ...)
# We use UUID chunk IDs ("abc-123-def-456")
# Solution: Maintain mappings

self.id_mapping = {0: "chunk-abc", 1: "chunk-def", ...}
self.reverse_mapping = {"chunk-abc": 0, "chunk-def": 1, ...}
```

**Persistence**:
- FAISS index → `.faiss` file (binary format)
- Mappings → `.faiss.meta` file (pickled dict)

**Scalability**:
```
10K vectors   = ~15MB RAM,  <1ms search
100K vectors  = ~150MB RAM, ~5ms search
1M vectors    = ~1.5GB RAM, ~50ms search
```

---

### 6. LLM Client (`sentry/services/llm.py`)

**Purpose**: Generate natural language responses using local LLM.

**Technology**: Ollama
- Local LLM server (runs on localhost:11434)
- Manages models (download, load, serve)
- Provides OpenAI-compatible API

**Current Model**: `deepseek-r1:1.5b`
- **Size**: ~2GB RAM
- **Speed**: ~2 second responses
- **Quality**: Good for factual tasks (log analysis)

**Key Methods**:
```python
llm = LLMClient()

# Simple generation
answer = llm.generate("What is DNS?")

# RAG with context (the main use case)
chunks = [chunk1, chunk2, chunk3]  # Retrieved from vector search
answer = llm.generate_with_context(
    query="What errors occurred?",
    context_chunks=chunks
)
# Returns: "Based on the logs, I found 3 disk errors..."

# Multi-turn chat
messages = [
    {'role': 'user', 'content': 'Show errors'},
    {'role': 'assistant', 'content': 'Found 3 errors...'},
    {'role': 'user', 'content': 'What caused them?'}
]
answer = llm.chat(messages)

# Streaming (for real-time UI)
for chunk in llm.stream_generate(prompt):
    print(chunk, end='')
```

**Prompt Engineering**:
```python
# System prompt (sets personality/instructions)
system_prompt = """You are Sentry-AI, an expert sysadmin.
Be concise and technical. Cite log entries."""

# User prompt (formatted with context)
prompt = f"""
QUESTION: {user_query}

LOG ENTRIES:
[1] 2025-11-03 09:01:03 | ERROR
{log_content_1}

[2] 2025-11-03 09:01:05 | ERROR
{log_content_2}

Answer based ONLY on these logs.
"""
```

**Why Low Temperature (0.1)?**
```python
temperature = 0.1  # Low = deterministic, factual
```
- High temperature (0.9) = creative, varied responses
- Low temperature (0.1) = consistent, factual responses
- For log analysis, we want facts, not creativity

**Model Comparison**:
```
Llama 3 8B:       6GB RAM, excellent quality, 8s response  ❌ Too slow
DeepSeek R1 1.5B: 2GB RAM, good quality, 2s response      ✅ Perfect
TinyLlama 1.1B:   1GB RAM, basic quality, 1s response     ⚠️ Quality drop
```

---

### 7. RAG Service (`sentry/services/rag.py`)

**Purpose**: Orchestrate all components to answer queries.

**This is the brain of Sentry-AI.**

**Query Flow**:
```python
def query(query_text: str) -> QueryResult:
    # 1. Embed the query
    query_vector = self.embedder.embed_text(query_text)
    
    # 2. Search vector store
    chunk_ids, scores = self.vector_store.search(query_vector, k=10)
    
    # 3. Retrieve full chunks from database
    chunks = db.get_chunks_by_ids(chunk_ids)
    
    # 4. Optional: Rerank with BM25 (keyword matching)
    if settings.use_reranking:
        chunks, scores = self._rerank_bm25(query_text, chunks, scores)
    
    # 5. Generate answer with LLM
    answer = self.llm.generate_with_context(query_text, chunks)
    
    # 6. Return result
    return QueryResult(
        answer=answer,
        sources=chunks,
        confidence=avg(scores),
        query_time=elapsed_time
    )
```

**Indexing Flow**:
```python
def index_chunks_batch(chunks: List[LogChunk]) -> int:
    # 1. Save to database
    db.add_chunks_batch(chunks)
    
    # 2. Embed all contents
    vectors = self.embedder.embed_batch([c.content for c in chunks])
    
    # 3. Add to vector store
    self.vector_store.add(vectors, [c.id for c in chunks])
    
    return len(chunks)
```

**BM25 Reranking** (Optional):
```python
# Why rerank?
# - Vector search is semantic ("disk error" matches "drive failure")
# - BM25 is lexical (exact keyword matches like "EventID 7")
# - Combining both gives best results

# Hybrid scoring:
final_score = 0.7 * vector_score + 0.3 * bm25_score
```

**Key Methods**:
- `query()` - Main RAG query (end-to-end)
- `index_chunk()` / `index_chunks_batch()` - Add logs to system
- `search_similar()` - Find similar logs (no LLM, just search)
- `get_stats()` - System statistics
- `save_index()` - Persist vector store

---

## 📊 Data Storage

### File Locations (Default)
```
C:\Users\{USER}\.sentry-ai\
├── sentry.db              # SQLite database (metadata)
├── vectors.faiss          # FAISS vector index
├── vectors.faiss.meta     # FAISS metadata (mappings)
└── cache\
    └── models\            # Cached embedding model
        └── sentence-transformers_all-MiniLM-L6-v2\
```

### Data Sizes (Approximate)
```
Database (10K chunks):     ~5MB
FAISS Index (10K vectors): ~15MB
Embedding Model:           ~80MB (cached)
LLM Model:                 ~2GB (managed by Ollama)

Total Disk Usage:          ~100MB + 2GB for LLM
Total RAM Usage:           ~2.5GB (during queries)
```

---

## 🧪 Testing

All core components have test files:
```bash
# Test models and config
python test_core.py

# Test database operations
python test_database.py

# Test embeddings
python test_embedding.py

# Test vector store
python test_vectorstore.py

# Test LLM
python test_llm.py

# Test complete RAG pipeline
python test_rag.py
```

**Test Coverage**:
- ✅ Models: Validation, serialization
- ✅ Database: CRUD operations, foreign keys, in-memory handling
- ✅ Embeddings: Single/batch, similarity, performance
- ✅ Vector Store: Add, search, save/load, persistence
- ✅ LLM: Generation, context, streaming, multi-turn
- ✅ RAG: End-to-end query flow, indexing, reranking

---

## 🚀 Setup Instructions

### Prerequisites

1. **Python 3.9+**
```bash
   python --version  # Should be 3.9 or higher
```

2. **Ollama** (for LLM)
   - Download: https://ollama.com/download
   - Install and it auto-starts as a service
   - Pull model:
```bash
     ollama pull deepseek-r1:1.5b
```

### Installation
```bash
# 1. Clone repository
git clone <repo-url>
cd sentry-ai

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install package in development mode
pip install -e .
```

### Verify Installation
```bash
# Check Ollama is running
ollama list

# Run tests
python test_core.py
python test_embedding.py  # Downloads model on first run (~80MB)
python test_rag.py
```

---

## 🎯 What's Working (Completed)

### ✅ Core Data Layer
- Pydantic models for type-safe data
- Configuration management with env var overrides
- SQLite database with proper schema
- Foreign key constraints and cascade deletes

### ✅ AI Components
- **Embeddings**: Text → 384-dim vectors, batch processing
- **Vector Store**: FAISS index with <1ms search, persistence
- **LLM Client**: Ollama integration, streaming, multi-turn chat
- **RAG Pipeline**: Complete query flow from text to answer

### ✅ Features
- Semantic search (understands meaning, not just keywords)
- Citation of sources (answer includes which logs were used)
- Confidence scoring (based on similarity scores)
- Optional BM25 reranking (hybrid search)
- Batch indexing (fast bulk operations)
- Persistent storage (survives restarts)

### ✅ Testing
- All core components have working tests
- In-memory database handling for tests
- Performance benchmarks included

---

## 🚧 What's Not Done (TODO)

### High Priority

1. **Log Indexer** (`services/indexer.py`)
   - Parse log files
   - Chunk logs intelligently (by line, timestamp, or size)
   - Detect log levels (ERROR, WARNING, INFO)
   - Extract metadata (timestamps, file paths, line numbers)
   - Handle different log formats (syslog, JSON, custom)

2. **File Watcher** (`services/log_watcher.py`)
   - Monitor files/folders for changes
   - Detect new log entries
   - Incremental indexing (don't re-index entire file)
   - Debouncing (wait for file to stop changing)
   - Library: `watchdog`

3. **EventViewer Integration** (`services/eventviewer.py`)
   - Read Windows Event Logs (System, Application, Security)
   - Poll for new events
   - Convert to LogChunk format
   - Filter by level, event ID, time range
   - Library: `pywin32` (Windows only)

4. **CLI Interface** (`cli.py`)
   - Rich TUI using `textual` library
   - Chat interface for queries
   - Settings management (add/remove sources)
   - Status display (indexing progress, stats)
   - Real-time updates

5. **Source Management**
   - Add/remove log sources via UI
   - Trigger re-indexing
   - View source statistics
   - Enable/disable sources

### Medium Priority

6. **Advanced Features**
   - Time-range filtering ("errors in last hour")
   - Log level filtering ("only show CRITICAL")
   - Multi-source queries (search across all sources)
   - Query history (save past queries)
   - Export results (to text, JSON)

7. **Performance Optimizations**
   - Incremental indexing (only new logs)
   - Background indexing (don't block UI)
   - Chunk caching (avoid re-embedding)
   - Connection pooling

8. **Error Handling**
   - Graceful failures (if Ollama down, show message)
   - Retry logic (for transient failures)
   - Logging (structured logs for debugging)
   - User-friendly error messages

### Low Priority (Nice to Have)

9. **Advanced Search**
   - Regular expressions
   - Faceted search (filter by source, level, time)
   - Boolean operators (AND, OR, NOT)
   - Saved searches

10. **Analytics**
    - Error trends over time
    - Most common errors
    - Source statistics (which source has most errors)
    - Visualization (charts, graphs)

11. **Export/Backup**
    - Backup database
    - Export index
    - Import/export sources
    - Cloud sync (optional)

---

## 📝 Implementation Guide for New Developers

### Starting Point: Indexer Service

This is the most critical missing piece. Here's the roadmap:

#### 1. Log Indexer (`services/indexer.py`)

**Goal**: Parse log files and convert to LogChunks.

**Key Functions**:
```python
class LogIndexer:
    def parse_file(self, file_path: Path) -> List[LogChunk]:
        """
        Parse a log file into chunks
        
        Strategy:
        1. Read file line by line (memory efficient)
        2. Detect log format (regex patterns for common formats)
        3. Extract: timestamp, level, message
        4. Chunk by size (settings.chunk_size characters)
        5. Create LogChunk objects
        """
        pass
    
    def parse_folder(self, folder_path: Path) -> List[LogChunk]:
        """
        Recursively parse all .log files in folder
        """
        pass
    
    def detect_log_level(self, line: str) -> LogLevel:
        """
        Regex patterns:
        - ERROR|ERRO|ERR
        - WARN|WARNING
        - INFO|INFORMATION
        - CRITICAL|CRIT|FATAL
        - DEBUG|DBG
        """
        pass
    
    def extract_timestamp(self, line: str) -> Optional[datetime]:
        """
        Common patterns:
        - 2025-11-03 14:30:45
        - [2025/11/03 14:30:45]
        - Nov 3 14:30:45
        
        Use dateutil.parser for flexibility
        """
        pass
    
    def chunk_logs(self, content: str) -> List[str]:
        """
        Split content into chunks
        
        Options:
        1. By size (N characters with overlap)
        2. By lines (N lines per chunk)
        3. By log entry (keep related lines together)
        
        Recommended: Hybrid (by entry, max N chars)
        """
        pass
```

**Example Implementation**:
```python
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from dateutil import parser as date_parser

from ..core.models import LogChunk, LogLevel
from ..core.config import settings

class LogIndexer:
    # Regex patterns for log levels
    LEVEL_PATTERNS = {
        LogLevel.CRITICAL: re.compile(r'\b(CRITICAL|CRIT|FATAL|SEVERE)\b', re.I),
        LogLevel.ERROR: re.compile(r'\b(ERROR|ERR|FAIL|FAILED)\b', re.I),
        LogLevel.WARNING: re.compile(r'\b(WARN|WARNING)\b', re.I),
        LogLevel.INFO: re.compile(r'\b(INFO|INFORMATION)\b', re.I),
        LogLevel.DEBUG: re.compile(r'\b(DEBUG|DBG|TRACE)\b', re.I),
    }
    
    def parse_file(self, file_path: Path, source_id: str) -> List[LogChunk]:
        """Parse a single log file"""
        chunks = []
        
        # Read file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Split into chunks
        chunk_texts = self._chunk_content(content)
        
        # Create LogChunk objects
        for i, chunk_text in enumerate(chunk_texts):
            chunk = LogChunk(
                source_id=source_id,
                content=chunk_text,
                timestamp=self._extract_timestamp(chunk_text) or datetime.now(),
                log_level=self._detect_log_level(chunk_text),
                metadata={
                    'file': str(file_path),
                    'chunk_index': i
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_content(self, content: str) -> List[str]:
        """Split content into overlapping chunks"""
        chunks = []
        size = settings.chunk_size
        overlap = settings.chunk_overlap
        
        for i in range(0, len(content), size - overlap):
            chunk = content[i:i + size]
            if len(chunk) >= settings.min_chunk_size:
                chunks.append(chunk)
        
        return chunks
    
    def _detect_log_level(self, text: str) -> LogLevel:
        """Detect log level from text"""
        for level, pattern in self.LEVEL_PATTERNS.items():
            if pattern.search(text):
                return level
        return LogLevel.UNKNOWN
    
    def _extract_timestamp(self, text: str) -> Optional[datetime]:
        """Try to extract timestamp from first line"""
        first_line = text.split('\n')[0]
        
        try:
            # Try dateutil parser (flexible)
            return date_parser.parse(first_line, fuzzy=True)
        except:
            return None
```

**Usage**:
```python
indexer = LogIndexer()
chunks = indexer.parse_file(Path("C:/logs/system.log"), source_id)
rag.index_chunks_batch(chunks)
```

**Testing**:
```python
def test_indexer():
    indexer = LogIndexer()
    
    # Create test log file
    test_log = """
2025-11-03 09:01:03 ERROR: Disk failure
2025-11-03 09:01:05 WARNING: High memory
2025-11-03 09:01:08 INFO: Service started
"""
    Path("test.log").write_text(test_log)
    
    # Parse
    chunks = indexer.parse_file(Path("test.log"), "test-source")
    ```python
    # Parse
    chunks = indexer.parse_file(Path("test.log"), "test-source")
    
    # Verify
    assert len(chunks) > 0
    assert chunks[0].log_level == LogLevel.ERROR
    assert "Disk failure" in chunks[0].content
    
    print(f"✅ Parsed {len(chunks)} chunks")
```

---

#### 2. File Watcher (`services/log_watcher.py`)

**Goal**: Monitor files/folders for changes and auto-index new content.

**Key Functions**:

```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
from pathlib import Path
from typing import Callable

class LogFileHandler(FileSystemEventHandler):
    """Handles file system events"""
    
    def __init__(self, callback: Callable[[Path], None]):
        self.callback = callback
        self.last_modified = {}  # Debouncing
    
    def on_modified(self, event):
        """Called when a file is modified"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Only process log files
        if not settings.is_supported_file(file_path):
            return
        
        # Debounce (ignore rapid changes)
        now = time.time()
        last_time = self.last_modified.get(file_path, 0)
        
        if now - last_time < settings.watch_debounce:
            return
        
        self.last_modified[file_path] = now
        
        # Trigger callback
        self.callback(file_path)
    
    def on_created(self, event):
        """Called when a file is created"""
        if not event.is_directory:
            self.on_modified(event)


class LogWatcher:
    """Watches log files/folders for changes"""
    
    def __init__(self, indexer: LogIndexer, rag: RAGService):
        self.indexer = indexer
        self.rag = rag
        self.observer = Observer()
        self.watching = {}  # {source_id: watch}
    
    def watch_source(self, source: LogSource):
        """Start watching a source"""
        if source.source_type == SourceType.EVENTVIEWER:
            # EventViewer handled separately
            return
        
        path = Path(source.path)
        
        if not path.exists():
            logger.error(f"Path doesn't exist: {path}")
            return
        
        # Create handler
        handler = LogFileHandler(
            callback=lambda file_path: self._on_file_changed(file_path, source.id)
        )
        
        # Start watching
        if path.is_file():
            watch_path = path.parent
        else:
            watch_path = path
        
        watch = self.observer.schedule(
            handler,
            str(watch_path),
            recursive=True
        )
        
        self.watching[source.id] = watch
        
        logger.info(f"👁️  Watching: {path}")
    
    def _on_file_changed(self, file_path: Path, source_id: str):
        """Handle file change"""
        logger.info(f"📝 File changed: {file_path}")
        
        try:
            # Parse new content
            # TODO: Implement incremental parsing (only new lines)
            chunks = self.indexer.parse_file(file_path, source_id)
            
            # Index chunks
            self.rag.index_chunks_batch(chunks)
            
            logger.info(f"✅ Indexed {len(chunks)} new chunks")
            
        except Exception as e:
            logger.error(f"Failed to index {file_path}: {e}")
    
    def start(self):
        """Start the observer"""
        if not self.observer.is_alive():
            self.observer.start()
            logger.info("👁️  File watcher started")
    
    def stop(self):
        """Stop the observer"""
        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
            logger.info("👁️  File watcher stopped")
    
    def unwatch_source(self, source_id: str):
        """Stop watching a source"""
        if source_id in self.watching:
            watch = self.watching[source_id]
            self.observer.unschedule(watch)
            del self.watching[source_id]
```

**Usage**:

```python
watcher = LogWatcher(indexer, rag)

# Add sources to watch
for source in db.list_sources(active_only=True):
    watcher.watch_source(source)

# Start watching
watcher.start()

# Keep running
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    watcher.stop()
```

**Incremental Parsing** (Advanced):

```python
class LogIndexer:
    def parse_file_incremental(
        self, 
        file_path: Path, 
        source_id: str,
        last_position: int = 0
    ) -> tuple[List[LogChunk], int]:
        """
        Parse only new content since last_position
        
        Returns:
            (chunks, new_position)
        """
        chunks = []
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Seek to last position
            f.seek(last_position)
            
            # Read new content
            new_content = f.read()
            
            # Get new position
            new_position = f.tell()
        
        # Parse new content
        if new_content.strip():
            chunk_texts = self._chunk_content(new_content)
            
            for chunk_text in chunk_texts:
                chunk = LogChunk(
                    source_id=source_id,
                    content=chunk_text,
                    timestamp=self._extract_timestamp(chunk_text) or datetime.now(),
                    log_level=self._detect_log_level(chunk_text),
                    metadata={
                        'file': str(file_path),
                        'position': last_position
                    }
                )
                chunks.append(chunk)
        
        return chunks, new_position
```

**Store Last Position**:

```sql
-- Add to sources table
ALTER TABLE sources ADD COLUMN last_file_position INTEGER DEFAULT 0;
```

```python
# In database.py
def update_source_position(self, source_id: str, position: int):
    with self.get_connection() as conn:
        conn.execute(
            "UPDATE sources SET last_file_position = ? WHERE id = ?",
            (position, source_id)
        )
        conn.commit()
```

---

#### 3. EventViewer Integration (`services/eventviewer.py`)

**Goal**: Read Windows Event Logs in real-time.

**Windows Only** - Use `pywin32`:

```python
try:
    import win32evtlog
    import win32evtlogutil
    import win32con
    WINDOWS_AVAILABLE = True
except ImportError:
    WINDOWS_AVAILABLE = False
    logger.warning("pywin32 not available - EventViewer support disabled")

from datetime import datetime, timedelta
from typing import List, Optional
import time

from ..core.models import LogChunk, LogLevel, LogSource
from ..core.config import settings


class EventViewerReader:
    """Read Windows Event Logs"""
    
    # Map Windows event types to our LogLevel
    EVENT_TYPE_MAP = {
        win32evtlog.EVENTLOG_ERROR_TYPE: LogLevel.ERROR,
        win32evtlog.EVENTLOG_WARNING_TYPE: LogLevel.WARNING,
        win32evtlog.EVENTLOG_INFORMATION_TYPE: LogLevel.INFO,
        win32evtlog.EVENTLOG_AUDIT_FAILURE: LogLevel.ERROR,
        win32evtlog.EVENTLOG_AUDIT_SUCCESS: LogLevel.INFO,
    }
    
    def __init__(self):
        if not WINDOWS_AVAILABLE:
            raise ImportError("pywin32 required for EventViewer support")
        
        self.handles = {}  # {log_name: handle}
    
    def open_log(self, log_name: str):
        """
        Open an event log
        
        Common logs:
        - "System"
        - "Application"
        - "Security"
        """
        if log_name in self.handles:
            return
        
        try:
            handle = win32evtlog.OpenEventLog(None, log_name)
            self.handles[log_name] = handle
            logger.info(f"📖 Opened EventLog: {log_name}")
        except Exception as e:
            logger.error(f"Failed to open {log_name}: {e}")
            raise
    
    def close_log(self, log_name: str):
        """Close an event log"""
        if log_name in self.handles:
            win32evtlog.CloseEventLog(self.handles[log_name])
            del self.handles[log_name]
    
    def read_events(
        self,
        log_name: str,
        source_id: str,
        start_time: Optional[datetime] = None,
        max_events: int = None
    ) -> List[LogChunk]:
        """
        Read events from a log
        
        Args:
            log_name: Name of log ("System", "Application", etc.)
            source_id: Source ID for chunks
            start_time: Only events after this time (None = all)
            max_events: Max events to read (None = unlimited)
        
        Returns:
            List of LogChunks
        """
        if log_name not in self.handles:
            self.open_log(log_name)
        
        handle = self.handles[log_name]
        chunks = []
        
        flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
        
        max_events = max_events or settings.eventviewer_max_events
        
        try:
            events = win32evtlog.ReadEventLog(handle, flags, 0)
            
            for event in events:
                if len(chunks) >= max_events:
                    break
                
                # Get event time
                event_time = datetime.fromtimestamp(int(event.TimeGenerated))
                
                # Filter by start_time
                if start_time and event_time < start_time:
                    continue
                
                # Convert to LogChunk
                chunk = self._event_to_chunk(event, log_name, source_id)
                chunks.append(chunk)
        
        except Exception as e:
            logger.error(f"Failed to read events from {log_name}: {e}")
        
        return chunks
    
    def _event_to_chunk(
        self,
        event,
        log_name: str,
        source_id: str
    ) -> LogChunk:
        """Convert Windows event to LogChunk"""
        
        # Get event type (error, warning, info)
        event_type = event.EventType
        log_level = self.EVENT_TYPE_MAP.get(event_type, LogLevel.UNKNOWN)
        
        # Get event time
        timestamp = datetime.fromtimestamp(int(event.TimeGenerated))
        
        # Build content
        source_name = event.SourceName
        event_id = event.EventID & 0xFFFF  # Mask to get actual ID
        category = event.EventCategory
        
        # Get message (may need formatting)
        try:
            message = win32evtlogutil.SafeFormatMessage(event, log_name)
        except:
            message = f"Event ID {event_id}"
        
        content = f"[{source_name}] Event ID {event_id}: {message}"
        
        # Create chunk
        chunk = LogChunk(
            source_id=source_id,
            content=content,
            timestamp=timestamp,
            log_level=log_level,
            metadata={
                'log_name': log_name,
                'event_id': event_id,
                'source_name': source_name,
                'category': category,
                'computer': event.ComputerName
            }
        )
        
        return chunk
    
    def poll_new_events(
        self,
        log_name: str,
        source_id: str,
        last_check: datetime
    ) -> List[LogChunk]:
        """
        Poll for events since last_check
        
        Use this in a loop for real-time monitoring
        """
        return self.read_events(
            log_name=log_name,
            source_id=source_id,
            start_time=last_check,
            max_events=settings.eventviewer_max_events
        )
    
    def close_all(self):
        """Close all open logs"""
        for log_name in list(self.handles.keys()):
            self.close_log(log_name)


class EventViewerWatcher:
    """Continuously monitor EventViewer logs"""
    
    def __init__(self, reader: EventViewerReader, rag: RAGService):
        self.reader = reader
        self.rag = rag
        self.watching = {}  # {source_id: (log_name, last_check)}
        self.running = False
    
    def watch_source(self, source: LogSource):
        """Start watching an EventViewer source"""
        if source.source_type != SourceType.EVENTVIEWER:
            return
        
        log_name = source.eventlog_name
        
        if not log_name:
            logger.error(f"EventViewer source {source.id} has no log_name")
            return
        
        # Open the log
        self.reader.open_log(log_name)
        
        # Track it
        self.watching[source.id] = (log_name, datetime.now())
        
        logger.info(f"👁️  Watching EventLog: {log_name}")
    
    def start(self):
        """Start polling loop"""
        self.running = True
        
        logger.info("👁️  EventViewer watcher started")
        
        while self.running:
            for source_id, (log_name, last_check) in self.watching.items():
                try:
                    # Poll for new events
                    chunks = self.reader.poll_new_events(
                        log_name=log_name,
                        source_id=source_id,
                        last_check=last_check
                    )
                    
                    if chunks:
                        # Index new events
                        self.rag.index_chunks_batch(chunks)
                        logger.info(f"📝 Indexed {len(chunks)} events from {log_name}")
                    
                    # Update last check time
                    self.watching[source_id] = (log_name, datetime.now())
                
                except Exception as e:
                    logger.error(f"Error polling {log_name}: {e}")
            
            # Sleep before next poll
            time.sleep(settings.eventviewer_poll_interval)
    
    def stop(self):
        """Stop polling"""
        self.running = False
        self.reader.close_all()
        logger.info("👁️  EventViewer watcher stopped")
```

**Usage**:

```python
# Create EventViewer source
ev_source = LogSource(
    name="Windows System Events",
    source_type=SourceType.EVENTVIEWER,
    eventlog_name="System"
)
db.add_source(ev_source)

# Start watching
reader = EventViewerReader()
watcher = EventViewerWatcher(reader, rag)
watcher.watch_source(ev_source)

# Run in background thread
import threading
thread = threading.Thread(target=watcher.start, daemon=True)
thread.start()
```

**Testing** (Windows only):

```python
def test_eventviewer():
    reader = EventViewerReader()
    
    # Read last 10 System events
    source = LogSource(
        name="Test",
        source_type=SourceType.EVENTVIEWER,
        eventlog_name="System"
    )
    db.add_source(source)
    
    chunks = reader.read_events("System", source.id, max_events=10)
    
    print(f"✅ Read {len(chunks)} events")
    for chunk in chunks[:3]:
        print(f"  [{chunk.log_level.value}] {chunk.content[:60]}...")
```

---

#### 4. CLI Interface (`cli.py`)

**Goal**: Rich terminal UI using Textual.

**Basic Structure**:

```python
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Header, Footer, Input, Static, Button, DataTable
from textual.binding import Binding

from .services.rag import get_rag_service
from .core.database import db
from .core.models import ChatMessage


class ChatView(Static):
    """Display chat messages"""
    
    def __init__(self):
        super().__init__()
        self.messages = []
    
    def add_message(self, role: str, content: str):
        self.messages.append((role, content))
        self.update_display()
    
    def update_display(self):
        lines = []
        for role, content in self.messages:
            if role == "user":
                lines.append(f"[bold cyan]You:[/] {content}")
            else:
                lines.append(f"[bold green]Sentry:[/] {content}")
            lines.append("")
        
        self.update("\n".join(lines))


class SentryApp(App):
    """Sentry-AI TUI Application"""
    
    CSS = """
    #chat-view {
        height: 80%;
        border: solid green;
        padding: 1;
    }
    
    #input-area {
        height: 10%;
        dock: bottom;
    }
    
    #stats {
        height: 10%;
        border: solid blue;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("s", "show_sources", "Sources"),
        Binding("i", "show_stats", "Stats"),
    ]
    
    def __init__(self):
        super().__init__()
        self.rag = get_rag_service()
        self.chat_view = None
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        # Chat area
        self.chat_view = ChatView(id="chat-view")
        yield self.chat_view
        
        # Input area
        with Vertical(id="input-area"):
            yield Input(placeholder="Ask about your logs...", id="query-input")
            with Horizontal():
                yield Button("Send", variant="primary", id="send-btn")
                yield Button("Clear", id="clear-btn")
        
        # Stats area
        yield Static(id="stats")
        
        yield Footer()
    
    def on_mount(self):
        """Called when app starts"""
        self.update_stats()
        
        # Welcome message
        self.chat_view.add_message(
            "assistant",
            "Welcome to Sentry-AI! Ask me about your logs."
        )
    
    def on_button_pressed(self, event):
        """Handle button clicks"""
        if event.button.id == "send-btn":
            self.send_query()
        elif event.button.id == "clear-btn":
            self.chat_view.messages = []
            self.chat_view.update_display()
    
    def on_input_submitted(self, event):
        """Handle Enter key in input"""
        if event.input.id == "query-input":
            self.send_query()
    
    def send_query(self):
        """Send query to RAG"""
        input_widget = self.query_one("#query-input", Input)
        query = input_widget.value.strip()
        
        if not query:
            return
        
        # Clear input
        input_widget.value = ""
        
        # Show user message
        self.chat_view.add_message("user", query)
        
        # Show "thinking" message
        self.chat_view.add_message("assistant", "Searching logs...")
        
        # Query RAG (in background to avoid blocking UI)
        self.run_worker(self._query_rag(query))
    
    async def _query_rag(self, query: str):
        """Query RAG in background"""
        result = self.rag.query(query)
        
        # Remove "thinking" message
        self.chat_view.messages.pop()
        
        # Show answer
        answer = result.answer
        if result.sources:
            answer += f"\n\n[dim]({len(result.sources)} sources, confidence: {result.confidence:.2f})[/]"
        
        self.chat_view.add_message("assistant", answer)
        
        # Save to database
        db.add_chat_message(ChatMessage(role="user", content=query))
        db.add_chat_message(ChatMessage(
            role="assistant",
            content=result.answer,
            sources=[c.id for c in result.sources]
        ))
    
    def update_stats(self):
        """Update stats display"""
        stats = self.rag.get_stats()
        
        stats_text = f"""
[bold]Statistics[/]
Total Chunks: {stats['total_chunks']}
Sources: {stats['database_stats'].get('total_sources', 0)}
Model: {stats['llm_model']}
"""
        
        self.query_one("#stats", Static).update(stats_text)
    
    def action_show_sources(self):
        """Show sources screen"""
        # TODO: Implement sources management screen
        pass
    
    def action_show_stats(self):
        """Show detailed stats"""
        # TODO: Implement stats screen
        pass


def main():
    """Entry point"""
    app = SentryApp()
    app.run()


if __name__ == "__main__":
    main()
```

**Run it**:

```bash
python -m sentry.cli
```

**Advanced Features**:

```python
# Add syntax highlighting for logs
from rich.syntax import Syntax

# Add progress bars for indexing
from textual.widgets import ProgressBar

# Add tabs for different views
from textual.widgets import TabbedContent, TabPane

# Add file browser for adding sources
from textual.widgets import DirectoryTree
```

---

### 5. Putting It All Together

**Main Application Flow**:

```python
# sentry/cli.py (complete version)

import sys
import logging
from pathlib import Path
import threading

from textual.app import App

from .services.rag import get_rag_service
from .services.indexer import LogIndexer
from .services.log_watcher import LogWatcher
from .services.eventviewer import EventViewerReader, EventViewerWatcher, WINDOWS_AVAILABLE
from .core.database import db
from .core.config import settings


# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.data_dir / 'sentry.log'),
        logging.StreamHandler(sys.stdout)
    ]
)


class SentryApplication:
    """Main application controller"""
    
    def __init__(self):
        self.rag = get_rag_service()
        self.indexer = LogIndexer()
        self.log_watcher = None
        self.ev_watcher = None
        
    def initialize(self):
        """Initialize the application"""
        print("🔧 Initializing Sentry-AI...")
        
        # Check Ollama connection
        try:
            info = self.rag.llm.get_model_info()
            print(f"✅ LLM: {info['model']}")
        except Exception as e:
            print(f"⚠️  Warning: LLM not available ({e})")
            print("   Some features will be limited.")
        
        # Load existing sources
        sources = db.list_sources(active_only=True)
        print(f"📂 Loaded {len(sources)} active sources")
        
        # Start file watcher
        if settings.watch_enabled:
            self.log_watcher = LogWatcher(self.indexer, self.rag)
            
            for source in sources:
                if source.source_type != "eventviewer":
                    self.log_watcher.watch_source(source)
            
            self.log_watcher.start()
            print("👁️  File watcher started")
        
        # Start EventViewer watcher (Windows only)
        if WINDOWS_AVAILABLE and settings.eventviewer_enabled:
            reader = EventViewerReader()
            self.ev_watcher = EventViewerWatcher(reader, self.rag)
            
            for source in sources:
                if source.source_type == "eventviewer":
                    self.ev_watcher.watch_source(source)
            
            # Run in background thread
            thread = threading.Thread(target=self.ev_watcher.start, daemon=True)
            thread.start()
            print("👁️  EventViewer watcher started")
        
        print("✅ Sentry-AI ready!\n")
    
    def shutdown(self):
        """Clean shutdown"""
        print("\n🛑 Shutting down...")
        
        if self.log_watcher:
            self.log_watcher.stop()
        
        if self.ev_watcher:
            self.ev_watcher.stop()
        
        # Save vector index
        self.rag.save_index()
        
        print("✅ Goodbye!")


def main():
    """Main entry point"""
    app_controller = SentryApplication()
    
    try:
        # Initialize
        app_controller.initialize()
        
        # Launch TUI
        app = SentryApp(app_controller)
        app.run()
        
    except KeyboardInterrupt:
        pass
    finally:
        app_controller.shutdown()


if __name__ == "__main__":
    main()
```

---

## 🔧 Configuration Reference

All settings in `sentry/core/config.py`:

```python
# Storage
data_dir: Path = Path.home() / ".sentry-ai"
db_path: Path = data_dir / "sentry.db"
vector_index_path: Path = data_dir / "vectors.faiss"

# Embedding
embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
embedding_dimension: int = 384
embedding_batch_size: int = 32

# Chunking
chunk_size: int = 512
chunk_overlap: int = 50
min_chunk_size: int = 50

# Vector Search
top_k_results: int = 10
similarity_threshold: float = 0.7
use_reranking: bool = True

# LLM
ollama_host: str = "http://localhost:11434"
llm_model: str = "deepseek-r1:1.5b"
llm_temperature: float = 0.1
llm_context_window: int = 4096
llm_max_tokens: int = 1024

# Indexing
max_file_size_mb: int = 100
supported_extensions: List[str] = [".log", ".txt", ".out", ".err"]
parallel_indexing: bool = True
max_workers: int = 4

# File Watching
watch_enabled: bool = True
watch_poll_interval: float = 2.0
watch_debounce: float = 1.0

# EventViewer
eventviewer_enabled: bool = True
eventviewer_poll_interval: int = 5
eventviewer_max_events: int = 1000
eventviewer_default_logs: List[str] = ["System", "Application"]
```

---

## 🐛 Common Issues & Solutions

### 1. Ollama Connection Failed

**Symptom**: `Failed to connect to Ollama: ...`

**Solutions**:
```bash
# Check if Ollama is running
ollama list

# Start Ollama manually
ollama serve

# Check port
curl http://localhost:11434/api/tags
```

### 2. Model Not Found

**Symptom**: `Model 'deepseek-r1:1.5b' not found`

**Solution**:
```bash
ollama pull deepseek-r1:1.5b
```

### 3. Foreign Key Constraint

**Symptom**: `FOREIGN KEY constraint failed`

**Cause**: Trying to add chunks without a valid source

**Solution**:
```python
# Always create source first
source = LogSource(name="My Logs", source_type=SourceType.FILE, path="/path")
db.add_source(source)

# Then use source.id for chunks
chunk = LogChunk(source_id=source.id, ...)
```

### 4. Embedding Model Download Slow

**Symptom**: First run takes a while

**Cause**: Downloading 80MB model from HuggingFace

**Solution**: Be patient (one-time download) or pre-download:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
```

### 5. Out of Memory

**Symptom**: Python crashes or system freezes

**Solutions**:
- Use smaller LLM model: `ollama pull tinyllama:1.1b`
- Reduce batch size: `SENTRY_EMBEDDING_BATCH_SIZE=16`
- Reduce context window: `SENTRY_LLM_CONTEXT_WINDOW=2048`
- Close other applications

### 6. Slow Queries

**Symptom**: RAG queries take >5 seconds

**Causes & Solutions**:
- **LLM too large**: Switch to smaller model (deepseek-r1:1.5b or tinyllama)
- **Too many results**: Reduce `top_k_results` to 5
- **Large chunks**: Reduce `chunk_size` to 256
- **Old CPU**: Consider cloud deployment or better hardware

---

## 📊 Performance Benchmarks

### Query Performance (Typical)

```
Component          Time (ms)    Notes
─────────────────────────────────────────────────
Embed query        5-10         Single text embedding
Vector search      <1           FAISS search (10k vectors)
DB retrieval       2-5          SQLite lookup
LLM generation     1500-2500    DeepSeek R1 1.5B
─────────────────────────────────────────────────
Total              1500-2500    Dominated by LLM
```

### Indexing Performance

```
Operation          Speed              Notes
──────────────────────────────────────────────────────────
Parse log file     ~1MB/s             Depends on format
Batch embedding    2000 texts/s       CPU-bound
FAISS indexing     10000 vectors/s    Very fast
DB insertion       5000 chunks/s      Batch operations
──────────────────────────────────────────────────────────
Overall            ~500 chunks/s      End-to-end
```

### Memory Usage

```
Component          RAM Usage
─────────────────────────────────
Python runtime     ~50MB
Embedding model    ~200MB (loaded)
Vector index       ~15MB per 10k vectors
LLM (DeepSeek)     ~2GB
─────────────────────────────────
Total              ~2.3GB minimum
```

---

## 🔐 Security & Privacy

### Data Privacy

- ✅ **100% Local**: No data leaves the machine
- ✅ **No telemetry**: No analytics or tracking
- ✅ **Offline capable**: Works without internet (after initial setup)
- ✅ **Encrypted at rest**: Consider enabling Windows BitLocker

### Sensitive Logs

**Warning**: This application stores logs in plaintext in SQLite.

**Recommendations**:
- Don't index logs containing passwords