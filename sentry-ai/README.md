<p align="center">
  <img src="https://img.shields.io/badge/version-0.2.0-blue?style=for-the-badge" alt="Version"/>
  <img src="https://img.shields.io/badge/codename-Helix-purple?style=for-the-badge" alt="Codename"/>
  <img src="https://img.shields.io/badge/python-3.10+-green?style=for-the-badge&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/license-MIT-orange?style=for-the-badge" alt="License"/>
</p>

<h1 align="center">🛡️ Sentry-AI</h1>

<p align="center">
  <strong>AI-Powered Local Log Analysis & Root Cause Detection</strong>
</p>

<p align="center">
  <em>Analyze log files, detect anomalies, and find root causes — all running locally on your machine. Your data never leaves your system.</em>
</p>

---

## 🎯 What is Sentry-AI?

Sentry-AI is a **privacy-first, local-first** log analysis tool that combines:

- **RAG (Retrieval Augmented Generation)** for intelligent log querying
- **Helix Vector** — a novel DNA-like encoding system for log pattern mining
- **Markov Chain Anomaly Detection** for identifying unusual log sequences
- **Semantic Vector Search** for finding similar log entries
- **LLM Integration** (local Ollama or cloud OpenRouter) for natural language answers

Unlike cloud-based solutions, Sentry-AI runs entirely on your machine. Your logs, your analysis, your privacy.

---

## 🧬 The Science Behind Helix Vector

### DNA Encoding: How It Works

Sentry-AI uses a bioinformatics-inspired approach to log analysis. The **Helix Vector** system treats logs like genetic sequences:

```
Raw Logs → Drain3 Clustering → Cluster IDs (DNA) → Markov Chain → Anomaly Detection
```

#### 1. Pattern Mining with Drain3

[Drain3](https://github.com/logpai/Drain3) clusters log messages into templates by extracting common patterns:

```
Input:  "Connection timeout to 192.168.1.1:3306 after 30s"
        "Connection timeout to 10.0.0.5:5432 after 45s"
        
Output: Template: "Connection timeout to <*>:<*> after <*>s" → Cluster ID: 7
```

This transforms millions of unique log lines into a manageable vocabulary of ~50-200 templates.

#### 2. Markov Chain Transition Probabilities

Once logs are encoded as cluster IDs, we build a transition probability matrix:

```
P(Cluster_j | Cluster_i) = Count(i → j) / Count(i → *)
```

For example, if cluster 5 (successful connection) is always followed by cluster 6 (query execution):

```
P(6|5) = 0.95  → Normal transition (95% of the time)
P(8|5) = 0.02  → Rare transition (2% of the time) → ANOMALY!
```

#### 3. Anomaly Scoring

Each log entry receives an **anomaly score** based on:

```python
anomaly_score = (1 - transition_probability) + severity_weight
```

Where:
- **transition_probability**: How likely this log follows the previous one (from Markov chain)
- **severity_weight**: Keyword-based penalty (ERROR=0.5, CRITICAL=0.8, FATAL=0.9)

Logs are flagged as anomalies when `anomaly_score > threshold` (default: 0.20).

### Mathematical Foundation

| Concept | Formula | Description |
|---------|---------|-------------|
| Transition Probability | `P(j|i) = N(i→j) / Σ N(i→k)` | Conditional probability of cluster j following cluster i |
| Anomaly Score | `A = (1 - P) + S` | Combined transition rarity and severity |
| Similarity Score | `cos(θ) = (A·B) / (‖A‖‖B‖)` | Cosine similarity for vector search |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              Sentry-AI                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────────────┐ │
│  │   Streamlit │    │   Anomaly    │    │         FastAPI             │ │
│  │   Chat UI   │    │  Dashboard   │    │    (Headless Mode)          │ │
│  └──────┬──────┘    └──────┬───────┘    └─────────────┬───────────────┘ │
│         │                  │                          │                  │
│         └──────────────────┼──────────────────────────┘                  │
│                            ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                        RAG Service                                   │ │
│  │   ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌─────────────────┐   │ │
│  │   │ Embedder │  │ VectorStore│  │   LLM    │  │ Intent Classifier│  │ │
│  │   │(MiniLM)  │  │  (FAISS)   │  │(Ollama/  │  │   (Regex-based)  │  │ │
│  │   │          │  │            │  │OpenRouter│  │                  │  │ │
│  │   └──────────┘  └────────────┘  └──────────┘  └─────────────────┘   │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                            │                                             │
│                            ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                     Helix Vector Service                             │ │
│  │   ┌──────────┐  ┌────────────────┐  ┌─────────────────────────────┐ │ │
│  │   │  Drain3  │  │  Markov Chain  │  │  Anomaly Classifier         │ │ │
│  │   │ Clustering│  │  Transitions   │  │  (Pattern + Keyword)        │ │ │
│  │   └──────────┘  └────────────────┘  └─────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                            │                                             │
│                            ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                       Data Layer                                     │ │
│  │   ┌──────────────┐  ┌─────────────┐  ┌────────────────────────────┐ │ │
│  │   │   SQLite     │  │ FAISS Index │  │      Log Indexer           │ │ │
│  │   │  (Metadata)  │  │  (Vectors)  │  │  (.log, .txt, .csv, etc.)  │ │ │
│  │   └──────────────┘  └─────────────┘  └────────────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
sentry-ai/
├── sentry/                     # Main application package
│   ├── core/                   # Core infrastructure
│   │   ├── config.py           # All settings & environment variables
│   │   ├── models.py           # Pydantic data models (LogChunk, QueryResult, etc.)
│   │   ├── database.py         # SQLite database operations
│   │   └── security.py         # Input sanitization & API key management
│   │
│   ├── services/               # Business logic services
│   │   ├── rag.py              # RAG orchestrator (main entry point)
│   │   ├── helix.py            # Helix Vector: Drain3 + Markov chain anomaly detection
│   │   ├── embedding.py        # Sentence-transformers embedding service
│   │   ├── vectorstore.py      # FAISS vector database
│   │   ├── llm.py              # LLM client (Ollama local / OpenRouter cloud)
│   │   ├── indexer.py          # Log file parser & chunking
│   │   ├── intent.py           # Query intent classification
│   │   ├── log_watcher.py      # File system watcher for live logs
│   │   └── eventviewer.py      # Windows Event Viewer integration
│   │
│   ├── integrations/           # External service integrations
│   │   ├── base.py             # Base integration class
│   │   ├── vercel.py           # Vercel logs integration
│   │   ├── posthog.py          # PostHog analytics integration
│   │   └── datadog.py          # DataDog logs integration
│   │
│   ├── ui/                     # UI components
│   │
│   ├── streamlit_app.py        # Main chat interface
│   ├── anomaly_dashboard.py    # Dedicated anomaly detection dashboard
│   └── cli.py                  # Command line interface
│
├── tests/                      # Test suite
│   ├── test_helix.py           # Helix Vector tests
│   ├── test_rag.py             # RAG service tests
│   ├── test_embedding.py       # Embedding service tests
│   ├── test_vectorstore.py     # Vector store tests
│   └── ...
│
├── requirements.txt            # Python dependencies
└── setup.py                    # Package installation
```

---

## 📦 Installation

### Prerequisites

- **Python 3.10+**
- **Ollama** (for local LLM) — [Install Ollama](https://ollama.ai/download)
- **Git**

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/sentry-ai-python.git
cd sentry-ai-python/sentry-ai
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install and Start Ollama

```bash
# Install a lightweight model (recommended for CPU)
ollama pull gemma3:1b

# Or for better quality (requires more RAM)
ollama pull llama3:8b
```

### Step 5: Run the Application

**Chat Interface:**
```bash
streamlit run sentry/streamlit_app.py
```

**Anomaly Dashboard:**
```bash
streamlit run sentry/anomaly_dashboard.py
```

---

## ⚙️ Configuration

All settings can be configured via environment variables with the `SENTRY_` prefix.

### Environment Variables

Create a `.env` file in the `sentry/` directory:

```env
# ===== LLM SETTINGS =====
SENTRY_LLM_MODEL=gemma3:1b              # Ollama model name
SENTRY_OLLAMA_HOST=http://localhost:11434
SENTRY_LLM_TEMPERATURE=0.55              # 0=factual, 1=creative
SENTRY_LLM_MAX_TOKENS=1024

# ===== OPENROUTER CLOUD (Optional) =====
SENTRY_OPENROUTER_API_KEY=sk-or-v1-xxx   # Get from https://openrouter.ai/keys
SENTRY_OPENROUTER_MODEL=qwen/qwen3-coder:free
SENTRY_USE_CLOUD_LLM=true                # true=cloud, false=local Ollama

# ===== EMBEDDING MODEL =====
SENTRY_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
SENTRY_EMBEDDING_DIMENSION=384

# ===== VECTOR SEARCH =====
SENTRY_TOP_K_RESULTS=25                  # Number of results to return
SENTRY_SIMILARITY_THRESHOLD=0.25         # Minimum similarity (0-1)
SENTRY_USE_RERANKING=true                # BM25 reranking after vector search

# ===== HELIX VECTOR (Anomaly Detection) =====
SENTRY_HELIX_ENABLED=true
SENTRY_HELIX_ANOMALY_THRESHOLD=0.20      # Transitions below this = anomaly
SENTRY_HELIX_DRAIN_SIM_TH=0.4            # Drain3 similarity (lower = more clusters)
SENTRY_HELIX_DRAIN_DEPTH=4               # Parse tree depth

# Severity weights (used in anomaly scoring)
SENTRY_HELIX_SEVERITY_FATAL=0.9
SENTRY_HELIX_SEVERITY_CRITICAL=0.8
SENTRY_HELIX_SEVERITY_ERROR=0.5
SENTRY_HELIX_SEVERITY_WARNING=0.3

# ===== STORAGE =====
SENTRY_DATA_DIR=~/.sentry-ai             # Where to store data
SENTRY_LOG_LEVEL=INFO                    # DEBUG, INFO, WARNING, ERROR

# ===== INDEXING =====
SENTRY_MAX_FILE_SIZE_MB=100              # Skip files larger than this
SENTRY_CHUNK_SIZE=1000                   # Characters per chunk
SENTRY_CHUNK_OVERLAP=100                 # Overlap between chunks
```

### Configuration Reference

| Category | Variable | Default | Description |
|----------|----------|---------|-------------|
| **LLM** | `SENTRY_LLM_MODEL` | `gemma3:1b` | Ollama model for generation |
| **LLM** | `SENTRY_LLM_TEMPERATURE` | `0.55` | Creativity level (0-1) |
| **LLM** | `SENTRY_USE_CLOUD_LLM` | `true` | Toggle cloud/local LLM |
| **Cloud** | `SENTRY_OPENROUTER_API_KEY` | — | OpenRouter API key |
| **Cloud** | `SENTRY_OPENROUTER_MODEL` | `qwen/qwen3-coder:free` | Cloud model to use |
| **Embedding** | `SENTRY_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| **Search** | `SENTRY_TOP_K_RESULTS` | `25` | Max results per query |
| **Search** | `SENTRY_SIMILARITY_THRESHOLD` | `0.25` | Min similarity score |
| **Helix** | `SENTRY_HELIX_ANOMALY_THRESHOLD` | `0.20` | Anomaly detection threshold |
| **Helix** | `SENTRY_HELIX_DRAIN_SIM_TH` | `0.4` | Drain3 clustering threshold |
| **Storage** | `SENTRY_DATA_DIR` | `~/.sentry-ai` | Data directory path |

---

## 🚀 Usage

### Chat Interface

The main Streamlit interface provides a chat-based interaction:

```bash
streamlit run sentry/streamlit_app.py
```

**Features:**
- Natural language queries about your logs
- Automatic intent detection (errors, patterns, frequencies, etc.)
- Source citation for each answer
- Toggle between local (Ollama) and cloud (OpenRouter) LLM

**Example Queries:**
- "What errors occurred in the last hour?"
- "Why did the database timeout happen?"
- "Show me the most common log patterns"
- "Find logs similar to 'connection refused'"

### Anomaly Dashboard

Dedicated dashboard for anomaly detection and visualization:

```bash
streamlit run sentry/anomaly_dashboard.py
```

**Features:**
- Upload log files or watch folders
- Real-time Drain3 template mining
- Transition probability heatmaps
- Anomaly score distribution charts
- Interactive template explorer
- Vector search without LLM

### API Mode

For headless/programmatic use:

```bash
uvicorn sentry.api:app --host 0.0.0.0 --port 8000
```

**Endpoints:**
- `POST /api/query` — RAG query
- `POST /api/index` — Index new logs
- `GET /api/stats` — System statistics

---

## 📊 Data Models

### LogChunk (Core Unit)

The atomic unit of the RAG system:

```python
class LogChunk(BaseModel):
    id: str                          # Unique identifier
    source_id: str                   # Reference to LogSource
    content: str                     # The actual log text
    timestamp: datetime              # When the log occurred
    log_level: LogLevel              # CRITICAL, ERROR, WARNING, INFO, DEBUG
    
    # Helix Vector annotations
    cluster_id: int                  # DNA cluster from Drain3
    cluster_template: str            # Pattern template (e.g., "Error in <*>")
    is_anomaly: bool                 # Flagged as anomalous?
    anomaly_type: str                # e.g., "database_timeout", "connection_error"
    anomaly_score: float             # 0.0 (normal) to 1.0 (highly anomalous)
    severity_weight: float           # Keyword-based severity
    transition_prob: float           # Markov chain probability
```

### Query Intents

The intent classifier routes queries to optimal data sources:

| Intent | Description | Routing |
|--------|-------------|---------|
| `FREQUENCY` | "How often do errors occur?" | Helix cluster statistics |
| `ERROR` | "What errors happened?" | Anomaly-filtered vector search |
| `ANOMALY` | "Show me unusual patterns" | Helix anomaly summary |
| `WHY` | "Why did X happen?" | Raw logs with context |
| `SIMILAR` | "Find logs like this" | Pure vector similarity |
| `TIMELINE` | "What happened before X?" | Time-ordered search |
| `GENERAL` | Everything else | Standard RAG |

---

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_helix.py -v

# Run with coverage
pytest tests/ --cov=sentry --cov-report=html
```

### Test Files

| File | Description |
|------|-------------|
| `test_helix.py` | Helix Vector: Drain3 clustering, Markov chains, anomaly detection |
| `test_rag.py` | RAG service: query flow, reranking, intent routing |
| `test_embedding.py` | Embedding service: text-to-vector, similarity |
| `test_vectorstore.py` | FAISS operations: add, search, persistence |
| `test_indexer.py` | Log parsing: .log, .txt, .csv, semantic chunking |
| `test_llm.py` | LLM client: local Ollama, cloud OpenRouter |
| `test_database.py` | SQLite operations |
| `test_log_watcher.py` | File system monitoring |

---

## 🔧 Troubleshooting

### Ollama Not Running

```
Error: Connection refused to http://localhost:11434
```

**Solution:** Start Ollama:
```bash
ollama serve
```

### Model Not Found

```
Error: Model 'gemma3:1b' not found
```

**Solution:** Pull the model:
```bash
ollama pull gemma3:1b
```

### CUDA Out of Memory

**Solution:** Use a smaller model or CPU mode:
```env
SENTRY_LLM_MODEL=gemma3:1b  # Smaller model
```

### Slow Embedding

**Solution:** Reduce batch size:
```env
SENTRY_EMBEDDING_BATCH_SIZE=16
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[Drain3](https://github.com/logpai/Drain3)** — Log parsing and template mining
- **[Sentence-Transformers](https://www.sbert.net/)** — Semantic embeddings
- **[FAISS](https://github.com/facebookresearch/faiss)** — Efficient similarity search
- **[Ollama](https://ollama.ai/)** — Local LLM inference
- **[OpenRouter](https://openrouter.ai/)** — Cloud LLM access
- **[Streamlit](https://streamlit.io/)** — UI framework

---

<p align="center">
  <strong>Built with ❤️ for privacy-conscious developers who want AI-powered log analysis without cloud dependencies.</strong>
</p>