<h1>SentryAI</h1>

<p>
  <strong>AI and Math powered Log Analysis & Root Cause Detection</strong>
</p>


<p>Follow me on <a href="https://x.com/rebalhassan">X</a> I might post about this project there</p>

![Anomaly Dashboard](<sentry-ai/assets/Anomalydashboard.png>)
---
![RAG Chat Interface(btw this is a 1b model)](<sentry-ai/assets/Screenshot 2025-12-20 000550.png>)

## Installation

### Prerequisites

- **Python 3.10+**
- **Ollama** (for local LLM) — [Install Ollama](https://ollama.ai/download)
- **Packages in requirements.txt file**

### Step 1: Clone the Repository

```bash
git clone https://github.com/rebalhassan/sentry-ai-python.git
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

### Step 4: Install and Start Ollama and configure the model in config.py file

```bash
# Install a lightweight model (recommended for CPU)
ollama pull gemma3:1b (or any other model you wanna use available in Ollama)
```
You have to configure the model in config.py file with the exact name of the model you installed you can find it using Ollama list command

### Step 5: Run the Application

**Chat Interface:**
```bash
streamlit run sentry/streamlit_app.py
```

**Anomaly Dashboard:**
```bash
streamlit run sentry/anomaly_dashboard.py
```

## What is SentryAI?

SentryAI is a log analysis tool that combines:

- **RAG (Retrieval Augmented Generation)** for chatting with your logs
- **Helix Vector** — a novel DNA-like encoding system for log pattern mining
- **Markov Chain Anomaly Detection** for identifying unusual log sequences
- **Semantic Vector Search** for finding similar log entries

Unlike cloud-based solutions, SentryAI can run entirely on your machine if you want to. Your logs, Your business.

---

## The Science Behind Helix Vector

### DNA Encoding: How It Works

SentryAI uses a bioinformatics-inspired approach to log analysis. The **Helix Vector** system treats logs like genetic sequences:

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

This transforms millions of unique log lines into a manageable vocabulary of templates.

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

Each log entry receives an **anomaly score** combining statistical rarity and semantic severity:

```
effective_probability = transition_probability × (1 - severity_weight)
anomaly_score = 1 - effective_probability
```

**How it works:**
- **Rare transitions** (low `transition_probability`) → Higher anomaly score
- **Severe keywords** (ERROR, CRITICAL, FATAL) → Higher `severity_weight` → Lower `effective_probability` → Higher anomaly score
- Logs are flagged when `effective_probability < 0.20` (default threshold, you can change it in the config)

**Example:**
- Normal log: `P=0.85`, `S=0.1` → `P_eff = 0.765` → `A = 0.235` (not flagged)
- Anomalous log: `P=0.15`, `S=0.8` → `P_eff = 0.03` → `A = 0.97` (flagged!)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              SentryAI                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────┐    ┌──────────────┐                                   │
│   │   Streamlit │    │   Anomaly    │                                   │
│   │   Chat UI   │    │  Dashboard   │                                   │
│   └──────┬──────┘    └──────┬───────┘                                   │
│         │                  │                                            │
│         └──────────────────┘                                            │
│                            ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐ │ 
│  │                        RAG Service                                   │ │
│  │   ┌──────────┐  ┌────────────┐  ┌──────────┐  ┌─────────────────┐   │ │
│  │   │ Embedder │  │ VectorStore│  │   LLM    │  │ Intent Classifier│  │ │
│  │   │(MiniLM)  │  │  (FAISS)   │  │  Ollama/ │  │   (Regex-based)  │  │ │
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

## Project Structure

```
sentry-ai/
├── sentry/                     # Main application package
│   ├── core/                   # Foundation: config, data models, database, security
│   ├── services/               # All system services: RAG, Helix, LLM, embeddings, etc.         
│   ├── streamlit_app.py        # Chat interface
│   └── anomaly_dashboard.py    # Anomaly detection dashboard
│
├── tests/                      # Comprehensive test suite
└── requirements.txt            # Python dependencies
```

**Key Components:**
- **`core/`**: Configuration (`config.py`), data models (`models.py`), database operations (`database.py`), and security utilities
- **`services/`**: RAG orchestration, Helix Vector, LLM clients, embeddings, vector stores, indexing, and intent classification

---

## Configuration

All settings can be configured via `config.py` file in the core directory.


## Usage

### Chat Interface

The main Streamlit interface provides a chat-based interaction:

```bash
streamlit run sentry/streamlit_app.py
```

**Features:**
- Chat with your logs
- Automatic intent detection (errors, patterns, frequencies, etc.)
- Dual-vector system with routing based off intent
- Toggle between local (Ollama) and cloud (OpenRouter) LLM


### Anomaly Dashboard

Dedicated dashboard for anomaly detection and visualization:

```bash
streamlit run sentry/anomaly_dashboard.py
```

**Features:**
- Drag and drop or upload log files (supported formats: `.log`, `.txt`, `.csv`, `.out`, `.err`)
- Watch folders for real-time log monitoring
- Drain3 template mining
- Transition probability heatmaps
- Anomaly score distribution charts
- Interactive template explorer
- Vector search without LLM

## Data Models

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

## Testing

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


## Acknowledgments

- **[Drain3](https://github.com/logpai/Drain3)** — Log parsing and template mining
- **[Sentence-Transformers](https://www.sbert.net/)** — Semantic embeddings
- **[FAISS](https://github.com/facebookresearch/faiss)** — Efficient similarity search
- **[Ollama](https://ollama.ai/)** — Local LLM inference
- **[OpenRouter](https://openrouter.ai/)** — Cloud LLM access
- **[Streamlit](https://streamlit.io/)** — UI framework
