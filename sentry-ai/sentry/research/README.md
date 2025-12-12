# Log DNA: Turning Logs into DNA-like Sequences 🧬

## The Big Idea (In Simple Terms)

Imagine you have thousands of log messages like:
```
User john123 logged in from 192.168.1.1
User alice99 logged in from 10.0.0.5
Error: Connection failed to database server-01
User bob42 logged in from 172.16.0.1
Error: Connection failed to database server-02
```

**Problem**: These logs are messy and repetitive. An LLM would waste tokens reading all this noise.

**Solution**: Turn them into a compact "DNA string" like:
```
1-1-2-1-2
```

Where:
- **Cluster 1** = "User <*> logged in from <*>" (login events)
- **Cluster 2** = "Error: Connection failed to database <*>" (database errors)

Now the LLM sees a **pattern**, not noise! It's like compressing a book into its chapter structure.

---

## How Drain3 Works (Think of a Librarian)

Imagine a librarian sorting books:

1. **First Pass**: A new log comes in → "Never seen this pattern before" → Create a new shelf (cluster)
2. **Second Pass**: Similar log comes in → "This looks like something I've seen" → Put it on the same shelf
3. **Learning**: Over time, the librarian builds a "template" for each shelf

**Example**:
| Log Message | Template Created | Cluster ID |
|-------------|-----------------|------------|
| `connected to 10.0.0.1` | `connected to <IP>` | 1 |
| `connected to 192.168.1.1` | (matches existing) | 1 |
| `user login: admin` | `user login: <*>` | 2 |

---

## What We're Building

### The Pipeline

```
Raw Logs → Drain3 Parser → Cluster IDs → DNA String
    ↓
[Full log text]  →  [Pattern matching]  →  [1,1,2,3,1,2]  →  "1-1-2-3-1-2"
```

### What Gets Saved

1. **Cluster Dictionary** (the "codebook")
   ```json
   {
     "1": {"template": "User <*> logged in from <*>", "count": 1523},
     "2": {"template": "Error: Connection failed to <*>", "count": 42},
     "3": {"template": "Transaction <*> completed in <*>ms", "count": 8901}
   }
   ```

2. **DNA String** (the compressed representation)
   ```
   "1-1-1-2-1-1-3-3-2-1-3-3-3-1-2"
   ```

---

## Why This Helps LLMs

### Before (wasteful)
```
"Here are your logs:
User john123 logged in from 192.168.1.1
User alice99 logged in from 10.0.0.5
User bob42 logged in from 172.16.0.1
..."
(1000s of tokens wasted on repetitive text)
```

### After (smart)
```
"Log DNA: 1-1-1-2-1-1-3-3-2-1-3-3-3-1-2

Codebook:
1 = User login event (1523 occurrences)
2 = Database connection error (42 occurrences)  
3 = Transaction completed (8901 occurrences)

What patterns do you see?"
```

The LLM can now:
- See the **sequence** of events (temporal patterns)
- Count **frequency** of each event type
- Spot **anomalies** (e.g., "why do errors always follow logins?")

---

## Simple Implementation Plan

### Phase 1: Core Components 🔧

| File | Purpose |
|------|---------|
| `log_dna_encoder.py` | Main class that uses Drain3 to encode logs |
| `cluster_store.py` | Saves/loads the cluster dictionary (codebook) |
| `dna_formatter.py` | Converts cluster IDs(encoded logs) to DNA strings |

### Phase 2: Integration (Later - Skip for now) 🔗

- Connect to existing DataDog/Vercel integrations
- Store DNA alongside raw logs in our data models
- Add DNA context to RAG queries

---

## Files We'll Create

```
sentry/research/
├── README.md           # This file (you're reading it!)
├── __init__.py         # Package init
├── log_dna_encoder.py  # Main Drain3 wrapper
├── cluster_store.py    # Cluster persistence
├── dna_formatter.py    # DNA string formatting
└── example_usage.py    # Demo script
```

---

## Dependencies - I'll install it later

```bash
pip install drain3
```

---

## Quick Preview (What the Code Will Look Like)

```python
from research.log_dna_encoder import LogDNAEncoder

# Create encoder
encoder = LogDNAEncoder()

# Feed it logs
logs = [
    "User john logged in from 192.168.1.1",
    "User alice logged in from 10.0.0.2", 
    "Error: DB connection failed",
    "User bob logged in from 172.16.0.1",
]

# Get DNA string
dna = encoder.encode(logs)
print(dna)  # Output: "1-1-2-1"

# Get the codebook
codebook = encoder.get_codebook()
print(codebook)
# {1: "User <*> logged in from <*>", 2: "Error: DB connection failed"}
```

---

## Next Steps

Once you approve this plan, I'll implement:

1. ✅ `log_dna_encoder.py` - The main Drain3 wrapper
2. ✅ `cluster_store.py` - JSON-based storage for clusters
3. ✅ `dna_formatter.py` - DNA string utilities
4. ✅ `example_usage.py` - Working demo with sample logs

**Ready to proceed?** 🚀
