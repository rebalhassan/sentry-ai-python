"""
LLM Log Analysis Agent - Research Module
=========================================

An LLM-powered agent that can analyze logs by querying two data sources:

1. **Trace Vector DB** - For transition frequencies
   Query: "How often does X happen?" / "What's the frequency of Y?"
   Returns: Transition probabilities, pattern counts
   
2. **Raw Logs Vector DB** - For root cause context
   Query: "Why did X happen?" / "What caused Y?"
   Returns: Actual log lines around the anomaly

## How It Works

1. User asks a question about logs
2. Regex intent detection classifies the query
3. Appropriate data source is queried
4. Context is built and sent to LLM
5. LLM generates an answer

## Intent Types

- FREQUENCY: "How often...", "How many times...", "Count of..."
- WHY: "Why did...", "What caused...", "Explain..."
- ERROR: "What errors...", "Show anomalies...", "Problems..."
- GENERAL: Everything else

Usage:
    agent = LogAnalysisAgent()
    agent.index_logs(logs)
    response = agent.query("Why did the database timeout happen?")
    print(response)
"""

import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import os
import json

# For Ollama HTTP client
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Import our research modules
from anomaly_detector import AnomalyDetector, AnomalyInfo, AnnotatedChunk
from log_dna import LogDNAEncoder, EMBEDDINGS_AVAILABLE
from trace_vectorstore import TraceVectorStore


class OllamaClient:
    """
    Client for Ollama local LLM inference.
    
    Ollama runs LLMs locally and exposes a REST API at localhost:11434.
    
    ## Setup
    
    1. Install Ollama: https://ollama.ai
    2. Pull a model: `ollama pull llama3.2` or `ollama pull mistral`
    3. Ollama runs automatically in background
    
    ## Usage
    
        client = OllamaClient(model="llama3.2")
        response = client.generate("What is 2+2?")
        print(response)
    """
    
    def __init__(
        self, 
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: int = 120
    ):
        """
        Initialize Ollama client.
        
        Args:
            model: Model name (e.g., "llama3.2", "mistral", "codellama")
            base_url: Ollama API URL (default: localhost:11434)
            timeout: Request timeout in seconds
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library not installed. Run: pip install requests")
        
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        
        # Verify connection
        self._check_connection()
    
    def _check_connection(self):
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError(f"Ollama returned status {response.status_code}")
            
            # Check if model is available
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            
            if self.model not in model_names and f"{self.model}:latest" not in [m.get("name") for m in models]:
                print(f"⚠️  Model '{self.model}' not found. Available: {model_names}")
                print(f"   Run: ollama pull {self.model}")
            else:
                print(f"✅ Ollama connected. Model: {self.model}")
                
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}\n"
                "Make sure Ollama is running:\n"
                "  1. Install from https://ollama.ai\n"
                "  2. Run: ollama serve\n"
                "  3. Pull a model: ollama pull llama3.2"
            )
    
    def generate(self, prompt: str, stream: bool = False) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: The input prompt
            stream: Whether to stream the response (not implemented)
            
        Returns:
            The generated text response
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,  # Get complete response
            "options": {
                "temperature": 0.7,
                "num_predict": 1024,  # Max tokens
            }
        }
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.Timeout:
            return f"[Ollama timeout after {self.timeout}s - try a smaller model]"
        except requests.exceptions.RequestException as e:
            return f"[Ollama error: {e}]"
    
    def chat(self, messages: List[Dict[str, str]]) -> str:
        """
        Chat completion API (for multi-turn conversations).
        
        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            
        Returns:
            Assistant's response
        """
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("message", {}).get("content", "")
            
        except requests.exceptions.RequestException as e:
            return f"[Ollama error: {e}]"
    
    def __repr__(self):
        return f"<OllamaClient(model={self.model}, url={self.base_url})>"


class QueryIntent(Enum):
    """Types of queries the agent can handle."""
    FREQUENCY = "frequency"      # How often does X happen?
    WHY = "why"                  # Why did X happen?
    ERROR = "error"              # What errors occurred?
    SIMILAR = "similar"          # Find similar patterns
    GENERAL = "general"          # General question


@dataclass
class QueryResult:
    """Result from querying the agent."""
    intent: QueryIntent
    query: str
    context: str              # Built context for LLM
    anomalies: List[AnomalyInfo]
    raw_logs: List[str]
    answer: Optional[str] = None


class IntentClassifier:
    """
    Regex-based intent classifier for log analysis queries.
    
    This is a simple rule-based classifier that we'll later replace with TinyBERT.
    It looks for keywords and patterns to determine what kind of analysis is needed.
    """
    
    # Regex patterns for each intent type
    PATTERNS = {
        QueryIntent.FREQUENCY: [
            r'\bhow (often|many|frequently)\b',
            r'\bfrequency\b',
            r'\bcount\s*(of)?\b',
            r'\bhow many times\b',
            r'\boccurrences?\b',
            r'\bstatistics?\b',
            r'\brate\s*(of)?\b',
            r'\bprobabilit(y|ies)\b',
            r'\btransition\b',
        ],
        QueryIntent.WHY: [
            r'\bwhy\b',
            r'\bwhat caused\b',
            r'\bexplain\b',
            r'\breason\s*(for|why)?\b',
            r'\broot cause\b',
            r'\bwhat happened\b',
            r'\bwhat led to\b',
            r'\bhow did .*happen\b',
            r'\bcontext\b',
        ],
        QueryIntent.ERROR: [
            r'\berror(s)?\b',
            r'\banomal(y|ies|ous)\b',
            r'\bfailure(s)?\b',
            r'\bproblem(s)?\b',
            r'\bissue(s)?\b',
            r'\bcrash(es|ed)?\b',
            r'\bexception(s)?\b',
            r'\bbug(s)?\b',
            r'\bfatal\b',
            r'\bcritical\b',
        ],
        QueryIntent.SIMILAR: [
            r'\bsimilar\b',
            r'\blike this\b',
            r'\bpattern(s)?\b',
            r'\brelated\b',
            r'\bother .*like\b',
            r'\bfind .*matching\b',
        ],
    }
    
    def classify(self, query: str) -> Tuple[QueryIntent, float]:
        """
        Classify a query into an intent type.
        
        Args:
            query: The user's question
            
        Returns:
            Tuple of (intent, confidence)
            - intent: The classified QueryIntent
            - confidence: How confident we are (0.0 to 1.0)
        
        ## How It Works
        
        1. Lowercase the query
        2. Check each intent's regex patterns
        3. Count matches for each intent
        4. Return the intent with most matches
        5. Confidence = matches / total_patterns
        """
        query_lower = query.lower()
        
        scores: Dict[QueryIntent, int] = {}
        
        for intent, patterns in self.PATTERNS.items():
            matches = sum(
                1 for pattern in patterns 
                if re.search(pattern, query_lower)
            )
            scores[intent] = matches
        
        # Find best match
        if not any(scores.values()):
            return QueryIntent.GENERAL, 0.0
        
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]
        
        # Confidence: proportion of patterns matched (capped at 1.0)
        max_patterns = len(self.PATTERNS.get(best_intent, []))
        confidence = min(best_score / max(max_patterns, 1), 1.0)
        
        return best_intent, confidence
    
    def extract_subject(self, query: str) -> str:
        """
        Extract the subject of the query (what are they asking about?).
        
        Examples:
            "Why did the database timeout?" → "database timeout"
            "How often does authentication fail?" → "authentication fail"
        """
        query_lower = query.lower()
        
        # Patterns to remove (question words, etc.)
        remove_patterns = [
            r'^(why|how|what|when|where|who|which)\s+',
            r'^(did|does|do|is|are|was|were|has|have|had)\s+',
            r'^(the|a|an)\s+',
            r'\?$',
            r'\bhappen(ed|s|ing)?\b',
            r'\boccur(red|s|ring)?\b',
            r'\bcause(d|s)?\b',
            r'\boften\b',
            r'\bmany\b',
            r'\btimes?\b',
        ]
        
        subject = query_lower
        for pattern in remove_patterns:
            subject = re.sub(pattern, ' ', subject)
        
        # Clean up whitespace
        subject = ' '.join(subject.split())
        
        return subject if subject else query


class LogAnalysisAgent:
    """
    LLM-powered log analysis agent.
    
    This agent can:
    1. Index logs with anomaly detection
    2. Answer questions about log patterns
    3. Query appropriate data sources based on intent
    4. Use LLM to generate human-readable answers
    
    ## Architecture
    
    User Query → Intent Detection → Data Source Query → Context Building → LLM → Answer
                      ↓
           ┌─────────┴─────────┐
           ↓                   ↓
    Trace Vector DB      Raw Logs DB
    (frequencies)        (context)
    """
    
    def __init__(self, llm_client=None, model_name: str = None):
        """
        Initialize the agent.
        
        Args:
            llm_client: Optional LLM client. If None, uses mock responses.
            model_name: Model name for embeddings
        """
        # Core components
        self.encoder = LogDNAEncoder()
        self.detector = AnomalyDetector(anomaly_threshold=0.20)
        self.classifier = IntentClassifier()
        
        # Vector store for trace embeddings
        self.trace_store = TraceVectorStore(dimension=384)
        
        # Storage for indexed data
        self.indexed_logs: List[str] = []
        self.cluster_ids: List[int] = []
        self.annotated_chunks: List[AnnotatedChunk] = []
        
        # LLM client (optional - mock if not provided)
        self.llm_client = llm_client
        
        # Indexing state
        self._indexed = False
    
    def index_logs(self, logs: List[str], session_id: str = "session_001"):
        """
        Index logs for analysis.
        
        This:
        1. Encodes logs into DNA sequences (Drain3)
        2. Computes transition probabilities
        3. Detects anomalies
        4. Creates context-stuffed chunks
        5. Stores in trace vector DB
        
        Args:
            logs: List of raw log lines
            session_id: Unique identifier for this log session
        """
        print(f"\n📥 Indexing {len(logs)} logs...")
        
        # Store raw logs
        self.indexed_logs = logs
        
        # Step 1: Encode with Drain3
        self.cluster_ids = self.encoder.encode_logs(logs)
        codebook = self.encoder.get_codebook()
        
        print(f"   Encoded into {len(codebook)} clusters")
        
        # Step 2: Fit anomaly detector
        self.detector.fit(self.cluster_ids, codebook)
        
        # Step 3: Create annotated chunks with context
        self.annotated_chunks = self.detector.create_annotated_chunks(
            logs, self.cluster_ids, window_size=2
        )
        
        # Count anomalies
        anomaly_count = sum(1 for c in self.annotated_chunks if c.is_anomaly)
        print(f"   Detected {anomaly_count} anomalies")
        
        # Step 4: Store trace in vector DB
        if EMBEDDINGS_AVAILABLE:
            self.encoder.init_embeddings()
            trace_embedding = self.encoder.get_trace_vector(self.cluster_ids)
            
            # Get transition probabilities for metadata
            transitions = {}
            for from_c, to_dict in self.detector.transition_probs.items():
                for to_c, prob in to_dict.items():
                    transitions[f"{from_c}->{to_c}"] = prob
            
            # Get anomalies for metadata
            anomalies = self.detector.detect(self.cluster_ids)
            anomaly_types = list(set(a.anomaly_type for a in anomalies))
            
            self.trace_store.add_trace(
                trace_id=session_id,
                embedding=trace_embedding,
                metadata={
                    "cluster_ids": self.cluster_ids,
                    "dna_string": self.encoder.to_dna_string(self.cluster_ids),
                    "raw_logs": logs,
                    "transitions": transitions,
                    "anomaly_count": anomaly_count,
                    "anomaly_types": anomaly_types,
                }
            )
            print(f"   Stored in trace vector DB")
        
        self._indexed = True
        print(f"✅ Indexing complete!")
    
    def _build_frequency_context(self, subject: str) -> str:
        """
        Build context for FREQUENCY queries.
        
        This provides:
        - Transition probabilities
        - How often each pattern occurs
        - Anomaly statistics
        """
        if not self._indexed:
            return "No logs indexed yet."
        
        lines = ["## Transition Frequency Analysis\n"]
        
        # Overall stats
        total_events = len(self.cluster_ids)
        anomaly_count = sum(1 for c in self.annotated_chunks if c.is_anomaly)
        lines.append(f"Total events: {total_events}")
        lines.append(f"Anomalies detected: {anomaly_count}")
        lines.append(f"Anomaly rate: {anomaly_count/total_events:.1%}\n")
        
        # Transition probabilities
        lines.append("### Transition Probabilities\n")
        codebook = self.encoder.get_codebook()
        
        for from_c in sorted(self.detector.transition_probs.keys()):
            from_template = codebook.get(from_c, {}).get("template", f"Cluster {from_c}")
            lines.append(f"From: {from_template[:50]}...")
            
            for to_c, prob in sorted(
                self.detector.transition_probs[from_c].items(),
                key=lambda x: -x[1]
            ):
                to_template = codebook.get(to_c, {}).get("template", f"Cluster {to_c}")
                severity = self.detector._get_severity_penalty(to_template)
                flag = "⚠️" if prob < 0.2 or severity >= 0.5 else ""
                lines.append(f"  → {to_template[:40]}... : {prob:.1%} {flag}")
            lines.append("")
        
        # Filter by subject if provided
        if subject:
            relevant = [l for l in lines if subject.lower() in l.lower()]
            if relevant:
                return "\n".join(lines[:5] + relevant)  # Keep header + relevant lines
        
        return "\n".join(lines)
    
    def _build_why_context(self, subject: str) -> str:
        """
        Build context for WHY queries.
        
        This provides:
        - Anomalous chunks with surrounding context
        - Template patterns
        - Severity information
        """
        if not self._indexed:
            return "No logs indexed yet."
        
        lines = ["## Root Cause Analysis Context\n"]
        
        # Get all anomalies
        anomalies = [c for c in self.annotated_chunks if c.is_anomaly]
        
        if not anomalies:
            return "No anomalies detected in the indexed logs."
        
        # Filter by subject if provided
        if subject:
            anomalies = [
                a for a in anomalies 
                if subject.lower() in a.text.lower() or 
                   subject.lower() in (a.anomaly_type or "").lower()
            ]
        
        lines.append(f"Found {len(anomalies)} relevant anomalies:\n")
        
        for i, chunk in enumerate(anomalies[:5]):  # Limit to 5
            lines.append(f"### Anomaly {i+1}: {chunk.anomaly_type}")
            lines.append(f"Severity: {chunk.anomaly.severity_weight:.1f}")
            lines.append(f"Score: {chunk.anomaly_score:.2f}")
            lines.append(f"Transition: {chunk.anomaly.transition_prob:.1%} probability")
            lines.append("\nContext (log lines around the anomaly):")
            
            for j, log_line in enumerate(chunk.log_lines):
                marker = ">>>" if j == chunk.center_index else "   "
                lines.append(f"  {marker} {log_line}")
            
            lines.append("")
        
        # Add codebook for reference
        lines.append("### Template Reference")
        codebook = self.encoder.get_codebook()
        for cid, info in codebook.items():
            template = info.get("template", "Unknown")
            count = info.get("count", 0)
            lines.append(f"  Cluster {cid} ({count}x): {template}")
        
        return "\n".join(lines)
    
    def _build_error_context(self, subject: str) -> str:
        """
        Build context for ERROR queries.
        
        Lists all detected errors and anomalies.
        """
        if not self._indexed:
            return "No logs indexed yet."
        
        lines = ["## Error Summary\n"]
        
        # Get all anomalies
        anomalies = self.detector.detect(self.cluster_ids)
        
        if not anomalies:
            return "No errors or anomalies detected."
        
        # Group by type
        by_type: Dict[str, List[AnomalyInfo]] = {}
        for a in anomalies:
            by_type.setdefault(a.anomaly_type, []).append(a)
        
        lines.append(f"Total anomalies: {len(anomalies)}\n")
        
        for anomaly_type, anomaly_list in sorted(by_type.items()):
            lines.append(f"### {anomaly_type} ({len(anomaly_list)} occurrences)")
            
            for a in anomaly_list[:3]:  # Limit per type
                lines.append(f"  - Index {a.index}: {a.template[:50]}...")
                lines.append(f"    Severity: {a.severity_weight:.1f}, Score: {a.anomaly_score:.2f}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _build_similar_context(self, subject: str) -> str:
        """
        Build context for SIMILAR pattern queries.
        """
        if not self._indexed:
            return "No logs indexed yet."
        
        # For now, just show patterns
        codebook = self.encoder.get_codebook()
        lines = ["## Similar Patterns\n"]
        
        for cid, info in codebook.items():
            template = info.get("template", "Unknown")
            if subject.lower() in template.lower():
                count = info.get("count", 0)
                lines.append(f"Cluster {cid} ({count}x): {template}")
        
        return "\n".join(lines) if len(lines) > 1 else "No similar patterns found."
    
    def _generate_llm_response(self, query: str, context: str, intent: QueryIntent) -> str:
        """
        Generate a response using LLM.
        
        If no LLM client is configured, uses a mock response that explains
        what the LLM would see.
        """
        # Build the prompt
        prompt = f"""You are a log analysis expert. Analyze the following log data and answer the user's question.

## User Question
{query}

## Query Type
{intent.value}

## Log Analysis Data
{context}

## Instructions
- Be concise and direct
- Reference specific log lines when relevant
- Explain technical terms in simple language
- If you see anomalies, explain their significance
- For "why" questions, trace back through the log sequence

## Your Analysis
"""
        
        if self.llm_client:
            # Use real LLM
            try:
                response = self.llm_client.generate(prompt)
                return response
            except Exception as e:
                return f"LLM Error: {e}\n\nContext that would be sent:\n{context[:500]}..."
        else:
            # Mock response - just return the context with explanation
            return f"""[Mock LLM Response - No LLM client configured]

Intent detected: {intent.value}
Question: {query}

Based on the log analysis data, here's what I found:

{context[:1500]}...

[To get real LLM responses, pass an llm_client when creating LogAnalysisAgent]
"""
    
    def query(self, question: str) -> QueryResult:
        """
        Query the agent with a question about logs.
        
        This is the main entry point for asking questions.
        
        Args:
            question: Natural language question about logs
            
        Returns:
            QueryResult with the answer and context
            
        Example:
            >>> agent.query("Why did the database timeout happen?")
            >>> agent.query("How often do authentication errors occur?")
            >>> agent.query("What errors happened in the last session?")
        """
        print(f"\n🔍 Query: {question}")
        
        # Step 1: Classify intent
        intent, confidence = self.classifier.classify(question)
        print(f"   Intent: {intent.value} (confidence: {confidence:.0%})")
        
        # Step 2: Extract subject
        subject = self.classifier.extract_subject(question)
        print(f"   Subject: '{subject}'")
        
        # Step 3: Build context based on intent
        if intent == QueryIntent.FREQUENCY:
            context = self._build_frequency_context(subject)
        elif intent == QueryIntent.WHY:
            context = self._build_why_context(subject)
        elif intent == QueryIntent.ERROR:
            context = self._build_error_context(subject)
        elif intent == QueryIntent.SIMILAR:
            context = self._build_similar_context(subject)
        else:
            # General: combine error + frequency
            context = self._build_error_context(subject) + "\n\n" + self._build_frequency_context(subject)
        
        # Step 4: Get anomalies and raw logs
        anomalies = self.detector.detect(self.cluster_ids) if self._indexed else []
        raw_logs = self.indexed_logs[:10] if self._indexed else []
        
        # Step 5: Generate LLM response
        answer = self._generate_llm_response(question, context, intent)
        
        return QueryResult(
            intent=intent,
            query=question,
            context=context,
            anomalies=anomalies,
            raw_logs=raw_logs,
            answer=answer
        )


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Log Analysis Agent Demo")
    parser.add_argument("--model", default="llama3.2", help="Ollama model to use")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM (no Ollama)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("🤖 LLM LOG ANALYSIS AGENT DEMO (with Ollama)")
    print("=" * 70)
    
    # Sample logs
    logs = [
        "2024-01-15 10:00:01 INFO User john connected from 192.168.1.1",
        "2024-01-15 10:00:02 INFO User john requested dashboard",
        "2024-01-15 10:00:03 INFO Query completed in 45ms",
        "2024-01-15 10:00:04 INFO User john logged out",
        "2024-01-15 10:01:01 INFO User alice connected from 10.0.0.1",
        "2024-01-15 10:01:02 INFO User alice requested analytics",
        "2024-01-15 10:01:03 ERROR Database connection timeout after 30s",
        "2024-01-15 10:01:04 CRITICAL Service crashed - restarting",
        "2024-01-15 10:01:05 INFO Service recovered after restart",
        "2024-01-15 10:02:01 INFO User bob connected from 172.16.0.1",
        "2024-01-15 10:02:02 INFO User bob requested dashboard",
        "2024-01-15 10:02:03 INFO Query completed in 32ms",
        "2024-01-15 10:02:04 INFO User bob logged out",
    ]
    
    # Try to create Ollama client
    llm_client = None
    if not args.mock:
        try:
            print(f"\n🔌 Connecting to Ollama (model: {args.model})...")
            llm_client = OllamaClient(model=args.model)
        except ConnectionError as e:
            print(f"\n⚠️  Ollama not available: {e}")
            print("   Running with mock LLM responses.")
            print("   To use real LLM: install Ollama and run 'ollama pull llama3.2'")
    else:
        print("\n📝 Running in mock mode (--mock flag set)")
    
    # Create agent
    agent = LogAnalysisAgent(llm_client=llm_client)
    
    # Index logs
    agent.index_logs(logs)
    
    # Demo queries
    queries = [
        "Why did the database timeout happen?",
        "How often do errors occur?",
        "What errors happened in this session?",
    ]
    
    for q in queries:
        print("\n" + "=" * 70)
        result = agent.query(q)
        print("\n📝 Answer:")
        print("-" * 40)
        print(result.answer)
    
    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE!")
    print("=" * 70)
    
    if not llm_client:
        print("\n💡 To use with Ollama:")
        print("   1. Install Ollama: https://ollama.ai")
        print("   2. Pull a model: ollama pull llama3.2")
        print("   3. Run this script without --mock")

