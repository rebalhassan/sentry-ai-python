"""
Helix Vector Benchmark - Research Module
=========================================

Side-by-side comparison of:
1. **Helix Vector** - Our approach (DNA encoding + anomaly detection + structured context)
2. **Raw Logs** - Traditional approach (just embed raw log lines)

This demonstrates why pre-processing logs with pattern recognition and
anomaly detection helps small LLMs punch above their weight.

## What We're Testing

- Same logs, same LLM, same question
- Different context strategies:
  - Helix: Clustered templates, transition probabilities, anomaly markers
  - Raw: Just the log lines as-is

## Metrics

- Response quality (manual review)
- Time to first token
- Context token count
- Whether root cause was correctly identified

Usage:
    python benchmark.py --model gemma3:1b
    python benchmark.py --model llama3.2 --questions 5
"""

import time
import argparse
from typing import List, Dict, Any
from dataclasses import dataclass

# Import our modules
from llm_log_agent import LogAnalysisAgent, OllamaClient, EMBEDDINGS_AVAILABLE
from log_dna import LogDNAEncoder
from anomaly_detector import AnomalyDetector


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    approach: str           # "helix" or "raw"
    question: str
    context_length: int     # Characters in context
    response_time: float    # Seconds
    answer: str
    context_preview: str    # First 500 chars of context


def build_raw_context(logs: List[str], question: str) -> str:
    """
    Build context the traditional way - just raw logs.
    
    This is what you'd normally do:
    1. Embed the question
    2. Search for similar log chunks
    3. Pass the raw logs to the LLM
    """
    context = """## Raw Log Data

Here are the log lines from the system:

"""
    for i, log in enumerate(logs):
        context += f"{i+1}. {log}\n"
    
    context += """
## Instructions
Analyze these logs and answer the user's question.
Look for errors, patterns, and root causes.
"""
    return context


def build_helix_context(
    logs: List[str],
    encoder: LogDNAEncoder,
    detector: AnomalyDetector,
    question: str
) -> str:
    """
    Build context the Helix Vector way:
    1. DNA encoding (cluster patterns)
    2. Transition probabilities
    3. Anomaly detection with severity
    4. Structured presentation
    """
    # Encode logs
    cluster_ids = encoder.encode_logs(logs)
    codebook = encoder.get_codebook()
    
    # Fit anomaly detector
    detector.fit(cluster_ids, codebook)
    
    # Detect anomalies
    anomalies = detector.detect(cluster_ids)
    
    # Build structured context
    context = """## Helix Vector Log Analysis

### DNA Sequence
The logs have been encoded into a pattern sequence:
"""
    context += f"DNA: {encoder.to_dna_string(cluster_ids)}\n\n"
    
    # Codebook
    context += "### Pattern Codebook\n"
    for cid, info in codebook.items():
        template = info.get("template", "Unknown")
        count = info.get("count", 0)
        severity = detector._get_severity_penalty(template)
        sev_label = "⚠️ ERROR" if severity >= 0.5 else "CRITICAL" if severity >= 0.8 else ""
        context += f"  Cluster {cid} ({count}x): {template} {sev_label}\n"
    
    # Anomalies
    if anomalies:
        context += "\n### Detected Anomalies\n"
        for a in anomalies:
            context += f"\n**Anomaly at position {a.index}**\n"
            context += f"  - Type: {a.anomaly_type}\n"
            context += f"  - Severity: {a.severity_weight:.1f}\n"
            context += f"  - Transition: [{a.from_cluster}] → [{a.to_cluster}] ({a.transition_prob:.1%} probability)\n"
            context += f"  - Template: {a.template}\n"
            context += f"  - Log: {logs[a.index] if a.index < len(logs) else 'N/A'}\n"
    
    # Transition probabilities
    context += "\n### Transition Probabilities\n"
    for from_c in sorted(detector.transition_probs.keys()):
        from_template = codebook.get(from_c, {}).get("template", f"Cluster {from_c}")[:30]
        for to_c, prob in detector.transition_probs[from_c].items():
            flag = "⚠️ RARE" if prob < 0.2 else ""
            context += f"  [{from_c}] → [{to_c}]: {prob:.0%} {flag}\n"
    
    context += """
### Instructions
Use the pattern analysis above to answer the user's question.
Focus on the anomalies and rare transitions - they indicate problems.
"""
    return context


def run_benchmark(
    logs: List[str],
    questions: List[str],
    llm_client: OllamaClient
) -> List[BenchmarkResult]:
    """Run benchmark comparing both approaches."""
    
    results = []
    
    # Initialize Helix components
    encoder = LogDNAEncoder()
    detector = AnomalyDetector(anomaly_threshold=0.20)
    
    for question in questions:
        print(f"\n{'='*60}")
        print(f"📝 Question: {question}")
        print("="*60)
        
        # === RAW APPROACH ===
        print("\n🔵 RAW LOGS Approach:")
        raw_context = build_raw_context(logs, question)
        
        raw_prompt = f"""You are a log analysis expert. Answer the question based on the log data.

## Question
{question}

{raw_context}

## Your Analysis (be concise)
"""
        
        start_time = time.time()
        raw_answer = llm_client.generate(raw_prompt)
        raw_time = time.time() - start_time
        
        results.append(BenchmarkResult(
            approach="raw",
            question=question,
            context_length=len(raw_context),
            response_time=raw_time,
            answer=raw_answer,
            context_preview=raw_context[:500]
        ))
        
        print(f"   Context: {len(raw_context)} chars")
        print(f"   Time: {raw_time:.1f}s")
        print(f"   Answer preview: {raw_answer[:200]}...")
        
        # === HELIX APPROACH ===
        print("\n🟢 HELIX VECTOR Approach:")
        helix_context = build_helix_context(logs, encoder, detector, question)
        
        helix_prompt = f"""You are a log analysis expert. Answer the question based on the structured analysis.

## Question
{question}

{helix_context}

## Your Analysis (be concise)
"""
        
        start_time = time.time()
        helix_answer = llm_client.generate(helix_prompt)
        helix_time = time.time() - start_time
        
        results.append(BenchmarkResult(
            approach="helix",
            question=question,
            context_length=len(helix_context),
            response_time=helix_time,
            answer=helix_answer,
            context_preview=helix_context[:500]
        ))
        
        print(f"   Context: {len(helix_context)} chars")
        print(f"   Time: {helix_time:.1f}s")
        print(f"   Answer preview: {helix_answer[:200]}...")
    
    return results


def print_comparison(results: List[BenchmarkResult]):
    """Print side-by-side comparison of results."""
    
    print("\n" + "="*80)
    print("📊 BENCHMARK RESULTS COMPARISON")
    print("="*80)
    
    # Group by question
    questions = list(set(r.question for r in results))
    
    for question in questions:
        raw_result = next(r for r in results if r.question == question and r.approach == "raw")
        helix_result = next(r for r in results if r.question == question and r.approach == "helix")
        
        print(f"\n{'─'*80}")
        print(f"❓ {question}")
        print(f"{'─'*80}")
        
        # Metrics table
        print(f"\n{'Metric':<25} {'Raw Logs':<25} {'Helix Vector':<25}")
        print(f"{'-'*25} {'-'*25} {'-'*25}")
        print(f"{'Context Size':<25} {raw_result.context_length:>20} chars {helix_result.context_length:>20} chars")
        print(f"{'Response Time':<25} {raw_result.response_time:>20.1f}s {helix_result.response_time:>20.1f}s")
        
        # Answers
        print(f"\n🔵 RAW LOGS Answer:")
        print(f"   {raw_result.answer[:400]}...")
        
        print(f"\n🟢 HELIX VECTOR Answer:")
        print(f"   {helix_result.answer[:400]}...")
    
    # Summary
    print("\n" + "="*80)
    print("📈 SUMMARY")
    print("="*80)
    
    raw_results = [r for r in results if r.approach == "raw"]
    helix_results = [r for r in results if r.approach == "helix"]
    
    avg_raw_time = sum(r.response_time for r in raw_results) / len(raw_results)
    avg_helix_time = sum(r.response_time for r in helix_results) / len(helix_results)
    
    avg_raw_ctx = sum(r.context_length for r in raw_results) / len(raw_results)
    avg_helix_ctx = sum(r.context_length for r in helix_results) / len(helix_results)
    
    print(f"\nAverage Response Time: RAW={avg_raw_time:.1f}s, HELIX={avg_helix_time:.1f}s")
    print(f"Average Context Size:  RAW={avg_raw_ctx:.0f}, HELIX={avg_helix_ctx:.0f} chars")
    print(f"\nHelix provides {avg_helix_ctx/avg_raw_ctx:.1f}x more structured context")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helix Vector Benchmark")
    parser.add_argument("--model", default="gemma3:1b", help="Ollama model")
    parser.add_argument("--questions", type=int, default=3, help="Number of questions")
    args = parser.parse_args()
    
    print("="*80)
    print("🧪 HELIX VECTOR BENCHMARK")
    print("   Comparing: Raw Logs vs. Helix Vector (DNA + Anomaly Detection)")
    print("="*80)
    
    # Sample logs with clear anomaly pattern
    logs = [
        "2024-01-15 10:00:01 INFO User john connected from 192.168.1.1",
        "2024-01-15 10:00:02 INFO User john requested dashboard",
        "2024-01-15 10:00:03 INFO Query completed in 45ms",
        "2024-01-15 10:00:04 INFO User john logged out",
        "2024-01-15 10:01:01 INFO User alice connected from 10.0.0.1",
        "2024-01-15 10:01:02 INFO User alice requested analytics",
        "2024-01-15 10:01:03 ERROR Database connection timeout after 30s",
        "2024-01-15 10:01:04 CRITICAL Service crashed - memory corruption at 0x7fff5fb",
        "2024-01-15 10:01:05 INFO Service recovered after restart",
        "2024-01-15 10:02:01 INFO User bob connected from 172.16.0.1",
        "2024-01-15 10:02:02 INFO User bob requested dashboard",
        "2024-01-15 10:02:03 INFO Query completed in 32ms",
        "2024-01-15 10:02:04 INFO User bob logged out",
    ]
    
    questions = [
        "Why did the service crash?",
        "What errors occurred and what caused them?",
        "How many users were affected by the database timeout?",
    ][:args.questions]
    
    # Connect to Ollama
    print(f"\n🔌 Connecting to Ollama (model: {args.model})...")
    try:
        llm = OllamaClient(model=args.model, timeout=180)
    except ConnectionError as e:
        print(f"❌ {e}")
        print("\nMake sure Ollama is running: ollama serve")
        exit(1)
    
    # Run benchmark
    print(f"\n📋 Running benchmark with {len(questions)} questions...")
    results = run_benchmark(logs, questions, llm)
    
    # Print comparison
    print_comparison(results)
    
    print("\n" + "="*80)
    print("✅ BENCHMARK COMPLETE!")
    print("="*80)
    print("\n💡 Manual evaluation: Review the answers above.")
    print("   Which approach identified the root cause more accurately?")
    print("   Which gave more actionable insights?")
