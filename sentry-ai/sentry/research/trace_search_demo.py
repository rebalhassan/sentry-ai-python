"""
Trace Search Demo - Research Module
====================================

Demonstrates the full pipeline:
1. Encode logs into DNA strings (Drain3)
2. Convert traces to embeddings (Sentence Transformer)
3. Store in vector database with metadata
4. Search for similar error traces

This is what we want to achieve:
- "Show me errors similar to this database timeout"
- "What other issues happened on this IP?"
- "Find traces with similar patterns to this failure"

Run:
    python trace_search_demo.py
"""

from log_dna import LogDNAEncoder, EMBEDDINGS_AVAILABLE
from trace_vectorstore import TraceVectorStore
import numpy as np
from datetime import datetime


def main():
    print("=" * 70)
    print("🔍 TRACE SEARCH DEMO - Finding Similar Error Patterns")
    print("=" * 70)
    
    if not EMBEDDINGS_AVAILABLE:
        print("❌ sentence-transformers not installed")
        print("   Run: pip install sentence-transformers")
        return
    
    # =========================================================
    # STEP 1: Create sample log data (simulating real logs)
    # =========================================================
    print("\n📋 STEP 1: Creating sample log traces...")
    
    # These simulate different user sessions with various outcomes
    sample_sessions = [
        {
            "session_id": "session_001",
            "source": "datadog",
            "logs": [
                "2024-01-15 10:00:01 INFO [PID:1234] User john connected from 192.168.1.100",
                "2024-01-15 10:00:02 INFO [PID:1234] User john requested dashboard",
                "2024-01-15 10:00:03 INFO [PID:1234] Query completed in 45ms",
                "2024-01-15 10:00:04 INFO [PID:1234] User john logged out",
            ]
        },
        {
            "session_id": "session_002",
            "source": "datadog",
            "logs": [
                "2024-01-15 10:05:01 INFO [PID:2345] User alice connected from 10.0.0.50",
                "2024-01-15 10:05:02 INFO [PID:2345] User alice requested dashboard",
                "2024-01-15 10:05:03 ERROR [PID:2345] Database connection timeout after 30s",
                "2024-01-15 10:05:04 ERROR [PID:2345] Session terminated unexpectedly",
            ]
        },
        {
            "session_id": "session_003",
            "source": "vercel",
            "logs": [
                "2024-01-15 10:10:01 INFO [PID:3456] User bob connected from 172.16.0.1",
                "2024-01-15 10:10:02 INFO [PID:3456] User bob requested dashboard",
                "2024-01-15 10:10:03 INFO [PID:3456] Query completed in 32ms",
                "2024-01-15 10:10:04 INFO [PID:3456] User bob logged out",
            ]
        },
        {
            "session_id": "session_004",
            "source": "datadog",
            "logs": [
                "2024-01-15 10:15:01 INFO [PID:4567] User charlie connected from 192.168.1.200",
                "2024-01-15 10:15:02 INFO [PID:4567] User charlie requested reports",
                "2024-01-15 10:15:03 ERROR [PID:4567] Database connection refused at 0x7fff5fbff000",
                "2024-01-15 10:15:04 ERROR [PID:4567] Service unavailable",
            ]
        },
        {
            "session_id": "session_005",
            "source": "vercel",
            "logs": [
                "2024-01-15 10:20:01 INFO [PID:5678] User diana connected from 10.0.0.100",
                "2024-01-15 10:20:02 INFO [PID:5678] User diana requested settings",
                "2024-01-15 10:20:03 INFO [PID:5678] Settings updated successfully",
                "2024-01-15 10:20:04 INFO [PID:5678] User diana logged out",
            ]
        },
        {
            "session_id": "session_006",
            "source": "datadog",
            "logs": [
                "2024-01-15 10:25:01 INFO [PID:6789] User eve connected from 192.168.1.100",
                "2024-01-15 10:25:02 INFO [PID:6789] User eve requested dashboard",
                "2024-01-15 10:25:03 ERROR [PID:6789] Database timeout: query exceeded 30s limit",
                "2024-01-15 10:25:04 ERROR [PID:6789] Request failed with 503",
            ]
        },
    ]
    
    print(f"   Created {len(sample_sessions)} sample sessions (3 normal, 3 with errors)")
    
    # =========================================================
    # STEP 2: Encode all logs into DNA sequences
    # =========================================================
    print("\n🧬 STEP 2: Encoding logs into DNA sequences with Drain3...")
    
    encoder = LogDNAEncoder()
    
    # Process all logs through Drain3 to build the cluster templates
    all_logs = []
    for session in sample_sessions:
        all_logs.extend(session["logs"])
    
    # First pass: encode all logs to build templates
    all_cluster_ids = encoder.encode_logs(all_logs)
    
    # Show the codebook
    print(f"\n{encoder.get_summary()}")
    
    # =========================================================
    # STEP 3: Create embeddings and store in vector DB
    # =========================================================
    print("\n💾 STEP 3: Creating embeddings and storing in vector DB...")
    
    # Initialize encoder's embedding model
    encoder.init_embeddings()
    
    # Create vector store
    store = TraceVectorStore(
        dimension=384,  # MiniLM dimension
        index_path=None  # In-memory for demo (set path to persist)
    )
    
    # Process each session
    log_index = 0
    for session in sample_sessions:
        session_id = session["session_id"]
        logs = session["logs"]
        
        # Get cluster IDs for this session's logs
        session_cluster_ids = all_cluster_ids[log_index:log_index + len(logs)]
        log_index += len(logs)
        
        # Get trace embedding
        trace_embedding = encoder.get_trace_vector(session_cluster_ids)
        
        # Create metadata
        metadata = {
            "cluster_ids": session_cluster_ids,
            "dna_string": encoder.to_dna_string(session_cluster_ids),
            "raw_logs": logs,
            "source": session["source"],
            "timestamp": logs[0].split(" ")[0] + " " + logs[0].split(" ")[1],
            # PIDs, IPs, memory_addresses will be auto-extracted by the store
        }
        
        # Add to store
        store.add_trace(session_id, trace_embedding, metadata)
        
        # Show what was stored
        has_error = store.get_trace(session_id)["has_error"]
        pids = store.get_trace(session_id)["pids"]
        ips = store.get_trace(session_id)["ips"]
        print(f"   ✅ {session_id}: DNA={metadata['dna_string']}, "
              f"Error={has_error}, PIDs={pids}, IPs={ips}")
    
    # Show store stats
    print(f"\n📊 Store stats: {store}")
    
    # =========================================================
    # STEP 4: Search for similar traces
    # =========================================================
    print("\n" + "=" * 70)
    print("🔍 STEP 4: Searching for similar error traces")
    print("=" * 70)
    
    # Take an error trace as query
    query_session = sample_sessions[1]  # session_002 (database timeout)
    query_logs = query_session["logs"]
    query_cluster_ids = encoder.encode_logs(query_logs)
    query_embedding = encoder.get_trace_vector(query_cluster_ids)
    
    print(f"\n📝 Query: session_002 (Database timeout error)")
    print(f"   DNA: {encoder.to_dna_string(query_cluster_ids)}")
    
    # Search all traces
    print("\n🔎 All similar traces:")
    results = store.search(query_embedding, k=5)
    for trace_id, score, metadata in results:
        error_str = "❌ ERROR" if metadata.get("has_error") else "✅ OK"
        print(f"   {trace_id}: {score:.3f} similarity - {error_str}")
        print(f"      DNA: {metadata.get('dna_string')}")
    
    # Search only error traces
    print("\n🔎 Similar ERROR traces only:")
    error_results = store.search_error_traces(query_embedding, k=3)
    for trace_id, score, metadata in error_results:
        print(f"   {trace_id}: {score:.3f} similarity")
        print(f"      DNA: {metadata.get('dna_string')}")
        print(f"      IPs: {metadata.get('ips')}")
        print(f"      PIDs: {metadata.get('pids')}")
    
    # =========================================================
    # STEP 5: Search by IP (find all issues on a server)
    # =========================================================
    print("\n" + "=" * 70)
    print("🔍 STEP 5: Finding traces by IP address")
    print("=" * 70)
    
    target_ip = "192.168.1.100"
    print(f"\n📡 Searching for traces involving IP: {target_ip}")
    
    ip_results = store.search_by_ip(query_embedding, target_ip, k=5)
    for trace_id, score, metadata in ip_results:
        error_str = "❌ ERROR" if metadata.get("has_error") else "✅ OK"
        print(f"   {trace_id}: {score:.3f} - {error_str}")
        print(f"      PIDs: {metadata.get('pids')}, Source: {metadata.get('source')}")
    
    # =========================================================
    # DONE!
    # =========================================================
    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE!")
    print("=" * 70)
    print("\n💡 What we demonstrated:")
    print("   1. Encoded raw logs into DNA sequences using Drain3")
    print("   2. Created embeddings capturing the 'meaning' of each trace")
    print("   3. Stored traces with metadata (PIDs, IPs, etc.) in vector DB")
    print("   4. Searched for similar error patterns")
    print("   5. Filtered by IP to find related issues")
    print("\n🎯 Use cases:")
    print("   - 'Find errors similar to this database timeout'")
    print("   - 'What other issues happened on server 192.168.1.100?'")
    print("   - 'Show me traces with similar patterns to this failure'")


if __name__ == "__main__":
    main()
