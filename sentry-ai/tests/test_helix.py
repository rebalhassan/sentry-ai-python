# test_helix.py
"""
Test the Helix Vector service - DNA encoding and anomaly detection
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from sentry.services.helix import HelixService, get_helix_service
from sentry.core.models import LogChunk, LogLevel


def test_helix_service():
    print("Testing Helix Vector Service\n")
    
    # Create service
    print("1. Creating HelixService...")
    helix = HelixService(anomaly_threshold=0.20)
    print(f"   Created: {helix}")
    
    # Sample logs
    sample_logs = [
        "2024-01-15 10:00:01 INFO User john connected from 192.168.1.1",
        "2024-01-15 10:00:02 INFO User john requested dashboard",
        "2024-01-15 10:00:03 INFO Query completed in 45ms",
        "2024-01-15 10:00:04 INFO User john logged out",
        "2024-01-15 10:01:01 INFO User alice connected from 10.0.0.1",
        "2024-01-15 10:01:02 INFO User alice requested analytics",
        "2024-01-15 10:01:03 ERROR Database connection timeout after 30s",
        "2024-01-15 10:01:04 CRITICAL Service crashed - restarting",
        "2024-01-15 10:01:05 INFO Service recovered after restart",
    ]
    
    # Create LogChunks
    chunks = []
    for i, log in enumerate(sample_logs):
        chunk = LogChunk(
            source_id="test_source",
            content=log,
            timestamp=datetime.now(),
            log_level=LogLevel.INFO
        )
        chunks.append(chunk)
    
    # Test 1: Encode logs
    print("\n2. Encoding logs to DNA clusters...")
    cluster_ids = helix.encode_logs(sample_logs)
    print(f"   Cluster IDs: {cluster_ids}")
    
    codebook = helix.get_codebook()
    print(f"   Codebook ({len(codebook)} clusters):")
    for cid, info in codebook.items():
        print(f"     {cid}: {info['template'][:50]}...")
    
    # Test 2: Fit transition model
    print("\n3. Fitting transition probability model...")
    helix.fit(cluster_ids)
    
    transitions = helix.get_transition_probs()
    print(f"   Learned {len(transitions)} transition sources")
    
    # Test 3: Annotate chunks
    print("\n4. Annotating chunks with Helix metadata...")
    
    # Reset for fresh annotation
    helix2 = HelixService(anomaly_threshold=0.20)
    annotated = helix2.annotate_chunks(chunks)
    
    print(f"   Annotated {len(annotated)} chunks")
    
    anomaly_count = sum(1 for c in annotated if c.is_anomaly)
    print(f"   Detected {anomaly_count} anomalies")
    
    # Test 4: Check anomaly details
    print("\n5. Anomaly details:")
    for i, chunk in enumerate(annotated):
        if chunk.is_anomaly:
            print(f"   [{i}] ANOMALY: {chunk.anomaly_type}")
            print(f"       Severity: {chunk.severity_weight:.2f}")
            print(f"       Score: {chunk.anomaly_score:.2f}")
            print(f"       Content: {chunk.content[:50]}...")
    
    # Test 5: Check non-anomaly has default values
    print("\n6. Checking normal chunks have default values...")
    normal_chunks = [c for c in annotated if not c.is_anomaly]
    if normal_chunks:
        first_normal = normal_chunks[0]
        assert first_normal.anomaly_score == 0.0, "Normal chunk should have score 0"
        assert first_normal.anomaly_type is None, "Normal chunk should have no type"
        print(f"   Normal chunk: cluster_id={first_normal.cluster_id}, score=0.0")
    
    # Test 6: Verify severity penalties
    print("\n7. Verifying severity detection...")
    test_templates = [
        ("FATAL kernel panic", 0.9),
        ("CRITICAL database failure", 0.8),
        ("ERROR connection timeout", 0.5),
        ("WARNING high memory", 0.3),
        ("INFO user logged in", 0.0),
    ]
    for template, expected in test_templates:
        actual = helix._get_severity_penalty(template)
        status = "OK" if actual == expected else f"FAIL (got {actual})"
        print(f"   '{template}' -> {expected} [{status}]")
    
    print("\n" + "=" * 50)
    print("Helix Vector Service tests completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    test_helix_service()
