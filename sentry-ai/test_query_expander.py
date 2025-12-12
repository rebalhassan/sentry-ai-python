"""
Test script for regex-based query expander
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sentry.services.llm import get_llm_client

def test_query_expansion():
    """Test the regex-based query expander with various queries"""
    
    print("=" * 80)
    print("REGEX-BASED QUERY EXPANDER TEST")
    print("=" * 80)
    print()
    
    # Get LLM client
    llm = get_llm_client()
    
    # Test queries
    test_queries = [
        "why is it slow?",
        "database errors",
        "checkout failing",
        "memory issues",
        "disk full",
        "authentication failed",
        "network timeout",
        "500 errors",
        "container crashed",
        "cache miss",
        "queue overflow",
        "deployment failed",
        "email not sending",
        "upload timeout",
        "EventID 7",  # Specific ID - should not expand much
        "show me logs",  # Generic - might not expand
    ]
    
    print(f"Testing {len(test_queries)} queries...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n[{i}] Query: \"{query}\"")
        print("-" * 80)
        
        # Expand the query
        expanded = llm.expand_query(query)
        
        # Count expansion terms
        if " OR " in expanded:
            terms = expanded.split(" OR ")
            expansion_count = len(terms) - 1  # Subtract original query
            print(f"✅ Expanded to {expansion_count} additional terms")
            print(f"Preview: {' OR '.join(terms[:6])}...")
            if len(terms) > 6:
                print(f"         ... and {len(terms) - 6} more terms")
        else:
            print(f"ℹ️  No expansion (returned as-is)")
        
        print()
    
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    test_query_expansion()
