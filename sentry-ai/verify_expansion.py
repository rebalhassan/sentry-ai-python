import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'sentry-ai')))

from sentry.services.llm import get_llm_client
from sentry.core.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_expansion():
    print("=== Verifying Query Expander ===")
    
    # Initialize LLM client
    llm = get_llm_client()
    print(f"LLM Client initialized (Host: {llm.host})")
    
    # Test cases
    test_queries = [
        "why is it slow?",
        "checkout errors",
        "database connection failed",
        "EventID 4624"
    ]
    
    print(f"\nUsing model: smollm2:135m (hardcoded)")
    print(f"Temperature: {settings.query_expansion_temperature}")
    print("-" * 50)
    
    for query in test_queries:
        print(f"\nOriginal: '{query}'")
        try:
            expanded = llm.expand_query(query)
            print(f"Expanded: '{expanded}'")
            
            # Basic validation
            if query == expanded:
                print("Result: UNCHANGED (Fallback or empty)")
            else:
                print("Result: EXPANDED")
                
        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == "__main__":
    verify_expansion()
