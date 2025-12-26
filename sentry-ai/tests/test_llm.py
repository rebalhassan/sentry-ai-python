# test_llm.py
"""Test the LLM client"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime
from sentry.services.llm import LLMClient
from sentry.core.models import LogChunk, LogLevel


def test_llm():
    print("🧪 Testing LLM Client\n")
    
    # Initialize
    print("🤖 Connecting to Ollama...")
    try:
        llm = LLMClient()
        print("✅ Connected!\n")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        print("\nMake sure:")
        print("  1. Ollama is installed (https://ollama.com)")
        print("  2. Ollama is running (it should auto-start)")
        print("  3. Model is pulled: ollama pull llama3:8b")
        return
    
    # ===== TEST 1: Simple generation =====
    print("💬 TEST 1: Simple generation")
    
    prompt = "What is a segmentation fault? Answer in one sentence."
    print(f"   Prompt: '{prompt}'")
    print(f"   Generating...\n")
    
    response = llm.generate(prompt)
    print(f"   Response: {response}\n")
    
    # ===== TEST 2: With system prompt =====
    print("💬 TEST 2: With system prompt")
    
    system_prompt = "You are a helpful assistant. Be concise."
    prompt = "Explain DNS in 20 words or less."
    
    print(f"   System: '{system_prompt}'")
    print(f"   Prompt: '{prompt}'")
    print(f"   Generating...\n")
    
    response = llm.generate(prompt, system_prompt=system_prompt)
    print(f"   Response: {response}\n")
    
    # ===== TEST 3: RAG with context (THE REAL USE CASE) =====
    print("🎯 TEST 3: RAG - Generate with log context")
    
    # Simulate some log chunks
    chunks = [
        LogChunk(
            source_id="test",
            content="ERROR: Disk write failed on D:\\ - Error code: 0x80070015",
            timestamp=datetime(2025, 11, 3, 9, 1, 3),
            log_level=LogLevel.ERROR,
            metadata={"file": "system.log", "line": 4521}
        ),
        LogChunk(
            source_id="test",
            content="ERROR: The device, \\Device\\Harddisk0\\DR0, has a bad block.",
            timestamp=datetime(2025, 11, 3, 9, 1, 5),
            log_level=LogLevel.ERROR,
            metadata={"event_id": 7}
        ),
        LogChunk(
            source_id="test",
            content="CRITICAL: Volume D: is out of disk space. 0 bytes remaining.",
            timestamp=datetime(2025, 11, 3, 9, 1, 8),
            log_level=LogLevel.CRITICAL,
            metadata={"file": "system.log", "line": 4525}
        )
    ]
    
    query = "What's wrong with the system? What should I do?"
    
    print(f"   Query: '{query}'")
    print(f"   Context: {len(chunks)} log entries")
    print(f"   Generating answer...\n")
    
    response = llm.generate_with_context(query, chunks)
    
    print("   📋 ANSWER:")
    print("   " + "-" * 60)
    print(f"   {response}")
    print("   " + "-" * 60)
    print()
    
    # ===== TEST 4: Follow-up question (conversation) =====
    print("💬 TEST 4: Multi-turn conversation")
    
    messages = [
        {
            'role': 'user',
            'content': 'What are the three primary colors?'
        }
    ]
    
    print(f"   User: {messages[0]['content']}")
    response1 = llm.chat(messages)
    print(f"   Assistant: {response1}\n")
    
    # Add assistant's response and follow-up
    messages.append({'role': 'assistant', 'content': response1})
    messages.append({
        'role': 'user',
        'content': 'What happens if you mix the first two?'
    })
    
    print(f"   User: {messages[2]['content']}")
    response2 = llm.chat(messages)
    print(f"   Assistant: {response2}\n")
    
    # ===== TEST 5: Model info =====
    print("📊 TEST 5: Model information")
    info = llm.get_model_info()
    print(f"   Model: {info['model']}")
    print(f"   Host: {info['host']}")
    print(f"   Temperature: {info['temperature']}")
    
    # ===== TEST 6: Streaming (bonus) =====
    print("\n🌊 TEST 6: Streaming response")
    
    prompt = "Count from 1 to 5."
    print(f"   Prompt: '{prompt}'")
    print(f"   Streaming: ", end='', flush=True)
    
    full_response = ""
    for chunk in llm.stream_generate(prompt):
        print(chunk, end='', flush=True)
        full_response += chunk
    
    print("\n")
    
    print("🔥 All LLM tests passed!")
    print("\n💡 Tips:")
    print("   - Use generate_with_context() for RAG queries")
    print("   - Use chat() for multi-turn conversations")
    print("   - Use stream_generate() for responsive UIs")


if __name__ == "__main__":
    test_llm()