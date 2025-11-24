import streamlit as st
import os
import sys
from datetime import datetime
import time

# Add parent directory to path so we can import sentry modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentry.services.llm import get_llm_client
from sentry.services.eventviewer import EventViewerReader
from sentry.services.rag import get_rag_service
from sentry.core.models import LogChunk, LogLevel

# Page config
st.set_page_config(
    page_title="Sentry-AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize services
@st.cache_resource
def get_services():
    from sentry.core.config import settings
    return {
        "llm": get_llm_client(),
        "rag": get_rag_service(),
        "reader": EventViewerReader()
    }

# Clear cache button in sidebar (placed early so user can see it)
if st.sidebar.button("🔄 Reload Services", help="Clear cache and reload services with latest config"):
    st.cache_resource.clear()
    st.rerun()

services = get_services()

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "logs_indexed" not in st.session_state:
    st.session_state.logs_indexed = False

if "log_count" not in st.session_state:
    st.session_state.log_count = 0

if "chunks_to_index" not in st.session_state:
    st.session_state.chunks_to_index = []

def main():
    st.title("🛡️ Sentry-AI")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Source Selection
        source_type = st.radio(
            "Select Log Source",
            ["File", "Event Viewer"],
            index=1
        )
        
        if source_type == "File":
            uploaded_file = st.file_uploader("Upload Log File", type=["log", "txt"])
            if uploaded_file:
                # Save to temp file to read
                content = uploaded_file.getvalue().decode("utf-8")
                lines = content.splitlines()
                st.info(f"Loaded {len(lines)} lines from file")
                
                # Convert to chunks (simple line-based for now)
                st.session_state.chunks_to_index = []
                for i, line in enumerate(lines):
                    if line.strip():
                        chunk = LogChunk(
                            source_id=f"file_{uploaded_file.name}",
                            content=line,
                            timestamp=datetime.now(),
                            log_level=LogLevel.INFO,
                            metadata={"file": uploaded_file.name, "line": i+1}
                        )
                        st.session_state.chunks_to_index.append(chunk)

        else:  # Event Viewer
            log_name = st.selectbox(
                "Select Event Log",
                ["System", "Application", "Security"]
            )
            
            limit = st.number_input("Max Events", min_value=10, max_value=1000, value=100)
            
            if st.button("Fetch Events"):
                with st.spinner(f"Reading {log_name} logs..."):
                    try:
                        chunks = services["reader"].read_events(
                            log_name=log_name,
                            source_id=f"evt_{log_name}",
                            max_events=limit
                        )
                        st.session_state.chunks_to_index = chunks
                        st.success(f"Fetched {len(chunks)} events - ready to index!")
                    except Exception as e:
                        st.error(f"Error reading logs: {e}")

        # Indexing Action
        if st.session_state.chunks_to_index:
            if st.button("Index Logs for RAG", type="primary"):
                with st.spinner("Indexing logs..."):
                    count = services["rag"].index_chunks_batch(st.session_state.chunks_to_index)
                    st.session_state.logs_indexed = True
                    st.session_state.log_count += count
                    st.session_state.chunks_to_index = []  # Clear after indexing
                    st.success(f"Indexed {count} chunks! RAG is ready.")
        
        st.divider()
        
        # Display Current Model
        st.info(f"Using Model: **{services['llm'].model}**")
            
        # Global Context
        st.session_state.global_context = st.text_area(
            "Log Context",
            placeholder="e.g. These are logs from a production web server...",
            help="This context is added to the system prompt."
        )
        
        st.divider()
        
        # Stats
        if st.session_state.logs_indexed:
            st.metric("Indexed Chunks", st.session_state.log_count)
            if st.button("Clear Index"):
                services["rag"].vector_store.clear()
                st.session_state.logs_indexed = False
                st.session_state.log_count = 0
                st.rerun()

    # Main Chat Interface
    
    # RAG Toggle
    use_rag = st.toggle("Enable RAG (Search Logs)", value=True, help="If enabled, AI will search indexed logs for answers.")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if prompt := st.chat_input("Ask about your logs..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Thinking..."):
                try:
                    # System prompt with context
                    system_prompt = "You are Sentry-AI, a helpful log analysis assistant."
                    if "global_context" in st.session_state and st.session_state.global_context:
                        system_prompt += f"\n\nContext provided by user:\n{st.session_state.global_context}"
                    
                    if use_rag and st.session_state.logs_indexed:
                        # RAG Mode
                        result = services["rag"].query(prompt)
                        full_response = result.answer
                        
                        # Append sources if available
                        if result.sources:
                            full_response += "\n\n**Sources:**\n"
                            for i, chunk in enumerate(result.sources[:3]):
                                source_name = chunk.metadata.get('file') or chunk.metadata.get('log_name') or 'Unknown'
                                full_response += f"- [{source_name}] {chunk.content[:100]}...\n"
                                
                    else:
                        # Standard Mode (Direct LLM)
                        # We need to format history for the LLM
                        # For simplicity in this v1, we'll just send the last prompt + system prompt
                        # Ideally we should send full history
                        full_response = services["llm"].generate(
                            prompt,
                            system_prompt=system_prompt
                        )
                    
                    message_placeholder.markdown(full_response)
                    
                    # Add assistant message to history
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    
                except Exception as e:
                    st.error(f"Error generating response: {e}")

if __name__ == "__main__":
    main()
