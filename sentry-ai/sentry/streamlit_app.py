import streamlit as st
import os
import sys
from datetime import datetime

# Add parent directory to path so we can import sentry modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentry.services.llm import get_llm_client
from sentry.services.rag import get_rag_service
from sentry.services.helix import get_helix_service
from sentry.core.models import LogChunk, LogLevel
from sentry.core.config import settings
from sentry.core.credentials import get_credential_manager

# Page config
st.set_page_config(
    page_title="Sentry-AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize services (cached for stability)
@st.cache_resource
def get_services():
    """
    Get services with caching.
    
    Services are cached for the lifetime of the app.
    Use the "Reload Services" button to manually refresh.
    """
    from sentry.core.config import settings
    
    return {
        "llm": get_llm_client(),
        "rag": get_rag_service(),
        "helix": get_helix_service() if settings.helix_enabled else None,
        "credentials": get_credential_manager()
    }

# Clear cache button in sidebar (placed early so user can see it)
if st.sidebar.button("🔄 Reload Services", help="Clear cache and reload services with latest config"):
    # Reset singletons only when explicitly requested
    import sentry.services.llm as llm_module
    import sentry.services.rag as rag_module
    import sentry.services.helix as helix_module
    llm_module._llm_client = None
    rag_module._rag_service = None
    helix_module._helix_service = None
    st.cache_resource.clear()
    st.rerun()

# Get cached services (stable across reruns)
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

# Track which file was indexed (survives across all reruns)
if "indexed_file" not in st.session_state:
    st.session_state.indexed_file = None





def main():
    st.title("🛡️ Sentry-AI")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # File Upload - Single log source option
        st.subheader("📁 Log Source")
        
        uploaded_file = st.file_uploader(
            "Upload Log File", 
            type=["log", "txt"],
            help="Upload a .log or .txt file to analyze"
        )
        
        if uploaded_file:
            current_file_name = uploaded_file.name
            
            # Check if this file was ALREADY INDEXED (most important check)
            already_indexed = (
                st.session_state.indexed_file == current_file_name and
                st.session_state.logs_indexed
            )
            
            # Check if file is ready to index (processed but not indexed yet)
            ready_to_index = (
                "processed_file" in st.session_state and 
                st.session_state.processed_file == current_file_name and
                st.session_state.chunks_to_index
            )
            
            if already_indexed:
                # File is already indexed - DO NOT reset or reprocess
                st.success(f"✅ File '{current_file_name}' already indexed")
            elif ready_to_index:
                # File processed, ready to index
                st.info(f"📦 File '{current_file_name}' ready ({len(st.session_state.chunks_to_index)} chunks)")
            else:
                # NEW file - process it
                content = uploaded_file.getvalue().decode("utf-8")
                lines = [l for l in content.splitlines() if l.strip()]
                st.info(f"📄 Loaded {len(lines)} log lines from file")
                
                # Use Helix to create windowed chunks (context stuffing)
                if settings.helix_enabled and services["helix"]:
                    # Reset Helix ONLY for truly new files
                    services["helix"].reset()
                    
                    # Create windowed chunks
                    st.session_state.chunks_to_index = services["helix"].create_windowed_chunks(
                        logs=lines,
                        source_id=f"file_{uploaded_file.name}",
                        metadata_base={"file": uploaded_file.name}
                    )
                    st.success(f"✅ Created {len(st.session_state.chunks_to_index)} windowed chunks")
                else:
                    # Fallback: simple line-based chunks
                    st.session_state.chunks_to_index = []
                    for i, line in enumerate(lines):
                        chunk = LogChunk(
                            source_id=f"file_{uploaded_file.name}",
                            content=line,
                            timestamp=datetime.now(),
                            log_level=LogLevel.INFO,
                            metadata={"file": uploaded_file.name, "line": i+1}
                        )
                        st.session_state.chunks_to_index.append(chunk)
                    st.success(f"✅ Created {len(st.session_state.chunks_to_index)} chunks")
                
                # Mark as processed (but NOT indexed yet)
                st.session_state.processed_file = current_file_name


        # Indexing Action (common for all sources)
        st.divider()
        if st.session_state.chunks_to_index:
            st.info(f"📦 {len(st.session_state.chunks_to_index)} logs ready to index")
            if st.button("Index Logs for RAG", type="primary"):
                with st.spinner("Indexing logs..."):
                    count = services["rag"].index_chunks_batch(st.session_state.chunks_to_index)
                    st.session_state.logs_indexed = True
                    st.session_state.log_count += count
                    # Mark this file as indexed (prevents reset on future reruns)
                    st.session_state.indexed_file = st.session_state.processed_file
                    st.session_state.chunks_to_index = []  # Clear after indexing
                    st.success(f"Indexed {count} chunks! RAG is ready.")
        
        st.divider()
        
        # LLM Backend Selection
        cloud_available = services["llm"].is_cloud_available()
        
        if cloud_available:
            use_cloud = st.toggle(
                "☁️ Use Cloud LLM (OpenRouter)",
                value=services["llm"].use_cloud,
                help="Toggle between local Ollama and OpenRouter cloud LLMs"
            )
            # Sync toggle state to service
            if use_cloud != services["llm"].use_cloud:
                services["llm"].set_use_cloud(use_cloud)
            
            if use_cloud:
                st.info(f"☁️ Cloud Model: **{services['llm'].cloud_model}**")
            else:
                st.info(f"🖥️ Local Model: **{services['llm'].model}**")
        else:
            st.info(f"🖥️ Local Model: **{services['llm'].model}**")
            st.caption("💡 Set `SENTRY_OPENROUTER_API_KEY` to enable cloud LLMs")
            
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
            
            # Helix stats if enabled
            if settings.helix_enabled and services["helix"]:
                helix_stats = services["helix"].get_stats()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Clusters", helix_stats["cluster_count"])
                with col2:
                    st.metric("Threshold", f"{helix_stats['anomaly_threshold']:.2f}")
            
            if st.button("Clear Index"):
                services["rag"].vector_store.clear()
                st.session_state.logs_indexed = False
                st.session_state.log_count = 0
                st.rerun()
        
        # Clear Chat History
        st.divider()
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat History", help="Remove all chat messages"):
                st.session_state.messages = []
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
            
            with st.spinner("Processing..."):
                try:
                    # System prompt with context
                    system_prompt = '''
                    <Role>
                    You are a cybersecurity expert with deep knowledge of NIST standards, frameworks, and best practices. 
                    </Role>
                    <Task>
                    You provide accurate, detailed guidance on cybersecurity controls, risk management, cloud security, and compliance based on NIST publications including the 800 series, FIPS, and related documents. 
                    </Task>
                    What information does Security Content Automation Protocol (SCAP) Version 1.2 Validation Program Test Requirements provide? (Section 3) ; SCAP validated modules; SCAP validation The authors, Melanie Cook, Stephen Quinn, and David Waltermire of the National Institute of Standards and Technology (NIST), and Dragos Prisaca of G2, Inc. would like to thank the many people who reviewed and contributed to this document, in particular, John Banghart of Microsoft who was the original author and pioneered the first SCAP Validation Program. The authors thank Matt Kerr, and Danny Haynes of the MITRE Corporation for their insightful technical contribution to the design of the SCAP 1.2 Validation Program and creation of original SCAP 1.2 validation test content. We also thank our document reviewers, Kelley Dempsey of NIST and Jeffrey Blank of the National Security Agency for their input. This publication is intended for NVLAP accredited laboratories conducting SCAP product and module testing for the program, vendors interested in receiving SCAP validation for their products or modules, and organizations deploying SCAP products in their environments. Accredited laboratories use the information in this report to guide their testing and ensure all necessary requirements are met by a product before recommending to NIST that the product be awarded the requested validation. Vendors may use the information in this report to understand the features that products and modules need in order to be eligible for an SCAP validation. Government agencies and integrators use the information to gain insight into the criteria required for SCAP validated products. The secondary audience for this publication includes end users, who can review the test requirements in order to understand the capabilities of SCAP validated products and gain knowledge about SCAP validation. OVAL and CVE are registered trademarks, and CCE, CPE, and OCIL are trademarks of The MITRE Corporation. Red Hat is a registered trademark of Red Hat, Inc. Windows operating system is registered trademark of Microsoft Corporation.
                    '''
                    if "global_context" in st.session_state and st.session_state.global_context:
                        system_prompt += f"\n\nContext provided by user:\n{st.session_state.global_context}"
                    
                    if use_rag and st.session_state.logs_indexed:
                        # RAG Mode - use vector search with LLM
                        result = services["rag"].query(prompt)
                        full_response = result.answer
                        
                        # Note: Sources are disabled in API but we can still show them in UI if desired
                        # Commenting out for consistency with API behavior
                        # if result.sources:
                        #     full_response += "\n\n**Sources:**\n"
                        #     for i, chunk in enumerate(result.sources[:3]):
                        #         source_name = chunk.metadata.get('file') or chunk.metadata.get('log_name') or 'Unknown'
                        #         full_response += f"- [{source_name}] {chunk.content[:100]}...\n"
                                
                    else:
                        # Standard Mode (Direct LLM)
                        # Check if cloud mode is enabled
                        if services["llm"].use_cloud and services["llm"].is_cloud_available():
                            full_response = services["llm"].query_cloud(
                                prompt,
                                system_prompt=system_prompt
                            )
                        else:
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
