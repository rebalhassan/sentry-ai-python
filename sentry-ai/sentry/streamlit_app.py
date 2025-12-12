import streamlit as st
import os
import sys
from datetime import datetime
import time
import asyncio

# Add parent directory to path so we can import sentry modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentry.services.llm import get_llm_client
from sentry.services.eventviewer import EventViewerReader
from sentry.services.rag import get_rag_service
from sentry.core.models import LogChunk, LogLevel
from sentry.core.credentials import get_credential_manager
from sentry.integrations import VercelIntegration, PostHogIntegration, DataDogIntegration

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
        "reader": EventViewerReader(),
        "credentials": get_credential_manager()
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


def render_integration_ui(service_name: str, service_class, credential_schema: dict):
    """Render UI for a specific integration service"""
    st.subheader(f"{service_name.title()}")
    
    # Check if credentials exist
    creds = services["credentials"].get_credentials(service_name)
    has_creds = creds is not None
    
    # Credentials section
    with st.expander("🔑 Credentials", expanded=not has_creds):
        if has_creds:
            st.success(f"✅ Credentials configured")
            if st.button(f"Delete {service_name.title()} Credentials", key=f"delete_{service_name}"):
                services["credentials"].delete_credentials(service_name)
                st.rerun()
        else:
            st.warning(f"No credentials configured for {service_name.title()}")
        
        # Credential input form
        with st.form(f"{service_name}_creds_form"):
            st.write(f"**Configure {service_name.title()} Credentials**")
            
            cred_values = {}
            for field, info in credential_schema.items():
                if info.get("required"):
                    cred_values[field] = st.text_input(
                        info["label"],
                        type="password" if "key" in field.lower() else "default",
                        help=info.get("help", "")
                    )
                else:
                    cred_values[field] = st.text_input(
                        info["label"],
                        help=info.get("help", "")
                    )
            
            if st.form_submit_button("Save Credentials"):
                try:
                    # Filter out empty optional fields
                    filtered_creds = {k: v for k, v in cred_values.items() if v}
                    services["credentials"].store_credentials(service_name, filtered_creds)
                    st.success(f"✅ {service_name.title()} credentials saved!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving credentials: {e}")
    
    # Only show fetch UI if credentials are configured
    if not has_creds:
        st.info(f"Configure credentials above to fetch logs from {service_name.title()}")
        return
    
    # Fetch logs section
    with st.expander("📥 Fetch Logs", expanded=True):
        try:
            # Create integration instance
            integration = service_class(creds)
            
            # Async wrapper for sync Streamlit
            def run_async(coro):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(coro)
                finally:
                    loop.close()
            
            # List projects
            if st.button(f"List {service_name.title()} Projects", key=f"list_projects_{service_name}"):
                with st.spinner("Fetching projects..."):
                    try:
                        projects = run_async(integration.list_projects())
                        st.session_state[f"{service_name}_projects"] = projects
                        st.success(f"Found {len(projects)} projects")
                    except Exception as e:
                        st.error(f"Error fetching projects: {e}")
            
            # Show projects if available
            if f"{service_name}_projects" in st.session_state:
                projects = st.session_state[f"{service_name}_projects"]
                
                if projects:
                    project_options = {f"{p.name} ({p.id})": p.id for p in projects}
                    selected_project = st.selectbox(
                        "Select Project",
                        options=list(project_options.keys()),
                        key=f"project_select_{service_name}"
                    )
                    
                    if selected_project:
                        project_id = project_options[selected_project]
                        
                        # List deployments
                        if st.button(f"List Deployments", key=f"list_deployments_{service_name}"):
                            with st.spinner("Fetching deployments..."):
                                try:
                                    deployments = run_async(integration.list_deployments(project_id))
                                    st.session_state[f"{service_name}_deployments"] = deployments
                                    st.success(f"Found {len(deployments)} deployments")
                                except Exception as e:
                                    st.error(f"Error fetching deployments: {e}")
                        
                        # Show deployments if available
                        if f"{service_name}_deployments" in st.session_state:
                            deployments = st.session_state[f"{service_name}_deployments"]
                            
                            if deployments:
                                deployment_options = {f"{d.name} - {d.state}": d.id for d in deployments}
                                selected_deployment = st.selectbox(
                                    "Select Deployment/Time Period",
                                    options=list(deployment_options.keys()),
                                    key=f"deployment_select_{service_name}"
                                )
                                
                                if selected_deployment:
                                    deployment_id = deployment_options[selected_deployment]
                                    
                                    # Fetch logs
                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        log_limit = st.number_input(
                                            "Max Logs",
                                            min_value=10,
                                            max_value=1000,
                                            value=100,
                                            key=f"log_limit_{service_name}"
                                        )
                                    
                                    if st.button(f"Fetch Logs", type="primary", key=f"fetch_logs_{service_name}"):
                                        with st.spinner(f"Fetching logs from {service_name.title()}..."):
                                            try:
                                                logs = run_async(integration.fetch_logs(
                                                    project_id=project_id,
                                                    deployment_id=deployment_id,
                                                    limit=log_limit
                                                ))
                                                
                                                if logs:
                                                    # Convert to LogChunks
                                                    chunks = []
                                                    for log in logs:
                                                        # Map log levels
                                                        level_map = {
                                                            "debug": LogLevel.DEBUG,
                                                            "info": LogLevel.INFO,
                                                            "warning": LogLevel.WARNING,
                                                            "error": LogLevel.ERROR,
                                                            "critical": LogLevel.ERROR,
                                                        }
                                                        
                                                        chunk = LogChunk(
                                                            id=log.id,
                                                            source_id=f"{service_name}_{deployment_id}",
                                                            content=log.message,
                                                            timestamp=log.timestamp,
                                                            log_level=level_map.get(log.level.value, LogLevel.INFO),
                                                            metadata={
                                                                "integration": service_name,
                                                                "source": log.source,
                                                                **log.metadata
                                                            }
                                                        )
                                                        chunks.append(chunk)
                                                    
                                                    st.session_state.chunks_to_index = chunks
                                                    st.success(f"✅ Fetched {len(chunks)} logs - ready to index!")
                                                else:
                                                    st.warning("No logs found")
                                                    
                                            except Exception as e:
                                                st.error(f"Error fetching logs: {e}")
        
        except Exception as e:
            st.error(f"Error initializing {service_name.title()} integration: {e}")


def main():
    st.title("🛡️ Sentry-AI")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Source Selection with tabs
        tab1, tab2 = st.tabs(["📁 Local Sources", "🌐 External Sources"])
        
        with tab1:
            # Original local source selection
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
        
        with tab2:
            # External integrations
            st.write("**External Log Sources**")
            
            integration_service = st.selectbox(
                "Select Service",
                ["Vercel", "PostHog", "DataDog"],
                key="integration_service_select"
            )
            
            # Define credential schemas
            credential_schemas = {
                "vercel": {
                    "api_key": {"label": "API Token", "required": True, "help": "Vercel Access Token from dashboard"},
                    "team_id": {"label": "Team ID (optional)", "required": False, "help": "For team-scoped access"}
                },
                "posthog": {
                    "api_key": {"label": "Personal API Key", "required": True, "help": "From PostHog project settings"},
                    "project_id": {"label": "Project ID", "required": True, "help": "Your PostHog project ID"},
                    "region": {"label": "Region (us/eu)", "required": False, "help": "Default: us"}
                },
                "datadog": {
                    "api_key": {"label": "API Key", "required": True, "help": "DataDog API key"},
                    "app_key": {"label": "Application Key", "required": True, "help": "DataDog application key"},
                    "site": {"label": "Site (us1/us3/eu1)", "required": False, "help": "Default: us1"}
                }
            }
            
            integration_classes = {
                "vercel": VercelIntegration,
                "posthog": PostHogIntegration,
                "datadog": DataDogIntegration
            }
            
            service_key = integration_service.lower()
            render_integration_ui(
                service_key,
                integration_classes[service_key],
                credential_schemas[service_key]
            )

        # Indexing Action (common for all sources)
        st.divider()
        if st.session_state.chunks_to_index:
            st.info(f"📦 {len(st.session_state.chunks_to_index)} logs ready to index")
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
                        # RAG Mode
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
