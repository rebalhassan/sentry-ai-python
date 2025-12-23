# sentry/anomaly_dashboard.py
"""
Anomaly Detection Dashboard
===========================

A comprehensive Streamlit dashboard for log anomaly detection and analysis.

Features:
- Log file upload and folder watching
- Drain3 template mining and clustering
- Markov chain transition probability visualization
- Anomaly detection with severity-weighted scoring
- Dual vector search (FAISS semantic + Helix pattern matching)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentry.services.helix import HelixService, get_helix_service
from sentry.services.embedding import get_embedder
from sentry.services.vectorstore import get_vector_store
from sentry.services.indexer import LogIndexer
from sentry.core.models import LogChunk, LogLevel


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Anomaly Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Dark theme enhancements */
    .stMetric {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #2d3748;
    }
    
    .stMetric label {
        color: #a0aec0 !important;
    }
    
    /* Anomaly cards */
    .anomaly-card {
        background: linear-gradient(135deg, #2d1f1f 0%, #1a1a2e 100%);
        border-left: 4px solid #ff4b4b;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Template table styling */
    .template-table {
        font-family: 'Monaco', 'Consolas', monospace;
        font-size: 12px;
    }
    
    /* Search box styling */
    .search-header {
        background: linear-gradient(90deg, #0d1117 0%, #161b22 100%);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    
    /* Severity badges */
    .severity-critical { color: #ff4b4b; font-weight: bold; }
    .severity-high { color: #ff8c00; font-weight: bold; }
    .severity-medium { color: #ffd700; }
    .severity-low { color: #00d4ff; }
    
    /* Glassmorphism effect for cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
def init_session_state():
    """Initialize session state variables."""
    if "helix_service" not in st.session_state:
        st.session_state.helix_service = HelixService()
    
    if "logs_loaded" not in st.session_state:
        st.session_state.logs_loaded = False
    
    if "raw_logs" not in st.session_state:
        st.session_state.raw_logs = []
    
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    
    if "cluster_ids" not in st.session_state:
        st.session_state.cluster_ids = []
    
    if "indexed_for_search" not in st.session_state:
        st.session_state.indexed_for_search = False


init_session_state()


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def parse_log_file(file_content: str) -> List[str]:
    """Parse uploaded log file content into individual log lines."""
    lines = file_content.strip().split('\n')
    # Filter out empty lines
    return [line.strip() for line in lines if line.strip()]


def get_severity_color(severity: float) -> str:
    """Get color based on severity weight."""
    if severity >= 0.8:
        return "#ff4b4b"  # Critical - Red
    elif severity >= 0.5:
        return "#ff8c00"  # High - Orange
    elif severity >= 0.3:
        return "#ffd700"  # Medium - Yellow
    else:
        return "#00d4ff"  # Low - Cyan


def get_severity_label(severity: float) -> str:
    """Get severity label from weight."""
    if severity >= 0.8:
        return "🔴 CRITICAL"
    elif severity >= 0.5:
        return "🟠 HIGH"
    elif severity >= 0.3:
        return "🟡 MEDIUM"
    else:
        return "🔵 LOW"


def create_transition_heatmap(transition_probs: Dict[int, Dict[int, float]], codebook: Dict) -> go.Figure:
    """Create a heatmap visualization of transition probabilities with improved readability."""
    if not transition_probs:
        fig = go.Figure()
        fig.add_annotation(
            text="No transition data available yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#888")
        )
        fig.update_layout(
            template="plotly_dark",
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    
    # Get cluster IDs sorted by frequency (most common first)
    cluster_counts = {cid: codebook.get(cid, {}).get("count", 0) for cid in 
                      set(transition_probs.keys()) | 
                      {to_c for probs in transition_probs.values() for to_c in probs.keys()}}
    all_clusters = sorted(cluster_counts.keys(), key=lambda x: cluster_counts.get(x, 0), reverse=True)
    
    # Limit to top 12 clusters for better readability
    all_clusters = all_clusters[:12]
    
    # Build matrix and prepare hover text
    matrix = []
    hover_texts = []
    
    for from_c in all_clusters:
        row = []
        hover_row = []
        from_template = codebook.get(from_c, {}).get("template", f"Cluster {from_c}")
        
        for to_c in all_clusters:
            prob = transition_probs.get(from_c, {}).get(to_c, 0)
            row.append(prob)
            to_template = codebook.get(to_c, {}).get("template", f"Cluster {to_c}")
            hover_row.append(
                f"<b>From [{from_c}]:</b><br>{from_template[:80]}<br><br>"
                f"<b>To [{to_c}]:</b><br>{to_template[:80]}<br><br>"
                f"<b>Probability:</b> {prob:.1%}"
            )
        matrix.append(row)
        hover_texts.append(hover_row)
    
    # Use just cluster IDs for axis labels
    axis_labels = [f"[{cid}]" for cid in all_clusters]
    
    # Create heatmap with custom hover
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=axis_labels,
        y=axis_labels,
        colorscale='Viridis',
        hoverongaps=False,
        hoverinfo='text',
        text=hover_texts,
        colorbar=dict(
            title=dict(text="Prob", side="right"),
            tickformat=".0%",
            thickness=15,
            len=0.8
        )
    ))
    
    fig.update_layout(
        title=dict(text="Transition Probability Matrix", font=dict(size=14, color="#fff")),
        template="plotly_dark",
        height=500,
        margin=dict(l=60, r=80, t=50, b=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title="To Cluster",
            tickfont=dict(size=11),
            tickangle=0,
            side="bottom"
        ),
        yaxis=dict(
            title="From Cluster",
            tickfont=dict(size=11),
            autorange="reversed"
        )
    )
    
    return fig


def create_cluster_chart(codebook: Dict) -> go.Figure:
    """Create a bar chart of cluster frequencies with improved readability."""
    if not codebook:
        fig = go.Figure()
        fig.add_annotation(
            text="No cluster data available yet",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#888")
        )
        fig.update_layout(
            template="plotly_dark",
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    
    # Prepare data with full templates for hover
    data = []
    for cluster_id, info in codebook.items():
        template = info.get("template", "")
        count = info.get("count", 0)
        
        # Determine color based on template content
        if any(kw in template.upper() for kw in ["FATAL", "CRITICAL"]):
            color = "#ff4b4b"
            severity = "CRITICAL"
        elif any(kw in template.upper() for kw in ["ERROR", "FAIL"]):
            color = "#ff8c00"
            severity = "ERROR"
        elif "WARN" in template.upper():
            color = "#ffd700"
            severity = "WARNING"
        else:
            color = "#00d4ff"
            severity = "INFO"
        
        data.append({
            "cluster_id": cluster_id,
            "template_short": f"[{cluster_id}]",
            "template_full": template,
            "count": count,
            "color": color,
            "severity": severity
        })
    
    # Sort by count and take top 10
    data.sort(key=lambda x: x["count"], reverse=True)
    data = data[:10]
    
    # Create custom hover text
    hover_texts = [
        f"<b>Cluster {d['cluster_id']}</b><br>"
        f"<b>Count:</b> {d['count']}<br>"
        f"<b>Severity:</b> {d['severity']}<br><br>"
        f"<b>Template:</b><br>{d['template_full'][:100]}{'...' if len(d['template_full']) > 100 else ''}"
        for d in data
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=[d["count"] for d in data],
            y=[d["template_short"] for d in data],
            orientation='h',
            marker=dict(
                color=[d["color"] for d in data],
                line=dict(color='rgba(255,255,255,0.3)', width=1)
            ),
            hoverinfo='text',
            hovertext=hover_texts,
            text=[d["count"] for d in data],
            textposition='outside',
            textfont=dict(size=10, color="#fff")
        )
    ])
    
    fig.update_layout(
        title=dict(text="Top Cluster Frequencies", font=dict(size=14, color="#fff")),
        template="plotly_dark",
        height=500,
        margin=dict(l=60, r=80, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            title="Cluster ID",
            tickfont=dict(size=12),
            categoryorder='total ascending'
        ),
        xaxis=dict(
            title="Occurrence Count",
            tickfont=dict(size=10),
            gridcolor='rgba(255,255,255,0.1)'
        ),
        bargap=0.3
    )
    
    return fig


def create_anomaly_distribution(chunks: List[LogChunk]) -> go.Figure:
    """Create a histogram of anomaly scores."""
    scores = [c.anomaly_score for c in chunks if c.is_anomaly]
    
    if not scores:
        fig = go.Figure()
        fig.add_annotation(
            text="No anomalies detected",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="#888")
        )
        fig.update_layout(
            template="plotly_dark",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    
    fig = px.histogram(
        x=scores,
        nbins=20,
        color_discrete_sequence=["#ff4b4b"]
    )
    
    fig.update_layout(
        title=dict(text="Anomaly Score Distribution", font=dict(size=14)),
        template="plotly_dark",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title="Anomaly Score"),
        yaxis=dict(title="Count"),
        showlegend=False
    )
    
    return fig


# ============================================================
# SIDEBAR - LOG INPUT
# ============================================================
def render_sidebar():
    """Render the sidebar for log input and settings."""
    st.sidebar.title("Log Input")
    
    # File Upload Section
    st.sidebar.subheader("Upload Log File")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a log file",
        type=["log", "txt", "json"],
        help="Upload a log file to analyze"
    )
    
    if uploaded_file is not None:
        if st.sidebar.button("Process File", type="primary", use_container_width=True):
            with st.spinner("Processing log file..."):
                content = uploaded_file.read().decode("utf-8")
                logs = parse_log_file(content)
                
                if logs:
                    # Reset helix service for fresh analysis
                    st.session_state.helix_service.reset()
                    
                    # Store raw logs
                    st.session_state.raw_logs = logs
                    
                    # Create chunks and annotate with Helix
                    chunks = st.session_state.helix_service.create_windowed_chunks(
                        logs=logs,
                        source_id="uploaded_file",
                        metadata_base={"filename": uploaded_file.name}
                    )
                    
                    # Annotate chunks with cluster IDs and anomalies
                    annotated_chunks = st.session_state.helix_service.annotate_chunks(chunks)
                    
                    st.session_state.chunks = annotated_chunks
                    st.session_state.logs_loaded = True
                    st.session_state.indexed_for_search = False
                    
                    st.sidebar.success(f"Loaded {len(logs)} log lines")
                    st.rerun()
                else:
                    st.sidebar.error("No valid log lines found in file")
    
    st.sidebar.divider()
    
    # Folder Path Input
    st.sidebar.subheader("Watch Folder")
    folder_path = st.sidebar.text_input(
        "Folder Path",
        placeholder="C:\\Logs\\MyApp",
        help="Enter path to a folder containing log files"
    )
    
    if folder_path:
        path = Path(folder_path)
        if path.exists() and path.is_dir():
            st.sidebar.success("Valid folder")
            
            # List log files in folder
            log_files = list(path.glob("*.log")) + list(path.glob("*.txt"))
            st.sidebar.info(f"Found {len(log_files)} log files")
            
            if log_files and st.sidebar.button("Load All Files", use_container_width=True):
                with st.spinner("Loading log files..."):
                    all_logs = []
                    for log_file in log_files:
                        try:
                            content = log_file.read_text(encoding="utf-8", errors="ignore")
                            all_logs.extend(parse_log_file(content))
                        except Exception as e:
                            st.sidebar.warning(f"Skipped {log_file.name}: {e}")
                    
                    if all_logs:
                        st.session_state.helix_service.reset()
                        st.session_state.raw_logs = all_logs
                        
                        chunks = st.session_state.helix_service.create_windowed_chunks(
                            logs=all_logs,
                            source_id="folder_import",
                            metadata_base={"folder": str(folder_path)}
                        )
                        
                        annotated_chunks = st.session_state.helix_service.annotate_chunks(chunks)
                        st.session_state.chunks = annotated_chunks
                        st.session_state.logs_loaded = True
                        st.session_state.indexed_for_search = False
                        
                        st.sidebar.success(f"Loaded {len(all_logs)} log lines")
                        st.rerun()
        elif folder_path:
            st.sidebar.error("Invalid folder path")
    
    st.sidebar.divider()
    
    # Settings
    st.sidebar.subheader("Settings")
    
    threshold = st.sidebar.slider(
        "Anomaly Threshold",
        min_value=0.05,
        max_value=0.50,
        value=st.session_state.helix_service.anomaly_threshold,
        step=0.05,
        help="Lower = more sensitive (flags more anomalies)"
    )
    
    if threshold != st.session_state.helix_service.anomaly_threshold:
        st.session_state.helix_service.anomaly_threshold = threshold
        if st.session_state.logs_loaded:
            # Re-annotate with new threshold
            chunks = st.session_state.helix_service.create_windowed_chunks(
                logs=st.session_state.raw_logs,
                source_id="reprocessed"
            )
            st.session_state.chunks = st.session_state.helix_service.annotate_chunks(chunks)
            st.rerun()
    
    # Reset button
    if st.sidebar.button("Reset Dashboard", use_container_width=True):
        st.session_state.helix_service.reset()
        st.session_state.logs_loaded = False
        st.session_state.raw_logs = []
        st.session_state.chunks = []
        st.session_state.indexed_for_search = False
        st.rerun()
    
    # Stats
    if st.session_state.logs_loaded:
        st.sidebar.divider()
        st.sidebar.subheader("Current Stats")
        stats = st.session_state.helix_service.get_stats()
        st.sidebar.metric("Clusters", stats.get("cluster_count", 0))
        st.sidebar.metric("Chunks", len(st.session_state.chunks))
        anomaly_count = sum(1 for c in st.session_state.chunks if c.is_anomaly)
        st.sidebar.metric("Anomalies", anomaly_count)


# ============================================================
# MAIN CONTENT SECTIONS
# ============================================================
def render_overview_metrics():
    """Render the overview metrics section."""
    st.subheader("Overview Metrics")
    
    helix = st.session_state.helix_service
    chunks = st.session_state.chunks
    codebook = helix.get_codebook()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_logs = len(st.session_state.raw_logs)
        st.metric(
            label="Total Logs",
            value=f"{total_logs:,}",
            delta=None
        )
    
    with col2:
        cluster_count = len(codebook)
        st.metric(
            label="Unique Clusters",
            value=cluster_count,
            help="Number of distinct log patterns identified by Drain3"
        )
    
    with col3:
        anomaly_count = sum(1 for c in chunks if c.is_anomaly)
        st.metric(
            label="Anomalies Detected",
            value=anomaly_count,
            delta=f"{(anomaly_count/max(len(chunks),1)*100):.1f}%" if chunks else "0%",
            delta_color="inverse"
        )
    
    with col4:
        error_clusters = sum(1 for info in codebook.values() 
                           if any(kw in info.get("template", "").upper() 
                                  for kw in ["ERROR", "FAIL", "CRITICAL", "FATAL"]))
        st.metric(
            label="Error Patterns",
            value=error_clusters,
            help="Clusters containing error-related keywords"
        )


def render_visualizations():
    """Render the main visualization charts."""
    st.subheader("Visualizations")
    
    helix = st.session_state.helix_service
    codebook = helix.get_codebook()
    transition_probs = helix.get_transition_probs()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_cluster_chart(codebook)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_transition_heatmap(transition_probs, codebook)
        st.plotly_chart(fig, use_container_width=True)


def render_template_viewer():
    """Render the Drain3 template viewer section."""
    st.subheader("Drain3 Template Viewer")
    
    helix = st.session_state.helix_service
    codebook = helix.get_codebook()
    
    if not codebook:
        st.info("No templates available. Upload a log file to begin analysis.")
        return
    
    # Build dataframe
    data = []
    for cluster_id, info in codebook.items():
        template = info.get("template", "")
        count = info.get("count", 0)
        severity = helix._get_severity_penalty(template)
        anomaly_type = helix._classify_anomaly_type(template)
        
        data.append({
            "Cluster ID": cluster_id,
            "Template": template,
            "Count": count,
            "Severity": severity,
            "Severity Label": get_severity_label(severity),
            "Anomaly Type": anomaly_type
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values("Count", ascending=False)
    
    # Filter options
    col1, col2 = st.columns([1, 3])
    with col1:
        severity_filter = st.selectbox(
            "Filter by Severity",
            ["All", "CRITICAL (≥0.8)", "HIGH (≥0.5)", "MEDIUM (≥0.3)", "LOW (<0.3)"]
        )
    
    with col2:
        search_term = st.text_input("Search Templates", placeholder="Type to filter...")
    
    # Apply filters
    filtered_df = df.copy()
    
    if severity_filter != "All":
        if "CRITICAL" in severity_filter:
            filtered_df = filtered_df[filtered_df["Severity"] >= 0.8]
        elif "HIGH" in severity_filter:
            filtered_df = filtered_df[(filtered_df["Severity"] >= 0.5) & (filtered_df["Severity"] < 0.8)]
        elif "MEDIUM" in severity_filter:
            filtered_df = filtered_df[(filtered_df["Severity"] >= 0.3) & (filtered_df["Severity"] < 0.5)]
        else:
            filtered_df = filtered_df[filtered_df["Severity"] < 0.3]
    
    if search_term:
        filtered_df = filtered_df[
            filtered_df["Template"].str.contains(search_term, case=False, na=False)
        ]
    
    # Display table
    st.dataframe(
        filtered_df[["Cluster ID", "Template", "Count", "Severity Label", "Anomaly Type"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Cluster ID": st.column_config.NumberColumn("ID", width="small"),
            "Template": st.column_config.TextColumn("Template Pattern", width="large"),
            "Count": st.column_config.NumberColumn("Count", width="small"),
            "Severity Label": st.column_config.TextColumn("Severity", width="medium"),
            "Anomaly Type": st.column_config.TextColumn("Type", width="medium")
        }
    )
    
    st.caption(f"Showing {len(filtered_df)} of {len(df)} templates")


def render_anomaly_results():
    """Render the anomaly detection results section with sorting and filtering."""
    st.subheader("Anomaly Detection Results")
    
    chunks = st.session_state.chunks
    anomalous_chunks = [c for c in chunks if c.is_anomaly]
    
    if not anomalous_chunks:
        st.info("No anomalies detected in the current log data.")
        return
    
    # Anomaly distribution chart - make it full width with summary
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        fig = create_anomaly_distribution(chunks)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Summary by type
        type_counts = {}
        for c in anomalous_chunks:
            t = c.anomaly_type or "unknown"
            type_counts[t] = type_counts.get(t, 0) + 1
        
        if type_counts:
            fig = px.pie(
                values=list(type_counts.values()),
                names=list(type_counts.keys()),
                title="By Type",
                hole=0.5,
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(
                template="plotly_dark",
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=10, r=10, t=40, b=10),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3)
            )
            fig.update_traces(textposition='inside', textinfo='value')
            st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        # Severity distribution
        severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for c in anomalous_chunks:
            if c.severity_weight >= 0.8:
                severity_counts["CRITICAL"] += 1
            elif c.severity_weight >= 0.5:
                severity_counts["HIGH"] += 1
            elif c.severity_weight >= 0.3:
                severity_counts["MEDIUM"] += 1
            else:
                severity_counts["LOW"] += 1
        
        fig = px.bar(
            x=list(severity_counts.keys()),
            y=list(severity_counts.values()),
            title="By Severity",
            color=list(severity_counts.keys()),
            color_discrete_map={
                "CRITICAL": "#ff4b4b",
                "HIGH": "#ff8c00",
                "MEDIUM": "#ffd700",
                "LOW": "#00d4ff"
            }
        )
        fig.update_layout(
            template="plotly_dark",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=40, b=10),
            showlegend=False,
            xaxis_title="",
            yaxis_title="Count"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detected Anomalies with Sorting Controls
    st.write("### Detected Anomalies")
    
    # Sorting and filtering controls
    control_col1, control_col2, control_col3 = st.columns([1, 1, 2])
    
    with control_col1:
        sort_by = st.selectbox(
            "Sort by",
            ["Score (High→Low)", "Score (Low→High)", "Severity (High→Low)", 
             "Cluster ID", "Type"],
            key="anomaly_sort"
        )
    
    with control_col2:
        items_per_page = st.selectbox(
            "Show",
            [10, 25, 50, 100],
            key="anomaly_per_page"
        )
    
    with control_col3:
        type_filter = st.multiselect(
            "Filter by Type",
            options=sorted(set(c.anomaly_type for c in anomalous_chunks if c.anomaly_type)),
            key="anomaly_type_filter"
        )
    
    # Apply sorting
    if sort_by == "Score (High→Low)":
        anomalous_chunks.sort(key=lambda x: x.anomaly_score, reverse=True)
    elif sort_by == "Score (Low→High)":
        anomalous_chunks.sort(key=lambda x: x.anomaly_score)
    elif sort_by == "Severity (High→Low)":
        anomalous_chunks.sort(key=lambda x: x.severity_weight, reverse=True)
    elif sort_by == "Cluster ID":
        anomalous_chunks.sort(key=lambda x: x.cluster_id or 0)
    elif sort_by == "Type":
        anomalous_chunks.sort(key=lambda x: x.anomaly_type or "zzz")
    
    # Apply type filter
    if type_filter:
        anomalous_chunks = [c for c in anomalous_chunks if c.anomaly_type in type_filter]
    
    # Pagination
    total_items = len(anomalous_chunks)
    total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
    
    page_col1, page_col2 = st.columns([1, 3])
    with page_col1:
        page = st.selectbox(
            "Page",
            range(1, total_pages + 1),
            format_func=lambda x: f"Page {x} of {total_pages}",
            key="anomaly_page"
        )
    with page_col2:
        st.caption(f"Showing {min(items_per_page, total_items - (page-1)*items_per_page)} of {total_items} anomalies")
    
    start_idx = (page - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, total_items)
    
    # Render anomaly cards - ALL CLOSED by default
    for i, chunk in enumerate(anomalous_chunks[start_idx:end_idx], start=start_idx + 1):
        severity_label = get_severity_label(chunk.severity_weight)
        source_info = chunk.metadata.get("filename", chunk.metadata.get("folder", "Unknown"))
        
        with st.expander(
            f"**#{i}** | {severity_label} | Score: {chunk.anomaly_score:.2f} | "
            f"Type: {chunk.anomaly_type} | Cluster: [{chunk.cluster_id}]",
            expanded=False  # All closed by default
        ):
            # Metrics row
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("Anomaly Score", f"{chunk.anomaly_score:.3f}")
            with metric_col2:
                st.metric("Transition Prob", f"{chunk.transition_prob:.3f}" if chunk.transition_prob else "N/A")
            with metric_col3:
                st.metric("Cluster ID", chunk.cluster_id)
            with metric_col4:
                st.metric("Source", source_info[:20])
            
            # Template
            st.write("**Template Pattern:**")
            st.code(chunk.cluster_template or "N/A", language="text")
            
            # Log content with syntax highlighting
            st.write("**Log Content:**")
            st.code(chunk.content, language="log")
            
            # Metadata in collapsed section
            if chunk.metadata:
                with st.popover("View Metadata"):
                    st.json(chunk.metadata)


def render_vector_search():
    """Render the vector search section (without LLM)."""
    st.subheader("Vector Search (No LLM)")
    
    helix = st.session_state.helix_service
    chunks = st.session_state.chunks
    
    if not st.session_state.logs_loaded:
        st.info("Load logs first to enable vector search.")
        return
    
    col1, col2 = st.columns(2)
    
    # ===== FAISS Semantic Search =====
    with col1:
        st.write("### Semantic Search (FAISS)")
        st.caption("Find logs by meaning using embeddings")
        
        query = st.text_input(
            "Search Query",
            placeholder="e.g., database connection error",
            key="semantic_search"
        )
        
        if query:
            try:
                # Get embedder and vector store
                embedder = get_embedder()
                vector_store = get_vector_store()
                
                # Index chunks if not already done
                if not st.session_state.indexed_for_search:
                    with st.spinner("Indexing logs for search..."):
                        texts = [c.content for c in chunks]
                        embeddings = embedder.embed_batch(texts)
                        chunk_ids = [c.id for c in chunks]
                        vector_store.clear()
                        vector_store.add(embeddings, chunk_ids)
                        st.session_state.indexed_for_search = True
                
                # Search
                query_embedding = embedder.embed_text(query)
                chunk_ids, scores = vector_store.search(query_embedding, k=5)
                
                if chunk_ids:
                    st.success(f"Found {len(chunk_ids)} results")
                    
                    # Create ID to chunk mapping
                    chunk_map = {c.id: c for c in chunks}
                    
                    for idx, (cid, score) in enumerate(zip(chunk_ids, scores), 1):
                        chunk = chunk_map.get(cid)
                        if chunk:
                            with st.container():
                                st.markdown(f"**#{idx}** | Similarity: `{score:.3f}`")
                                if chunk.is_anomaly:
                                    st.warning(f"Anomaly: {chunk.anomaly_type}")
                                st.code(chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content)
                                st.divider()
                else:
                    st.info("No matching logs found")
                    
            except Exception as e:
                st.error(f"Search error: {e}")
    
    # ===== Helix Pattern Search =====
    with col2:
        st.write("### Pattern Search (Helix)")
        st.caption("Find clusters by template pattern matching")
        
        pattern = st.text_input(
            "Pattern Query",
            placeholder="e.g., timeout, database, error",
            key="pattern_search"
        )
        
        if pattern:
            matches = helix.search_clusters_by_pattern(pattern)
            
            if matches:
                st.success(f"Found {len(matches)} matching clusters")
                
                for match in matches[:10]:  # Top 10
                    severity_label = get_severity_label(match["severity"])
                    
                    with st.container():
                        st.markdown(
                            f"**Cluster {match['cluster_id']}** | "
                            f"Count: `{match['count']}` | "
                            f"Severity: {severity_label}"
                        )
                        st.code(match["template"])
                        
                        # Show transition context
                        with st.expander("View Transitions"):
                            context = helix.get_transition_context(match["cluster_id"])
                            
                            if context.get("incoming_transitions"):
                                st.write("**Incoming (what triggers this):**")
                                for t in context["incoming_transitions"][:3]:
                                    st.caption(f"← [{t['from_cluster']}] {t['template'][:50]}... ({t['probability']:.1%})")
                            
                            if context.get("outgoing_transitions"):
                                st.write("**Outgoing (what follows):**")
                                for t in context["outgoing_transitions"][:3]:
                                    st.caption(f"→ [{t['to_cluster']}] {t['template'][:50]}... ({t['probability']:.1%})")
                        
                        st.divider()
            else:
                st.info("No matching patterns found")


# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    """Main application entry point."""
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    if not st.session_state.logs_loaded:
        # Welcome screen
        st.markdown("""
        <div style="text-align: center; padding: 50px;">
            <h2>Welcome to the Anomaly Detection Dashboard</h2>
            <p style="color: #888; font-size: 18px;">
                Upload a log file or specify a folder path in the sidebar to begin analysis.
            </p>
            <br>
            <div style="display: flex; justify-content: center; gap: 40px; flex-wrap: wrap;">
                <div style="text-align: center;">
                    <p><strong>Upload</strong><br>Log files</p>
                </div>
                <div style="text-align: center;">
                    <p><strong>Analyze</strong><br>Patterns & Clusters</p>
                </div>
                <div style="text-align: center;">
                    <p><strong>Detect</strong><br>Anomalies</p>
                </div>
                <div style="text-align: center;">
                    <p><strong>Search</strong><br>Vector DB</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Dashboard sections
    render_overview_metrics()
    
    st.divider()
    render_visualizations()
    
    st.divider()
    render_template_viewer()
    
    st.divider()
    render_anomaly_results()
    
    st.divider()
    render_vector_search()
    
    # Footer
    st.divider()
    st.caption(
        f"Dashboard loaded at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"Powered by Drain3, FAISS, and Sentry-AI Services"
    )


if __name__ == "__main__":
    main()
