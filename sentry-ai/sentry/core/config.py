# sentry/core/config.py
"""
Configuration management for Sentry-AI
All settings in one place, can be overridden via environment variables
"""

from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, model_validator


class Settings(BaseSettings):
    """
    Global application settings
    Can be overridden with SENTRY_* environment variables
    """
    
    # ===== APP METADATA =====
    app_name: str = "Sentry-AI"
    version: str = "0.1.0"
    codename: str = "Cyrus"
    
    # ===== STORAGE PATHS =====
    data_dir: Path = Field(default=Path.home() / ".sentry-ai")
    db_path: Optional[Path] = Field(default=None)  # Made Optional
    vector_index_path: Optional[Path] = Field(default=None)  # Made Optional
    cache_dir: Optional[Path] = Field(default=None)  # Made Optional
    
    # ===== EMBEDDING MODEL =====
    # Upgraded to mpnet for +15% accuracy (MTEB: 64.0 vs 56.3)
    # Note: Changing model requires re-indexing all content
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384  # Output dimension of all-mpnet-base-v2
    embedding_batch_size: int = 32  # Process this many chunks at once
    
    # ===== CHUNKING STRATEGY =====
    # Increased for semantic chunking - groups complete log entries
    chunk_size: int = 1000  # Characters per chunk (increased for semantic chunking)
    chunk_overlap: int = 100  # Overlap between chunks (helps with context)
    min_chunk_size: int = 50  # Don't create tiny chunks
    
    # ===== VECTOR SEARCH =====
    top_k_results: int = 20  # Return top 20 most similar chunks
    similarity_threshold: float = 0.25  # Minimum similarity score (0-1)
    use_reranking: bool = True  # Use BM25 reranking after vector search
    
    # ===== LLM SETTINGS =====
    ollama_host: str = "http://localhost:11434"
    llm_model: str = "tinyllama:latest"
    main_prompt: str = """
    <Role>
                    You are a cybersecurity expert with deep knowledge of NIST standards, frameworks, and best practices. 
                    </Role>
                    <Task>
                    You provide accurate, detailed guidance on cybersecurity controls, risk management, cloud security, and compliance based on NIST publications including the 800 series, FIPS, and related documents. 
                    </Task>
                    What information does Security Content Automation Protocol (SCAP) Version 1.2 Validation Program Test Requirements provide? (Section 3) ; SCAP validated modules; SCAP validation The authors, Melanie Cook, Stephen Quinn, and David Waltermire of the National Institute of Standards and Technology (NIST), and Dragos Prisaca of G2, Inc. would like to thank the many people who reviewed and contributed to this document, in particular, John Banghart of Microsoft who was the original author and pioneered the first SCAP Validation Program. The authors thank Matt Kerr, and Danny Haynes of the MITRE Corporation for their insightful technical contribution to the design of the SCAP 1.2 Validation Program and creation of original SCAP 1.2 validation test content. We also thank our document reviewers, Kelley Dempsey of NIST and Jeffrey Blank of the National Security Agency for their input. This publication is intended for NVLAP accredited laboratories conducting SCAP product and module testing for the program, vendors interested in receiving SCAP validation for their products or modules, and organizations deploying SCAP products in their environments. Accredited laboratories use the information in this report to guide their testing and ensure all necessary requirements are met by a product before recommending to NIST that the product be awarded the requested validation. Vendors may use the information in this report to understand the features that products and modules need in order to be eligible for an SCAP validation. Government agencies and integrators use the information to gain insight into the criteria required for SCAP validated products. The secondary audience for this publication includes end users, who can review the test requirements in order to understand the capabilities of SCAP validated products and gain knowledge about SCAP validation. OVAL and CVE are registered trademarks, and CCE, CPE, and OCIL are trademarks of The MITRE Corporation. Red Hat is a registered trademark of Red Hat, Inc. Windows operating system is registered trademark of Microsoft Corporation.
                    
    """
    llm_temperature: float = 0.70  # Low = factual, High = creative
    llm_context_window: int = 2048  # Max tokens for context
    llm_max_tokens: int = 1024  # Max tokens in response
    llm_timeout: int = 60  # Seconds before timeout
    
    # Context generation (for embeddings) - uses a smaller/faster model
    context_model: str = "tinyllama:latest"  # Small model for fast context summaries
    context_temperature: float = 0.3  # Lower for consistent summaries
    context_max_tokens: int = 150  # Keep summaries concise
    
    # Query Expansion
    query_expansion_temperature: float = 0.2  # Low for deterministic expansions
    
    # ===== INDEXING =====
    max_file_size_mb: int = 100  # Skip files larger than this
    supported_extensions: List[str] = Field(default=[".log", ".txt", ".out", ".err"])
    index_on_startup: bool = True  # Auto-index all sources on app start
    parallel_indexing: bool = True  # Index multiple sources concurrently
    max_workers: int = 4  # Number of parallel indexing threads
    
    # ===== FILE WATCHING =====
    watch_enabled: bool = True  # Monitor files for changes
    watch_poll_interval: float = 2.0  # Seconds between checks
    watch_debounce: float = 1.0  # Wait this long after change before indexing
    
    # ===== EVENTVIEWER (Windows) =====
    eventviewer_enabled: bool = True  # Enable EventViewer integration
    eventviewer_poll_interval: int = 5  # Seconds between polls
    eventviewer_max_events: int = 1000  # Max events to fetch per poll
    eventviewer_default_logs: List[str] = Field(default=["System", "Application"])
    
    # ===== PERFORMANCE =====
    cache_embeddings: bool = True  # Cache embeddings to avoid recomputation
    cache_llm_responses: bool = False  # Cache LLM responses (for testing)
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    
    # ===== UI SETTINGS =====
    chat_history_limit: int = 50  # Keep last N messages
    show_sources: bool = True  # Show source chunks in responses
    max_source_preview_chars: int = 200  # Truncate long source previews
    
    model_config = SettingsConfigDict(
        env_prefix="SENTRY_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
    
    @field_validator("data_dir", mode="before")
    @classmethod
    def expand_data_dir(cls, v):
        """Expand ~ and resolve path"""
        if isinstance(v, str):
            return Path(v).expanduser().resolve()
        return v
    
    @model_validator(mode="after")
    def init_paths(self):
        """Initialize derived paths after all fields are set"""
        # Set default paths if not provided
        if self.db_path is None:
            self.db_path = self.data_dir / "sentry.db"
        
        if self.vector_index_path is None:
            self.vector_index_path = self.data_dir / "vectors.faiss"
        
        if self.cache_dir is None:
            self.cache_dir = self.data_dir / "cache"
        
        # Ensure paths are Path objects
        if isinstance(self.db_path, str):
            self.db_path = Path(self.db_path)
        if isinstance(self.vector_index_path, str):
            self.vector_index_path = Path(self.vector_index_path)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        return self
    
    def get_model_cache_path(self) -> Path:
        """Path where embedding model will be cached"""
        return self.cache_dir / "models"
    
    def is_supported_file(self, file_path: Path) -> bool:
        """Check if a file extension is supported"""
        return file_path.suffix.lower() in self.supported_extensions
    
    def is_file_too_large(self, file_path: Path) -> bool:
        """Check if a file exceeds max size"""
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            return size_mb > self.max_file_size_mb
        except:
            return False
    
    def __repr__(self):
        return f"<Settings(app={self.app_name} v{self.version}, data_dir={self.data_dir})>"


# ===== GLOBAL SETTINGS INSTANCE =====
# This is imported throughout the app
settings = Settings()


# ===== HELPER FUNCTIONS =====

def get_settings() -> Settings:
    """
    Get the global settings instance
    Useful for dependency injection in tests
    """
    return settings


def reload_settings():
    """
    Reload settings from environment
    Useful if env vars change during runtime
    """
    global settings
    settings = Settings()
    return settings