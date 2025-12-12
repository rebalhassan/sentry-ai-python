# sentry/integrations/base.py
"""
Base class for external service integrations.

All integrations inherit from BaseIntegration and implement:
- Credential validation
- Resource listing (projects, deployments, etc.)
- Log fetching
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class IntegrationError(Exception):
    """Base exception for integration errors."""
    
    def __init__(self, message: str, service: str, details: Optional[Dict] = None):
        self.message = message
        self.service = service
        self.details = details or {}
        super().__init__(f"[{service}] {message}")


class RateLimitError(IntegrationError):
    """Raised when API rate limit is exceeded."""
    
    def __init__(
        self, 
        service: str, 
        retry_after: Optional[int] = None,
        details: Optional[Dict] = None
    ):
        self.retry_after = retry_after
        message = "Rate limit exceeded."
        if retry_after:
            message += f" Retry after {retry_after} seconds."
        super().__init__(message, service, details)


class AuthenticationError(IntegrationError):
    """Raised when authentication fails."""
    
    def __init__(self, service: str, details: Optional[Dict] = None):
        super().__init__(
            "Authentication failed. Please check your API credentials.",
            service,
            details
        )


class LogLevel(Enum):
    """Log level for external logs."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class LogEntry:
    """
    A log entry from an external service.
    
    This is the standardized format that all integrations convert to.
    It can then be converted to LogChunk for indexing.
    """
    id: str
    timestamp: datetime
    message: str
    level: LogLevel = LogLevel.INFO
    source: str = ""  # e.g., "vercel/my-app/deployment-123"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_content(self) -> str:
        """Convert to content string for embedding."""
        parts = []
        if self.source:
            parts.append(f"[{self.source}]")
        parts.append(f"[{self.level.value.upper()}]")
        parts.append(f"[{self.timestamp.isoformat()}]")
        parts.append(self.message)
        return " ".join(parts)


@dataclass
class Deployment:
    """A deployment/project from an external service."""
    id: str
    name: str
    url: Optional[str] = None
    state: str = ""
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Project:
    """A project from an external service."""
    id: str
    name: str
    url: Optional[str] = None
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseIntegration(ABC):
    """
    Abstract base class for external service integrations.
    
    All integrations must implement:
    - validate_credentials(): Check if stored credentials are valid
    - list_projects(): List available projects/accounts
    - list_deployments(): List deployments for a project
    - fetch_logs(): Fetch logs for a deployment/project
    """
    
    SERVICE_NAME: str = "base"
    
    def __init__(self, credentials: Dict[str, str]):
        """
        Initialize the integration with credentials.
        
        Args:
            credentials: Dictionary containing API keys and other auth info
        """
        self.credentials = credentials
        self._validate_credentials_format()
    
    @abstractmethod
    def _validate_credentials_format(self) -> None:
        """
        Validate that required credential fields are present.
        
        Raises:
            ValueError: If required fields are missing
        """
        pass
    
    @abstractmethod
    async def validate_credentials(self) -> bool:
        """
        Validate credentials by making a test API call.
        
        Returns:
            True if credentials are valid
            
        Raises:
            AuthenticationError: If authentication fails
            IntegrationError: If validation fails for other reasons
        """
        pass
    
    @abstractmethod
    async def list_projects(self) -> List[Project]:
        """
        List available projects/accounts.
        
        Returns:
            List of Project objects
            
        Raises:
            IntegrationError: If the API call fails
        """
        pass
    
    @abstractmethod
    async def list_deployments(
        self, 
        project_id: str,
        limit: int = 20
    ) -> List[Deployment]:
        """
        List deployments for a project.
        
        Args:
            project_id: ID of the project
            limit: Maximum number of deployments to return
            
        Returns:
            List of Deployment objects
            
        Raises:
            IntegrationError: If the API call fails
        """
        pass
    
    @abstractmethod
    async def fetch_logs(
        self,
        project_id: Optional[str] = None,
        deployment_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        query: Optional[str] = None,
        limit: int = 100
    ) -> List[LogEntry]:
        """
        Fetch logs from the service.
        
        Args:
            project_id: Filter by project
            deployment_id: Filter by deployment
            start_time: Start of time range
            end_time: End of time range
            query: Search query (service-specific)
            limit: Maximum number of logs to return
            
        Returns:
            List of LogEntry objects
            
        Raises:
            RateLimitError: If rate limit is exceeded
            IntegrationError: If the API call fails
        """
        pass
    
    def _handle_rate_limit(self, response_headers: Dict[str, str]) -> None:
        """
        Check response headers for rate limit info and raise if exceeded.
        
        Args:
            response_headers: HTTP response headers
            
        Raises:
            RateLimitError: If rate limit headers indicate limit exceeded
        """
        # Common rate limit headers
        remaining = response_headers.get("x-ratelimit-remaining")
        retry_after = response_headers.get("retry-after")
        
        if retry_after:
            try:
                seconds = int(retry_after)
            except ValueError:
                seconds = 60  # Default
            raise RateLimitError(self.SERVICE_NAME, retry_after=seconds)
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(service={self.SERVICE_NAME})>"
