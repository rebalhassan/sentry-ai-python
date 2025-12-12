# sentry/integrations/datadog.py
"""
DataDog integration for fetching logs and metrics.

DataDog API Documentation: https://docs.datadoghq.com/api/

Authentication:
- API Key (DD-API-KEY header)
- Application Key (DD-APPLICATION-KEY header)
- Both created in DataDog Dashboard > Organization Settings > API Keys

Endpoints used:
- POST /api/v2/logs/events/search - Search logs
- GET /api/v1/logs-indexes - List log indexes
"""

import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any

import httpx

from .base import (
    BaseIntegration,
    IntegrationError,
    RateLimitError,
    AuthenticationError,
    LogEntry,
    LogLevel,
    Project,
    Deployment,
)

logger = logging.getLogger(__name__)


class DataDogIntegration(BaseIntegration):
    """
    DataDog integration for log analytics.
    
    Usage:
        >>> creds = {
        ...     "api_key": "xxx",
        ...     "app_key": "xxx",
        ...     "site": "datadoghq.com"
        ... }
        >>> dd = DataDogIntegration(creds)
        >>> logs = await dd.fetch_logs(query="service:my-app status:error")
    """
    
    SERVICE_NAME = "datadog"
    
    # DataDog sites/regions
    SITES = {
        "us1": "https://api.datadoghq.com",
        "us3": "https://api.us3.datadoghq.com",
        "us5": "https://api.us5.datadoghq.com",
        "eu1": "https://api.datadoghq.eu",
        "ap1": "https://api.ap1.datadoghq.com",
        # Also accept domain-style
        "datadoghq.com": "https://api.datadoghq.com",
        "us3.datadoghq.com": "https://api.us3.datadoghq.com",
        "us5.datadoghq.com": "https://api.us5.datadoghq.com",
        "datadoghq.eu": "https://api.datadoghq.eu",
        "ap1.datadoghq.com": "https://api.ap1.datadoghq.com",
    }
    
    # Map DataDog status to our log levels
    LEVEL_MAP = {
        "debug": LogLevel.DEBUG,
        "info": LogLevel.INFO,
        "warn": LogLevel.WARNING,
        "warning": LogLevel.WARNING,
        "error": LogLevel.ERROR,
        "critical": LogLevel.CRITICAL,
        "emergency": LogLevel.CRITICAL,
        "alert": LogLevel.CRITICAL,
    }
    
    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.api_key = credentials["api_key"]
        self.app_key = credentials["app_key"]
        site = credentials.get("site", "us1").lower()
        self.base_url = self.SITES.get(site, self.SITES["us1"])
    
    def _validate_credentials_format(self) -> None:
        """Validate required credential fields."""
        missing = []
        if "api_key" not in self.credentials:
            missing.append("api_key")
        if "app_key" not in self.credentials:
            missing.append("app_key")
        
        if missing:
            raise ValueError(f"DataDog integration requires: {missing}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        return {
            "DD-API-KEY": self.api_key,
            "DD-APPLICATION-KEY": self.app_key,
            "Content-Type": "application/json",
        }
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make an HTTP request to the DataDog API."""
        url = f"{self.base_url}{endpoint}"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.request(
                    method,
                    url,
                    headers=self._get_headers(),
                    params=params,
                    json=json_data,
                )
                
                # Check for rate limiting
                if response.status_code == 429:
                    retry_after = response.headers.get("x-ratelimit-reset")
                    raise RateLimitError(
                        self.SERVICE_NAME,
                        retry_after=int(retry_after) if retry_after else 60,
                        details={"endpoint": endpoint}
                    )
                
                # Check for auth errors
                if response.status_code in (401, 403):
                    raise AuthenticationError(
                        self.SERVICE_NAME,
                        details={"status": response.status_code}
                    )
                
                # Check for other errors
                if response.status_code >= 400:
                    error_data = response.json() if response.content else {}
                    raise IntegrationError(
                        f"API error: {response.status_code}",
                        self.SERVICE_NAME,
                        details={"status": response.status_code, "error": error_data}
                    )
                
                return response.json()
                
            except httpx.TimeoutException:
                raise IntegrationError(
                    "Request timed out",
                    self.SERVICE_NAME,
                    details={"endpoint": endpoint}
                )
            except httpx.RequestError as e:
                raise IntegrationError(
                    f"Request failed: {str(e)}",
                    self.SERVICE_NAME,
                    details={"endpoint": endpoint}
                )
    
    async def validate_credentials(self) -> bool:
        """Validate credentials by checking API key validity."""
        try:
            await self._make_request("GET", "/api/v1/validate")
            logger.info("DataDog credentials validated successfully")
            return True
        except AuthenticationError:
            return False
    
    async def list_projects(self, limit: int = 20) -> List[Project]:
        """
        List DataDog services (as projects).
        
        DataDog organizes by services, not projects, so we query
        for distinct services from recent logs.
        """
        # Query for distinct services
        now = datetime.now()
        start = now - timedelta(hours=24)
        
        try:
            data = await self._make_request(
                "POST",
                "/api/v2/logs/events/search",
                json_data={
                    "filter": {
                        "from": start.isoformat() + "Z",
                        "to": now.isoformat() + "Z",
                    },
                    "page": {"limit": 1},
                    "options": {"timeOffset": 0},
                }
            )
            
            # Get aggregations for services if available
            # For now, return log indexes as "projects"
            return await self._list_indexes_as_projects()
            
        except Exception as e:
            logger.warning(f"Could not list services: {e}")
            return await self._list_indexes_as_projects()
    
    async def _list_indexes_as_projects(self) -> List[Project]:
        """List log indexes as projects."""
        try:
            data = await self._make_request("GET", "/api/v1/logs/config/indexes")
            
            projects = []
            for idx in data.get("indexes", []):
                projects.append(Project(
                    id=idx.get("name", "main"),
                    name=idx.get("name", "Main Index"),
                    metadata={
                        "filter": idx.get("filter", {}).get("query"),
                        "retention_days": idx.get("num_retention_days"),
                    }
                ))
            
            # Always include a default if empty
            if not projects:
                projects.append(Project(
                    id="main",
                    name="Main Index",
                ))
            
            return projects
            
        except Exception:
            return [Project(id="main", name="Main Index")]
    
    async def list_deployments(
        self, 
        project_id: str,
        limit: int = 20
    ) -> List[Deployment]:
        """
        DataDog doesn't have deployments, so we return time periods.
        
        Similar to PostHog, we use time-based filtering.
        """
        now = datetime.now()
        return [
            Deployment(
                id="last_15m",
                name="Last 15 Minutes",
                state="active",
                created_at=now - timedelta(minutes=15),
            ),
            Deployment(
                id="last_1h",
                name="Last 1 Hour",
                state="active",
                created_at=now - timedelta(hours=1),
            ),
            Deployment(
                id="last_4h",
                name="Last 4 Hours",
                state="active",
                created_at=now - timedelta(hours=4),
            ),
            Deployment(
                id="last_1d",
                name="Last 1 Day",
                state="active",
                created_at=now - timedelta(days=1),
            ),
            Deployment(
                id="last_7d",
                name="Last 7 Days",
                state="active",
                created_at=now - timedelta(days=7),
            ),
        ]
    
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
        Fetch logs from DataDog.
        
        Args:
            project_id: Log index name (optional)
            deployment_id: Time period filter (last_15m, last_1h, etc.)
            start_time: Start of time range
            end_time: End of time range
            query: DataDog log query syntax (e.g., "service:my-app status:error")
            limit: Maximum logs to return
            
        Returns:
            List of LogEntry objects
        """
        # Determine time range from deployment_id
        now = datetime.now()
        if deployment_id:
            end_time = now
            if deployment_id == "last_15m":
                start_time = now - timedelta(minutes=15)
            elif deployment_id == "last_1h":
                start_time = now - timedelta(hours=1)
            elif deployment_id == "last_4h":
                start_time = now - timedelta(hours=4)
            elif deployment_id == "last_1d":
                start_time = now - timedelta(days=1)
            elif deployment_id == "last_7d":
                start_time = now - timedelta(days=7)
        
        if not start_time:
            start_time = now - timedelta(hours=1)
        if not end_time:
            end_time = now
        
        # Build query filter
        filter_query = query or "*"
        if project_id and project_id != "main":
            # Add index filter if specified
            filter_query = f"@index:{project_id} {filter_query}"
        
        # Make the search request
        request_body = {
            "filter": {
                "query": filter_query,
                "from": start_time.isoformat() + "Z",
                "to": end_time.isoformat() + "Z",
            },
            "sort": "timestamp",
            "page": {"limit": min(limit, 1000)},  # DataDog max is 1000
        }
        
        data = await self._make_request(
            "POST",
            "/api/v2/logs/events/search",
            json_data=request_body
        )
        
        logs = []
        for log_data in data.get("data", []):
            attrs = log_data.get("attributes", {})
            
            # Parse timestamp
            ts = attrs.get("timestamp")
            if ts:
                if isinstance(ts, str):
                    timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                else:
                    timestamp = datetime.fromtimestamp(ts / 1000 if ts > 1e10 else ts)
            else:
                timestamp = datetime.now()
            
            # Parse log level
            status = attrs.get("status", "info").lower()
            level = self.LEVEL_MAP.get(status, LogLevel.INFO)
            
            # Get message
            message = attrs.get("message", "")
            
            # Get service and host info
            service = attrs.get("service", "unknown")
            host = attrs.get("host", "")
            
            # Build source identifier
            source = f"datadog/{service}"
            if host:
                source += f"/{host}"
            
            logs.append(LogEntry(
                id=f"datadog_{log_data.get('id', len(logs))}",
                timestamp=timestamp,
                message=message,
                level=level,
                source=source,
                metadata={
                    "service": service,
                    "host": host,
                    "status": status,
                    "tags": attrs.get("tags", []),
                    "attributes": attrs.get("attributes", {}),
                }
            ))
        
        logger.info(f"Fetched {len(logs)} logs from DataDog")
        return logs
    
    async def fetch_errors(self, service: Optional[str] = None, limit: int = 50) -> List[LogEntry]:
        """
        Convenience method to fetch error logs.
        """
        query = "status:error"
        if service:
            query = f"service:{service} {query}"
        return await self.fetch_logs(query=query, limit=limit)
    
    async def fetch_by_service(self, service: str, limit: int = 100) -> List[LogEntry]:
        """
        Fetch logs for a specific service.
        """
        return await self.fetch_logs(query=f"service:{service}", limit=limit)
