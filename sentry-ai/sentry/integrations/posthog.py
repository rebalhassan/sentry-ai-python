# sentry/integrations/posthog.py
"""
PostHog integration for fetching analytics events and logs.

PostHog API Documentation: https://posthog.com/docs/api

Authentication:
- Personal API Key (for private endpoints)
- Created in PostHog Dashboard > Project Settings > Personal API Keys

Endpoints used:
- GET /api/projects/ - List projects
- POST /api/projects/:project_id/query/ - Query events using HogQL
- GET /api/projects/:project_id/logs/query/ - Query logs (if OTLP enabled)
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


class PostHogIntegration(BaseIntegration):
    """
    PostHog integration for event and log data.
    
    Usage:
        >>> creds = {"api_key": "phx_xxx", "project_id": "12345", "region": "us"}
        >>> posthog = PostHogIntegration(creds)
        >>> logs = await posthog.fetch_logs(query="$pageview")
    """
    
    SERVICE_NAME = "posthog"
    
    # Regional base URLs
    REGIONS = {
        "us": "https://us.posthog.com",
        "eu": "https://eu.posthog.com",
    }
    
    # Map PostHog event levels to our levels
    LEVEL_MAP = {
        "debug": LogLevel.DEBUG,
        "info": LogLevel.INFO,
        "warning": LogLevel.WARNING,
        "warn": LogLevel.WARNING,
        "error": LogLevel.ERROR,
        "critical": LogLevel.CRITICAL,
        "fatal": LogLevel.CRITICAL,
    }
    
    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.api_key = credentials["api_key"]
        self.project_id = credentials["project_id"]
        self.region = credentials.get("region", "us").lower()
        self.base_url = self.REGIONS.get(self.region, self.REGIONS["us"])
    
    def _validate_credentials_format(self) -> None:
        """Validate required credential fields."""
        missing = []
        if "api_key" not in self.credentials:
            missing.append("api_key")
        if "project_id" not in self.credentials:
            missing.append("project_id")
        
        if missing:
            raise ValueError(f"PostHog integration requires: {missing}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make an HTTP request to the PostHog API."""
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
                    retry_after = response.headers.get("retry-after", "60")
                    raise RateLimitError(
                        self.SERVICE_NAME,
                        retry_after=int(retry_after),
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
        """Validate credentials by fetching project info."""
        try:
            await self._make_request(
                "GET",
                f"/api/projects/{self.project_id}/"
            )
            logger.info("PostHog credentials validated successfully")
            return True
        except AuthenticationError:
            return False
    
    async def list_projects(self, limit: int = 20) -> List[Project]:
        """
        List PostHog projects/organizations.
        
        Note: Most users only have access to their own project.
        """
        data = await self._make_request("GET", "/api/projects/")
        
        projects = []
        for proj in data.get("results", []):
            projects.append(Project(
                id=str(proj["id"]),
                name=proj.get("name", "Unnamed Project"),
                url=f"{self.base_url}/project/{proj['id']}",
                created_at=datetime.fromisoformat(proj["created_at"].replace("Z", "+00:00")) if proj.get("created_at") else None,
                metadata={
                    "organization": proj.get("organization"),
                    "timezone": proj.get("timezone"),
                }
            ))
        
        logger.info(f"Found {len(projects)} PostHog projects")
        return projects
    
    async def list_deployments(
        self, 
        project_id: str,
        limit: int = 20
    ) -> List[Deployment]:
        """
        PostHog doesn't have deployments in the traditional sense.
        
        Instead, we return a list of distinct sessions or time periods
        that can be queried for events.
        """
        # Return recent time periods as "deployments"
        now = datetime.now()
        return [
            Deployment(
                id="last_1h",
                name="Last 1 Hour",
                state="active",
                created_at=now - timedelta(hours=1),
            ),
            Deployment(
                id="last_24h", 
                name="Last 24 Hours",
                state="active",
                created_at=now - timedelta(hours=24),
            ),
            Deployment(
                id="last_7d",
                name="Last 7 Days",
                state="active", 
                created_at=now - timedelta(days=7),
            ),
            Deployment(
                id="last_30d",
                name="Last 30 Days",
                state="active",
                created_at=now - timedelta(days=30),
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
        Fetch events/logs from PostHog.
        
        Args:
            project_id: Project ID (defaults to configured project)
            deployment_id: Time period filter (last_1h, last_24h, etc.)
            start_time: Start of time range
            end_time: End of time range
            query: HogQL WHERE clause or event name filter
            limit: Maximum events to return
            
        Returns:
            List of LogEntry objects
        """
        proj_id = project_id or self.project_id
        
        # Determine time range
        if deployment_id:
            end_time = datetime.now()
            if deployment_id == "last_1h":
                start_time = end_time - timedelta(hours=1)
            elif deployment_id == "last_24h":
                start_time = end_time - timedelta(hours=24)
            elif deployment_id == "last_7d":
                start_time = end_time - timedelta(days=7)
            elif deployment_id == "last_30d":
                start_time = end_time - timedelta(days=30)
        
        if not start_time:
            start_time = datetime.now() - timedelta(hours=24)
        if not end_time:
            end_time = datetime.now()
        
        # Build HogQL query
        where_clause = ""
        if query:
            # If query looks like an event name, filter by it
            if query.startswith("$") or not " " in query:
                where_clause = f"WHERE event = '{query}'"
            else:
                where_clause = f"WHERE {query}"
        
        hogql_query = f"""
        SELECT 
            uuid,
            event,
            timestamp,
            properties,
            distinct_id
        FROM events
        {where_clause}
        ORDER BY timestamp DESC
        LIMIT {limit}
        """
        
        # Execute query
        data = await self._make_request(
            "POST",
            f"/api/projects/{proj_id}/query/",
            json_data={
                "query": {
                    "kind": "HogQLQuery",
                    "query": hogql_query,
                },
                "name": "sentry-ai-log-fetch"
            }
        )
        
        logs = []
        results = data.get("results", [])
        columns = data.get("columns", ["uuid", "event", "timestamp", "properties", "distinct_id"])
        
        for row in results:
            # Map row to column names
            row_dict = dict(zip(columns, row))
            
            # Parse timestamp
            ts = row_dict.get("timestamp")
            if isinstance(ts, str):
                timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            elif isinstance(ts, (int, float)):
                timestamp = datetime.fromtimestamp(ts)
            else:
                timestamp = datetime.now()
            
            # Apply time filters
            if start_time and timestamp < start_time:
                continue
            if end_time and timestamp > end_time:
                continue
            
            # Parse properties
            properties = row_dict.get("properties", {})
            if isinstance(properties, str):
                import json
                try:
                    properties = json.loads(properties)
                except:
                    properties = {}
            
            # Determine log level from event or properties
            event_name = row_dict.get("event", "")
            level = LogLevel.INFO
            if "error" in event_name.lower():
                level = LogLevel.ERROR
            elif "warning" in event_name.lower() or "warn" in event_name.lower():
                level = LogLevel.WARNING
            elif properties.get("$level"):
                level = self.LEVEL_MAP.get(properties["$level"].lower(), LogLevel.INFO)
            
            # Build message
            message = f"Event: {event_name}"
            if properties.get("$current_url"):
                message += f" | URL: {properties['$current_url']}"
            if properties.get("$exception_message"):
                message += f" | Error: {properties['$exception_message']}"
            
            logs.append(LogEntry(
                id=f"posthog_{row_dict.get('uuid', len(logs))}",
                timestamp=timestamp,
                message=message,
                level=level,
                source=f"posthog/{proj_id}",
                metadata={
                    "event": event_name,
                    "distinct_id": row_dict.get("distinct_id"),
                    "properties": properties,
                }
            ))
        
        logger.info(f"Fetched {len(logs)} events from PostHog project {proj_id}")
        return logs
    
    async def fetch_exceptions(self, limit: int = 50) -> List[LogEntry]:
        """
        Convenience method to fetch exception events.
        """
        return await self.fetch_logs(query="$exception", limit=limit)
    
    async def fetch_pageviews(self, limit: int = 50) -> List[LogEntry]:
        """
        Convenience method to fetch pageview events.
        """
        return await self.fetch_logs(query="$pageview", limit=limit)
