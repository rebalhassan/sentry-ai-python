# sentry/integrations/vercel.py
"""
Vercel integration for fetching deployment logs.

Vercel API Documentation: https://vercel.com/docs/rest-api

Authentication:
- Bearer token using Vercel Access Token
- Token created in Vercel Dashboard > Account Settings > Tokens

Endpoints used:
- GET /v9/projects - List projects
- GET /v6/deployments - List deployments
- GET /v2/deployments/{id}/events - Get deployment events/logs
"""

import logging
from datetime import datetime
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


class VercelIntegration(BaseIntegration):
    """
    Vercel integration for deployment logs.
    
    Usage:
        >>> creds = {"api_key": "your_vercel_token", "team_id": "team_xxx"}
        >>> vercel = VercelIntegration(creds)
        >>> projects = await vercel.list_projects()
        >>> deployments = await vercel.list_deployments(projects[0].id)
        >>> logs = await vercel.fetch_logs(deployment_id=deployments[0].id)
    """
    
    SERVICE_NAME = "vercel"
    BASE_URL = "https://api.vercel.com"
    
    # Map Vercel log types to our log levels
    LEVEL_MAP = {
        "stdout": LogLevel.INFO,
        "stderr": LogLevel.ERROR,
        "error": LogLevel.ERROR,
        "warning": LogLevel.WARNING,
        "info": LogLevel.INFO,
        "debug": LogLevel.DEBUG,
    }
    
    def __init__(self, credentials: Dict[str, str]):
        super().__init__(credentials)
        self.api_key = credentials["api_key"]
        self.team_id = credentials.get("team_id")
    
    def _validate_credentials_format(self) -> None:
        """Validate required credential fields."""
        if "api_key" not in self.credentials:
            raise ValueError("Vercel integration requires 'api_key'")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def _get_params(self, **kwargs) -> Dict[str, Any]:
        """Get query parameters, adding team_id if configured."""
        params = {k: v for k, v in kwargs.items() if v is not None}
        if self.team_id:
            params["teamId"] = self.team_id
        return params
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make an HTTP request to the Vercel API.
        
        Handles common error cases and returns parsed JSON.
        """
        url = f"{self.BASE_URL}{endpoint}"
        params = self._get_params(**(params or {}))
        
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
                    retry_after = response.headers.get("retry-after")
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
        """Validate credentials by fetching user info."""
        try:
            await self._make_request("GET", "/v2/user")
            logger.info("Vercel credentials validated successfully")
            return True
        except AuthenticationError:
            return False
    
    async def list_projects(self, limit: int = 20) -> List[Project]:
        """
        List Vercel projects.
        
        Args:
            limit: Maximum number of projects to return
            
        Returns:
            List of Project objects
        """
        data = await self._make_request(
            "GET", 
            "/v9/projects",
            params={"limit": limit}
        )
        
        projects = []
        for proj in data.get("projects", []):
            projects.append(Project(
                id=proj["id"],
                name=proj.get("name", "Unnamed"),
                url=f"https://vercel.com/{proj.get('name', '')}",
                created_at=datetime.fromtimestamp(proj["createdAt"] / 1000) if proj.get("createdAt") else None,
                metadata={
                    "framework": proj.get("framework"),
                    "nodeVersion": proj.get("nodeVersion"),
                    "latestDeployment": proj.get("latestDeployments", [{}])[0].get("id") if proj.get("latestDeployments") else None,
                }
            ))
        
        logger.info(f"Found {len(projects)} Vercel projects")
        return projects
    
    async def list_deployments(
        self, 
        project_id: str,
        limit: int = 20,
        state: Optional[str] = None
    ) -> List[Deployment]:
        """
        List deployments for a project.
        
        Args:
            project_id: ID of the project
            limit: Maximum number of deployments
            state: Filter by state (QUEUED, BUILDING, READY, ERROR, CANCELED)
            
        Returns:
            List of Deployment objects
        """
        params = {"projectId": project_id, "limit": limit}
        if state:
            params["state"] = state
        
        data = await self._make_request("GET", "/v6/deployments", params=params)
        
        deployments = []
        for dep in data.get("deployments", []):
            deployments.append(Deployment(
                id=dep["uid"],
                name=dep.get("name", dep.get("url", "Unknown")),
                url=f"https://{dep['url']}" if dep.get("url") else None,
                state=dep.get("state", "UNKNOWN"),
                created_at=datetime.fromtimestamp(dep["created"] / 1000) if dep.get("created") else None,
                metadata={
                    "creator": dep.get("creator", {}).get("username"),
                    "source": dep.get("source"),
                    "meta": dep.get("meta", {}),
                }
            ))
        
        logger.info(f"Found {len(deployments)} deployments for project {project_id}")
        return deployments
    
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
        Fetch logs for a deployment.
        
        Args:
            project_id: Not used for Vercel (use deployment_id)
            deployment_id: ID of the deployment (required)
            start_time: Start of time range
            end_time: End of time range  
            query: Not used for Vercel
            limit: Maximum logs to return
            
        Returns:
            List of LogEntry objects
        """
        if not deployment_id:
            raise IntegrationError(
                "deployment_id is required for fetching Vercel logs",
                self.SERVICE_NAME
            )
        
        # Vercel's events endpoint returns build and runtime logs
        data = await self._make_request(
            "GET",
            f"/v2/deployments/{deployment_id}/events",
            params={"limit": limit}
        )
        
        logs = []
        for event in data if isinstance(data, list) else data.get("events", []):
            # Parse timestamp
            ts = event.get("created") or event.get("date")
            if ts:
                if isinstance(ts, (int, float)):
                    timestamp = datetime.fromtimestamp(ts / 1000 if ts > 1e10 else ts)
                else:
                    timestamp = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            else:
                timestamp = datetime.now()
            
            # Apply time filters
            if start_time and timestamp < start_time:
                continue
            if end_time and timestamp > end_time:
                continue
            
            # Parse log level
            log_type = event.get("type", "info").lower()
            level = self.LEVEL_MAP.get(log_type, LogLevel.INFO)
            
            # Get message content
            message = event.get("text") or event.get("payload", {}).get("text") or str(event)
            
            logs.append(LogEntry(
                id=f"vercel_{deployment_id}_{event.get('id', len(logs))}",
                timestamp=timestamp,
                message=message,
                level=level,
                source=f"vercel/{deployment_id}",
                metadata={
                    "deployment_id": deployment_id,
                    "event_type": log_type,
                    "serial": event.get("serial"),
                }
            ))
        
        logger.info(f"Fetched {len(logs)} logs from Vercel deployment {deployment_id}")
        return logs
    
    async def get_build_logs(self, deployment_id: str) -> List[LogEntry]:
        """
        Get build-specific logs for a deployment.
        
        This is a convenience method that filters for build events.
        """
        logs = await self.fetch_logs(deployment_id=deployment_id)
        return [log for log in logs if "build" in log.metadata.get("event_type", "").lower()]
    
    async def get_runtime_logs(
        self, 
        deployment_id: str,
        limit: int = 100
    ) -> List[LogEntry]:
        """
        Get runtime/function logs for a deployment.
        
        Uses the project logs endpoint for runtime logs.
        """
        # First get the project ID for this deployment
        dep_data = await self._make_request(
            "GET",
            f"/v13/deployments/{deployment_id}"
        )
        
        project_id = dep_data.get("projectId")
        if not project_id:
            raise IntegrationError(
                "Could not determine project ID for deployment",
                self.SERVICE_NAME
            )
        
        # Fetch runtime logs  
        data = await self._make_request(
            "GET",
            f"/v1/projects/{project_id}/deployments/{deployment_id}/logs",
            params={"limit": limit}
        )
        
        logs = []
        for log_line in data if isinstance(data, list) else []:
            timestamp = datetime.fromtimestamp(log_line.get("timestamp", 0) / 1000)
            logs.append(LogEntry(
                id=f"vercel_runtime_{deployment_id}_{len(logs)}",
                timestamp=timestamp,
                message=log_line.get("message", ""),
                level=LogLevel.INFO,
                source=f"vercel/{deployment_id}/runtime",
                metadata={
                    "deployment_id": deployment_id,
                    "project_id": project_id,
                    "type": "runtime"
                }
            ))
        
        return logs
