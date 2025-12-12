# sentry/api/routes/integrations.py
"""
API routes for external service integrations.

Provides endpoints for:
- Managing encrypted API credentials
- Listing projects and deployments
- Fetching and indexing logs from Vercel, PostHog, DataDog
"""

import logging
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, status

from ..auth import verify_api_key
from ..schemas import (
    IntegrationCredentials,
    IntegrationStatus,
    IntegrationProject,
    IntegrationDeployment,
    FetchLogsRequest,
    FetchLogsResponse,
    IntegrationLogEntry,
)
from ...core.credentials import get_credential_manager
from ...integrations import (
    VercelIntegration,
    PostHogIntegration,
    DataDogIntegration,
    IntegrationError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/integrations", 
    tags=["Integrations"],
    dependencies=[Depends(verify_api_key)]
)


# ===== Helper Functions =====

def _get_integration(service: str, credentials: dict):
    """Get the appropriate integration class for a service."""
    integrations = {
        "vercel": VercelIntegration,
        "posthog": PostHogIntegration,
        "datadog": DataDogIntegration,
    }
    
    service = service.lower()
    if service not in integrations:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown service: {service}. Supported: {list(integrations.keys())}"
        )
    
    return integrations[service](credentials)


def _handle_integration_error(e: Exception, service: str):
    """Convert integration errors to HTTP exceptions."""
    if isinstance(e, RateLimitError):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded for {service}. {e.message}",
            headers={"Retry-After": str(e.retry_after)} if e.retry_after else None
        )
    elif isinstance(e, IntegrationError):
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"{service} API error: {e.message}"
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error communicating with {service}: {str(e)}"
        )


# ===== Credential Management =====

@router.get("/status", response_model=List[IntegrationStatus])
async def list_integration_status():
    """
    List all integrations and their configuration status.
    
    Returns which integrations have credentials configured.
    """
    cm = get_credential_manager()
    statuses = []
    
    for service, is_configured in cm.list_services().items():
        statuses.append(IntegrationStatus(
            service=service,
            configured=is_configured,
            valid=None,  # Would require API call to check
            last_checked=None
        ))
    
    return statuses


@router.post("/{service}/credentials", status_code=status.HTTP_201_CREATED)
async def save_credentials(service: str, credentials: IntegrationCredentials):
    """
    Save encrypted credentials for an integration.
    
    The credentials are encrypted using machine-derived keys and stored locally.
    They can only be decrypted on this machine.
    
    **Security Note**: Your API keys are encrypted at rest using AES-128 encryption.
    The encryption key is derived from your machine's unique identifier, so credentials
    cannot be transferred to or decrypted on other machines.
    """
    cm = get_credential_manager()
    
    try:
        # Convert to dict, excluding None values
        creds_dict = {k: v for k, v in credentials.model_dump().items() if v is not None}
        cm.store_credentials(service, creds_dict)
        
        logger.info(f"Credentials saved for {service}")
        return {"message": f"Credentials for {service} saved successfully"}
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.delete("/{service}/credentials", status_code=status.HTTP_204_NO_CONTENT)
async def delete_credentials(service: str):
    """
    Delete stored credentials for an integration.
    """
    cm = get_credential_manager()
    
    if not cm.delete_credentials(service):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No credentials found for {service}"
        )


@router.get("/{service}/status", response_model=IntegrationStatus)
async def check_integration_status(service: str):
    """
    Check if credentials are configured and valid for an integration.
    
    This makes a test API call to verify the credentials work.
    """
    cm = get_credential_manager()
    
    credentials = cm.get_credentials(service)
    if not credentials:
        return IntegrationStatus(
            service=service,
            configured=False,
            valid=False,
            last_checked=datetime.now()
        )
    
    try:
        integration = _get_integration(service, credentials)
        is_valid = await integration.validate_credentials()
        
        return IntegrationStatus(
            service=service,
            configured=True,
            valid=is_valid,
            last_checked=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error validating {service} credentials: {e}")
        return IntegrationStatus(
            service=service,
            configured=True,
            valid=False,
            last_checked=datetime.now()
        )


# ===== Project/Deployment Listing =====

@router.get("/{service}/projects", response_model=List[IntegrationProject])
async def list_projects(service: str):
    """
    List projects/accounts available in the integration.
    
    - **Vercel**: Lists Vercel projects
    - **PostHog**: Lists PostHog projects
    - **DataDog**: Lists log indexes (DataDog organizes by service, not projects)
    """
    cm = get_credential_manager()
    
    credentials = cm.get_credentials(service)
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No credentials configured for {service}. Use POST /{service}/credentials first."
        )
    
    try:
        integration = _get_integration(service, credentials)
        projects = await integration.list_projects()
        
        return [
            IntegrationProject(
                id=p.id,
                name=p.name,
                url=p.url,
                created_at=p.created_at,
                metadata=p.metadata
            )
            for p in projects
        ]
        
    except Exception as e:
        _handle_integration_error(e, service)


@router.get("/{service}/projects/{project_id}/deployments", response_model=List[IntegrationDeployment])
async def list_deployments(service: str, project_id: str, limit: int = 20):
    """
    List deployments for a project.
    
    - **Vercel**: Lists actual deployments (READY, BUILDING, ERROR, etc.)
    - **PostHog/DataDog**: Returns time periods (last_1h, last_24h, etc.) 
      since they don't have deployment concepts
    """
    cm = get_credential_manager()
    
    credentials = cm.get_credentials(service)
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No credentials configured for {service}"
        )
    
    try:
        integration = _get_integration(service, credentials)
        deployments = await integration.list_deployments(project_id, limit=limit)
        
        return [
            IntegrationDeployment(
                id=d.id,
                name=d.name,
                url=d.url,
                state=d.state,
                created_at=d.created_at,
                metadata=d.metadata
            )
            for d in deployments
        ]
        
    except Exception as e:
        _handle_integration_error(e, service)


# ===== Log Fetching =====

@router.post("/{service}/logs", response_model=FetchLogsResponse)
async def fetch_logs(service: str, request: FetchLogsRequest):
    """
    Fetch logs from an integration.
    
    **Vercel**: Requires `deployment_id`. Returns build and runtime logs.
    
    **PostHog**: Uses `deployment_id` as time period (last_1h, last_24h, etc.)
    or `start_time`/`end_time`. Supports HogQL in `query`.
    
    **DataDog**: Uses `deployment_id` as time period or time range.
    Supports DataDog query syntax (e.g., "service:my-app status:error").
    """
    cm = get_credential_manager()
    
    credentials = cm.get_credentials(service)
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No credentials configured for {service}"
        )
    
    try:
        integration = _get_integration(service, credentials)
        
        logs = await integration.fetch_logs(
            project_id=request.project_id,
            deployment_id=request.deployment_id,
            start_time=request.start_time,
            end_time=request.end_time,
            query=request.query,
            limit=request.limit
        )
        
        return FetchLogsResponse(
            service=service,
            logs=[
                IntegrationLogEntry(
                    id=log.id,
                    timestamp=log.timestamp,
                    message=log.message,
                    level=log.level.value,
                    source=log.source,
                    metadata=log.metadata
                )
                for log in logs
            ],
            total=len(logs),
            fetched_at=datetime.now()
        )
        
    except Exception as e:
        _handle_integration_error(e, service)


@router.post("/{service}/logs/index")
async def fetch_and_index_logs(service: str, request: FetchLogsRequest):
    """
    Fetch logs from an integration and index them in the RAG system.
    
    This is a convenience endpoint that:
    1. Fetches logs from the external service
    2. Converts them to LogChunks
    3. Indexes them for RAG queries
    
    Use this to make external logs searchable alongside local logs.
    """
    from ...services.rag import get_rag_service
    from ...core.models import LogChunk, LogLevel as InternalLogLevel
    
    cm = get_credential_manager()
    
    credentials = cm.get_credentials(service)
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"No credentials configured for {service}"
        )
    
    try:
        integration = _get_integration(service, credentials)
        
        logs = await integration.fetch_logs(
            project_id=request.project_id,
            deployment_id=request.deployment_id,
            start_time=request.start_time,
            end_time=request.end_time,
            query=request.query,
            limit=request.limit
        )
        
        if not logs:
            return {
                "message": "No logs found to index",
                "indexed": 0
            }
        
        # Map external log levels to internal
        level_map = {
            "debug": InternalLogLevel.DEBUG,
            "info": InternalLogLevel.INFO,
            "warning": InternalLogLevel.WARNING,
            "error": InternalLogLevel.ERROR,
            "critical": InternalLogLevel.ERROR,
        }
        
        # Convert to LogChunks
        chunks = []
        source_id = f"integration_{service}_{request.deployment_id or request.project_id or 'default'}"
        
        for log in logs:
            chunk = LogChunk(
                id=log.id,
                source_id=source_id,
                content=log.to_content(),
                timestamp=log.timestamp,
                log_level=level_map.get(log.level.value, InternalLogLevel.INFO),
                metadata={
                    "integration": service,
                    "original_source": log.source,
                    **log.metadata
                }
            )
            chunks.append(chunk)
        
        # Index the chunks
        rag = get_rag_service()
        indexed_count = rag.index_chunks_batch(chunks)
        
        logger.info(f"Indexed {indexed_count} logs from {service}")
        
        return {
            "message": f"Successfully indexed {indexed_count} logs from {service}",
            "indexed": indexed_count,
            "source_id": source_id
        }
        
    except Exception as e:
        _handle_integration_error(e, service)
