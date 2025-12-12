# sentry/integrations/__init__.py
"""
External service integrations for fetching logs from cloud platforms.

Supported integrations:
- Vercel: Deployment and runtime logs
- PostHog: Event and application logs
- DataDog: Log analytics and monitoring

Each integration provides:
- Credential management (via encrypted storage)
- Deployment/project listing
- Log fetching and conversion to LogChunk format
"""

from .base import BaseIntegration, IntegrationError, RateLimitError
from .vercel import VercelIntegration
from .posthog import PostHogIntegration
from .datadog import DataDogIntegration

__all__ = [
    "BaseIntegration",
    "IntegrationError",
    "RateLimitError",
    "VercelIntegration",
    "PostHogIntegration",
    "DataDogIntegration",
]
