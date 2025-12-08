# sentry/api/auth.py
"""
Authentication and security for the API
Uses auto-generated API key stored in config
"""

import secrets
import hashlib
from pathlib import Path
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from ..core.config import settings

# API key header name
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# File to store the API key (in data directory)
API_KEY_FILE = Path(settings.data_dir) / ".api_key"


def _generate_api_key() -> str:
    """Generate a secure random API key"""
    return secrets.token_urlsafe(32)


def _hash_key(key: str) -> str:
    """Hash the API key for secure storage"""
    return hashlib.sha256(key.encode()).hexdigest()


def get_or_create_api_key() -> str:
    """
    Get existing API key or generate a new one
    
    The key is stored in plaintext in a local file for the user to retrieve.
    A hash is used internally for comparison.
    
    Returns:
        The plaintext API key
    """
    # Ensure data directory exists
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    
    if API_KEY_FILE.exists():
        return API_KEY_FILE.read_text().strip()
    
    # Generate new key
    new_key = _generate_api_key()
    API_KEY_FILE.write_text(new_key)
    
    # Make file readable only by owner (Windows will ignore this on non-NTFS)
    try:
        import os
        os.chmod(API_KEY_FILE, 0o600)
    except Exception:
        pass  # Ignore permission errors on Windows
    
    return new_key


def _get_stored_key_hash() -> Optional[str]:
    """Get the hash of the stored API key"""
    if not API_KEY_FILE.exists():
        return None
    
    key = API_KEY_FILE.read_text().strip()
    return _hash_key(key)


async def verify_api_key(api_key: Optional[str] = Security(API_KEY_HEADER)) -> str:
    """
    Dependency to verify API key for protected routes
    
    Args:
        api_key: The API key from request header
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: If API key is invalid or missing
    """
    if api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Include 'X-API-Key' header."
        )
    
    # Get stored key for comparison
    stored_key = get_or_create_api_key()
    
    # Use constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(api_key, stored_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return api_key
