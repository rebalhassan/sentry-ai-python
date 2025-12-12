# sentry/core/credentials.py
"""
Secure credential storage for external service API keys.

Security Model:
- Uses Fernet symmetric encryption (AES-128-CBC with HMAC)
- Encryption key derived from machine-specific identifier + salt using PBKDF2
- Salt is randomly generated on first use and stored alongside credentials
- Credentials file permissions set to owner-only (0o600)

This approach ensures:
1. API keys are encrypted at rest - not stored in plaintext
2. Only this machine can decrypt the credentials (machine-bound)
3. No password to remember - automatic but secure
4. Even if the file is copied, it cannot be decrypted on another machine
"""

import json
import os
import logging
import platform
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from base64 import urlsafe_b64encode

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from .config import settings

logger = logging.getLogger(__name__)


class CredentialManager:
    """
    Manages encrypted storage of API credentials for external services.
    
    Usage:
        >>> cm = CredentialManager()
        >>> cm.store_credentials("vercel", {"api_key": "xxx", "team_id": "team_123"})
        >>> creds = cm.get_credentials("vercel")
        >>> print(creds["api_key"])
        xxx
    
    Supported services:
        - vercel: api_key, team_id (optional)
        - posthog: api_key, project_id, region
        - datadog: api_key, app_key, region
    """
    
    # File to store encrypted credentials
    CREDENTIALS_FILE = ".credentials"
    SALT_FILE = ".credentials_salt"
    
    # Supported services and their required/optional fields
    SERVICE_SCHEMAS = {
        "vercel": {
            "required": ["api_key"],
            "optional": ["team_id"]
        },
        "posthog": {
            "required": ["api_key", "project_id"],
            "optional": ["region"]  # us, eu
        },
        "datadog": {
            "required": ["api_key", "app_key"],
            "optional": ["region", "site"]  # us1, us3, eu1, etc.
        }
    }
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize the credential manager.
        
        Args:
            data_dir: Directory to store credentials. Defaults to ~/.sentry-ai/
        """
        self.data_dir = Path(data_dir) if data_dir else settings.data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self._credentials_path = self.data_dir / self.CREDENTIALS_FILE
        self._salt_path = self.data_dir / self.SALT_FILE
        
        self._fernet: Optional[Fernet] = None
        self._credentials_cache: Optional[Dict[str, Any]] = None
    
    def _get_machine_id(self) -> str:
        """
        Get a machine-specific identifier.
        
        Uses a combination of:
        - Machine name
        - Processor info
        - Platform details
        
        This makes the encryption key machine-bound.
        """
        components = [
            platform.node(),  # Machine/hostname
            platform.machine(),  # CPU architecture
            platform.processor(),  # Processor name
            platform.system(),  # OS name
        ]
        
        # On Windows, try to get the machine GUID
        if platform.system() == "Windows":
            try:
                import winreg
                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SOFTWARE\Microsoft\Cryptography"
                )
                machine_guid, _ = winreg.QueryValueEx(key, "MachineGuid")
                components.append(machine_guid)
                winreg.CloseKey(key)
            except Exception:
                pass  # Fallback to other identifiers
        
        # Create a stable hash of all components
        machine_string = "|".join(str(c) for c in components if c)
        return hashlib.sha256(machine_string.encode()).hexdigest()
    
    def _get_or_create_salt(self) -> bytes:
        """Get existing salt or create a new one."""
        if self._salt_path.exists():
            return self._salt_path.read_bytes()
        
        # Generate new random salt
        salt = os.urandom(32)
        self._salt_path.write_bytes(salt)
        
        # Set secure permissions
        try:
            os.chmod(self._salt_path, 0o600)
        except Exception:
            pass  # Windows may not support this
        
        logger.info("Generated new encryption salt")
        return salt
    
    def _get_fernet(self) -> Fernet:
        """Get or create the Fernet encryption instance."""
        if self._fernet is None:
            machine_id = self._get_machine_id()
            salt = self._get_or_create_salt()
            
            # Derive key using PBKDF2
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100_000,
            )
            
            key = urlsafe_b64encode(kdf.derive(machine_id.encode()))
            self._fernet = Fernet(key)
        
        return self._fernet
    
    def _load_credentials(self) -> Dict[str, Any]:
        """Load and decrypt credentials from file."""
        if self._credentials_cache is not None:
            return self._credentials_cache
        
        if not self._credentials_path.exists():
            self._credentials_cache = {}
            return self._credentials_cache
        
        try:
            encrypted_data = self._credentials_path.read_bytes()
            fernet = self._get_fernet()
            decrypted_data = fernet.decrypt(encrypted_data)
            self._credentials_cache = json.loads(decrypted_data.decode())
            return self._credentials_cache
        except InvalidToken:
            logger.error(
                "Failed to decrypt credentials. File may be corrupted or "
                "from a different machine."
            )
            raise ValueError(
                "Cannot decrypt credentials. The credentials file may be "
                "corrupted or was created on a different machine."
            )
        except Exception as e:
            logger.error(f"Error loading credentials: {e}")
            raise
    
    def _save_credentials(self, credentials: Dict[str, Any]) -> None:
        """Encrypt and save credentials to file."""
        try:
            fernet = self._get_fernet()
            data = json.dumps(credentials, indent=2)
            encrypted_data = fernet.encrypt(data.encode())
            
            self._credentials_path.write_bytes(encrypted_data)
            
            # Set secure permissions
            try:
                os.chmod(self._credentials_path, 0o600)
            except Exception:
                pass  # Windows may not support this
            
            # Update cache
            self._credentials_cache = credentials
            
            logger.info("Credentials saved successfully")
        except Exception as e:
            logger.error(f"Error saving credentials: {e}")
            raise
    
    def store_credentials(
        self, 
        service: str, 
        credentials: Dict[str, str]
    ) -> None:
        """
        Store credentials for a service.
        
        Args:
            service: Service name (vercel, posthog, datadog)
            credentials: Dictionary of credential fields
            
        Raises:
            ValueError: If service is unknown or required fields missing
        """
        service = service.lower()
        
        if service not in self.SERVICE_SCHEMAS:
            raise ValueError(
                f"Unknown service: {service}. "
                f"Supported: {list(self.SERVICE_SCHEMAS.keys())}"
            )
        
        schema = self.SERVICE_SCHEMAS[service]
        
        # Validate required fields
        missing = [f for f in schema["required"] if f not in credentials]
        if missing:
            raise ValueError(
                f"Missing required fields for {service}: {missing}"
            )
        
        # Only keep known fields
        allowed_fields = set(schema["required"]) | set(schema.get("optional", []))
        filtered_creds = {
            k: v for k, v in credentials.items() 
            if k in allowed_fields
        }
        
        # Add metadata
        filtered_creds["_stored_at"] = datetime.now().isoformat()
        
        # Load existing, update, and save
        all_creds = self._load_credentials()
        all_creds[service] = filtered_creds
        self._save_credentials(all_creds)
        
        logger.info(f"Stored credentials for {service}")
    
    def get_credentials(self, service: str) -> Optional[Dict[str, str]]:
        """
        Get credentials for a service.
        
        Args:
            service: Service name
            
        Returns:
            Credentials dictionary or None if not found
        """
        service = service.lower()
        all_creds = self._load_credentials()
        return all_creds.get(service)
    
    def has_credentials(self, service: str) -> bool:
        """Check if credentials exist for a service."""
        return self.get_credentials(service) is not None
    
    def delete_credentials(self, service: str) -> bool:
        """
        Delete credentials for a service.
        
        Args:
            service: Service name
            
        Returns:
            True if deleted, False if not found
        """
        service = service.lower()
        all_creds = self._load_credentials()
        
        if service not in all_creds:
            return False
        
        del all_creds[service]
        self._save_credentials(all_creds)
        
        logger.info(f"Deleted credentials for {service}")
        return True
    
    def list_services(self) -> Dict[str, bool]:
        """
        List all supported services and whether credentials are configured.
        
        Returns:
            Dictionary of service -> is_configured
        """
        all_creds = self._load_credentials()
        return {
            service: service in all_creds
            for service in self.SERVICE_SCHEMAS.keys()
        }
    
    def clear_all(self) -> None:
        """Delete all stored credentials."""
        if self._credentials_path.exists():
            self._credentials_path.unlink()
        self._credentials_cache = {}
        logger.info("Cleared all credentials")


# ===== SINGLETON INSTANCE =====
_credential_manager: Optional[CredentialManager] = None


def get_credential_manager() -> CredentialManager:
    """Get the global credential manager instance."""
    global _credential_manager
    if _credential_manager is None:
        _credential_manager = CredentialManager()
    return _credential_manager
