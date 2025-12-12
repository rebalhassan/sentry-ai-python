# sentry/core/security.py
"""
Security utilities for input sanitization and protection
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class QuerySanitizer:
    """
    Sanitizes user input to prevent injection attacks and ensure safe processing
    
    Protects against:
    - XSS attempts
    - Template injection
    - Excessive input length
    - Control characters
    """
    
    # Maximum allowed query length
    MAX_QUERY_LENGTH = 10000
    
    # Patterns that could indicate injection attempts
    DANGEROUS_PATTERNS = [
        (r'<script.*?>.*?</script>', 'script tag'),
        (r'{{.*?}}', 'template injection'),
        (r'\${.*?}', 'variable injection'),
        (r'<%.*?%>', 'template tag'),
    ]
    
    # Control characters to remove (except newline and tab)
    CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
    
    def __init__(self, max_length: int = MAX_QUERY_LENGTH):
        self.max_length = max_length
        self._patterns = [(re.compile(p, re.IGNORECASE | re.DOTALL), name) 
                          for p, name in self.DANGEROUS_PATTERNS]
    
    def sanitize(self, query: str) -> str:
        """
        Sanitize a user query for safe processing
        
        Args:
            query: Raw user input
            
        Returns:
            Sanitized query string
        """
        if not query:
            return ""
        
        original_length = len(query)
        
        # Remove control characters
        query = self.CONTROL_CHARS.sub('', query)
        
        # Strip whitespace
        query = query.strip()
        
        # Check for dangerous patterns
        for pattern, pattern_name in self._patterns:
            if pattern.search(query):
                logger.warning(f"Potentially dangerous pattern detected: {pattern_name}")
                query = pattern.sub('[REMOVED]', query)
        
        # Truncate if too long
        if len(query) > self.max_length:
            logger.warning(f"Query truncated from {original_length} to {self.max_length} chars")
            query = query[:self.max_length]
        
        return query
    
    def is_safe(self, query: str) -> bool:
        """
        Check if a query is safe without modifying it
        
        Args:
            query: Query to check
            
        Returns:
            True if query appears safe
        """
        if not query or len(query) > self.max_length:
            return False
        
        for pattern, _ in self._patterns:
            if pattern.search(query):
                return False
        
        return True


# Singleton instance
_sanitizer = None


def get_sanitizer() -> QuerySanitizer:
    """Get the global query sanitizer instance"""
    global _sanitizer
    if _sanitizer is None:
        _sanitizer = QuerySanitizer()
    return _sanitizer


def sanitize_query(query: str) -> str:
    """Convenience function for quick sanitization"""
    return get_sanitizer().sanitize(query)
