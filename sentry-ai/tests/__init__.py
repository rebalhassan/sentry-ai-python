# tests/__init__.py
"""
Sentry-AI Test Suite

All test files consolidated here from root and debugging_tests.
Run tests with: python -m pytest tests/
"""

import sys
from pathlib import Path

# Add parent directory to path so tests can import from sentry module
# This is needed because tests/ is a sibling of sentry/, not inside it
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))
