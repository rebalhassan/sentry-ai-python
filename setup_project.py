#!/usr/bin/env python3
"""
Billy's Project Structure Generator
Run: python setup_project.py
"""

import os
from pathlib import Path


def create_structure():
    """Create the entire project structure"""
    
    project_root = Path("sentry-ai")
    
    # Directory structure
    dirs = [
        "sentry/core",
        "sentry/services",
        "sentry/ui",
    ]
    
    # Files to create
    files = [
        "sentry/__init__.py",
        "sentry/cli.py",
        "sentry/core/__init__.py",
        "sentry/core/config.py",
        "sentry/core/database.py",
        "sentry/core/models.py",
        "sentry/services/__init__.py",
        "sentry/services/embedding.py",
        "sentry/services/vectorstore.py",
        "sentry/services/llm.py",
        "sentry/services/rag.py",
        "sentry/services/indexer.py",
        "sentry/services/log_watcher.py",
        "sentry/services/eventviewer.py",
        "sentry/ui/__init__.py",
        "sentry/ui/chat.py",
        "sentry/ui/settings.py",
        "requirements.txt",
        "setup.py",
        "README.md",
        ".gitignore",
    ]
    
    print("🔥 Billy's Project Generator - Let's build this!")
    print(f"📁 Creating project in: {project_root.absolute()}\n")
    
    # Create project root
    project_root.mkdir(exist_ok=True)
    
    # Create directories
    for dir_path in dirs:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")
    
    # Create files
    for file_path in files:
        full_path = project_root / file_path
        full_path.touch()
        print(f"✅ Created file: {file_path}")
    
    # Add initial content to key files
    
    # requirements.txt
    (project_root / "requirements.txt").write_text(REQUIREMENTS_CONTENT)
    
    # .gitignore
    (project_root / ".gitignore").write_text(GITIGNORE_CONTENT)
    
    # setup.py
    (project_root / "setup.py").write_text(SETUP_CONTENT)
    
    # README.md
    (project_root / "README.md").write_text(README_CONTENT)
    
    # __init__.py files with version
    (project_root / "sentry" / "__init__.py").write_text('__version__ = "0.1.0"\n')
    
    print("\n🎉 Project structure created successfully!")
    print(f"\n📂 Next steps:")
    print(f"   cd {project_root}")
    print(f"   python -m venv venv")
    print(f"   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
    print(f"   pip install -r requirements.txt")
    print(f"\n🔥 Let's ship this!")


# File contents
REQUIREMENTS_CONTENT = """# Core
rich>=13.7.0
textual>=0.47.1
pydantic>=2.5.0
pydantic-settings>=2.1.0

# AI/ML
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
ollama>=0.1.6

# Data & Storage
numpy>=1.24.0
sqlite-utils>=3.35

# File & System Monitoring
watchdog>=3.0.0
pywin32>=306; sys_platform == 'win32'

# Utilities
python-dateutil>=2.8.2
click>=8.1.7
"""

GITIGNORE_CONTENT = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Sentry-AI specific
.sentry-ai/
*.db
*.faiss
*.log

# OS
.DS_Store
Thumbs.db
"""

SETUP_CONTENT = """from setuptools import setup, find_packages

setup(
    name="sentry-ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    entry_points={
        "console_scripts": [
            "sentry=sentry.cli:main",
        ],
    },
    author="Billy & Team",
    description="Local-first AI diagnostic tool for logs and system events",
    python_requires=">=3.9",
)
"""

README_CONTENT = """# 🔥 Sentry-AI

**Local-first AI diagnostic tool for logs and system events**

Built by Billy. Ships fast. Works offline.

## What It Does

- 🔍 Index all your local logs (files, folders, Windows EventViewer)
- 🤖 Query them with natural language using 100% local AI
- 🔒 Everything stays on your machine (privacy-first)
- ⚡ Find root causes in seconds, not hours

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run it
python -m sentry.cli

# Or install it
pip install -e .
sentry
```

## Requirements

- Python 3.9+
- Ollama (for local LLM)
- Windows (for EventViewer support)

## Architecture

- **Embeddings:** all-MiniLM-L6-v2 (local, 80MB)
- **Vector DB:** FAISS (in-memory, fast)
- **LLM:** Llama 3 8B via Ollama
- **Storage:** SQLite + FAISS

---

Built with 🔥 by Billy
"""


if __name__ == "__main__":
    create_structure()