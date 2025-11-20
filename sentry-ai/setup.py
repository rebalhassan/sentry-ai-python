from setuptools import setup, find_packages

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
