#!/usr/bin/env python3
"""
TERMINALIS-V.2 Setup Script
"""

from setuptools import setup, find_packages
import os
import sys

# Read version from main module
version = "2.0.0"

# Read long description from README
long_description = """
# TERMINALIS-V.2 🤖

Advanced Agentic AI Terminal System with:
- Multi-model AI support with Safetensors
- Real-time progress tracking
- Agent-based architecture
- Advanced tool integration
- Modern terminal interface

## Features
- 🧠 Multiple AI Models (BERT, GPT, LLaMA, etc.)
- 🚀 Fast Safetensors loading
- 📊 Real-time progress indicators
- 🤖 Agent management system
- 🔧 Extensible tool framework
- 🎨 Beautiful terminal UI

## Quick Install
```bash
iwr -useb https://raw.githubusercontent.com/Gokayofficialtrysolutions/TERMINALIS-V.2/main/install.ps1 | iex
```
"""

setup(
    name="terminalis-v2",
    version=version,
    description="Advanced Agentic AI Terminal System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Gokay Official",
    author_email="contact@gokayofficial.com",
    url="https://github.com/Gokayofficialtrysolutions/TERMINALIS-V.2",
    project_urls={
        "Bug Reports": "https://github.com/Gokayofficialtrysolutions/TERMINALIS-V.2/issues",
        "Source": "https://github.com/Gokayofficialtrysolutions/TERMINALIS-V.2",
        "Documentation": "https://github.com/Gokayofficialtrysolutions/TERMINALIS-V.2/wiki",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "safetensors>=0.3.0",
        "huggingface-hub>=0.16.0",
        "accelerate>=0.20.0",
        "tokenizers>=0.13.0",
        "rich>=13.0.0",
        "typer>=0.9.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
        "tqdm>=4.65.0",
        "psutil>=5.9.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "langchain>=0.0.200",
        "chromadb>=0.4.0",
        "gradio>=3.35.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "websockets>=11.0.0",
        "aiohttp>=3.8.0",
        "sentence-transformers>=2.2.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scikit-learn>=1.3.0",
        "sqlalchemy>=2.0.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "gpu": [
            "torch>=2.0.0+cu118",
            "torchvision>=0.15.0+cu118",
            "torchaudio>=2.0.0+cu118",
        ],
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
            "pytest-cov>=4.1.0",
            "pre-commit>=3.3.0",
        ],
        "all": [
            "streamlit>=1.24.0",
            "plotly>=5.15.0",
            "jupyter>=1.0.0",
            "ipywidgets>=8.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "terminalis=src.main:main",
            "terminalis-gui=src.gui:main",
            "terminalis-server=src.server:main",
        ],
    },
    include_package_data=True,
    package_data={
        "terminalis": [
            "config/*.yaml",
            "assets/*",
            "templates/*",
        ],
    },
    zip_safe=False,
    keywords="ai, terminal, agents, safetensors, transformers, nlp, machine-learning",
)
