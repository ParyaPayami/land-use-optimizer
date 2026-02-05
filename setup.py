#!/usr/bin/env python
"""
PIMALUOS - Physics Informed Multi-Agent Land Use Optimization Software

A comprehensive platform for urban planners, developers, and decision makers
to optimize land use using Graph Neural Networks, Multi-Agent Reinforcement
Learning, and physics-based simulation.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

setup(
    name="pimaluos",
    version="0.1.0",
    author="PIMALUOS Team",
    author_email="pimaluos@example.com",
    description="Physics Informed Multi-Agent Land Use Optimization Software",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pimaluos/pimaluos",
    packages=find_packages(exclude=["tests", "tests.*", "dashboard", "docs"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        # Core dependencies
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "torch-scatter>=2.1.0",
        "torch-sparse>=0.6.0",
        # Geospatial
        "geopandas>=0.13.0",
        "shapely>=2.0.0",
        "rasterio>=1.3.0",
        "pyproj>=3.5.0",
        # ML and optimization
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        # LLM and NLP
        "langchain>=0.1.0",
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
        "openai>=1.0.0",
        "anthropic>=0.7.0",
        # API
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "websockets>=12.0",
        "pydantic>=2.0.0",
        # Visualization
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        # Utilities
        "tqdm>=4.65.0",
        "requests>=2.31.0",
        "networkx>=3.1",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.4.0",
            "mkdocstrings[python]>=0.23.0",
        ],
        "ollama": [
            "ollama>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pimaluos=pimaluos.cli:main",
            "pimaluos-server=pimaluos.api.server:run_server",
        ],
    },
    include_package_data=True,
    package_data={
        "pimaluos": [
            "config/cities/*.yaml",
        ],
    },
)
