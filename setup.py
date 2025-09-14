#!/usr/bin/env python3
"""
UltrasoundRAG 包安装配置

使用方法:
    pip install -e .              # 开发模式安装
    pip install .                 # 正常安装
    python setup.py develop       # 开发模式（setuptools）
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取项目根目录的README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# 读取requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="ultrasoundrag",
    version="3.0.0",
    author="UltrasoundRAG Team",
    author_email="contact@ultrasoundrag.com",
    description="UltrasoundRAG - 企业级医学超声RAG系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/UltrasoundRAG",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.900",
            "pre-commit>=2.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.17.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ultrasoundrag=ultrasoundrag.cli.main:main",
            "rag=ultrasoundrag.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ultrasoundrag": [
            "config/*.yaml",
            "config/*.json", 
            "*.md",
        ],
    },
    zip_safe=False,
    keywords="rag, ultrasound, medical, ai, retrieval, multimodal",
    project_urls={
        "Bug Reports": "https://github.com/your-org/UltrasoundRAG/issues",
        "Source": "https://github.com/your-org/UltrasoundRAG",
        "Documentation": "https://ultrasoundrag.readthedocs.io/",
    },
)
