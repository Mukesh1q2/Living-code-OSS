"""
Setup configuration for Vidya Quantum Interface
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="vidya-quantum-interface",
    version="1.0.0",
    author="Vidya Development Team",
    author_email="dev@vidya-quantum.com",
    description="Sanskrit AI Consciousness Interface - Where Ancient Wisdom Meets Quantum Computing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mukesh1q2/Living-code-OSS",
    project_urls={
        "Bug Tracker": "https://github.com/Mukesh1q2/Living-code-OSS/issues",
        "Documentation": "https://github.com/Mukesh1q2/Living-code-OSS/docs",
        "Source Code": "https://github.com/Mukesh1q2/Living-code-OSS",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Education",
        "Topic :: Religion",
    ],
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
        ],
        "cloud": [
            "boto3>=1.34.0",
            "google-cloud-storage>=2.10.0",
            "azure-storage-blob>=12.19.0",
        ],
        "monitoring": [
            "prometheus-client>=0.19.0",
            "grafana-api>=1.0.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "vidya-server=vidya_quantum_interface.cli:main",
            "vidya-analyze=vidya_quantum_interface.cli:analyze",
        ],
    },
    include_package_data=True,
    package_data={
        "vidya_quantum_interface": [
            "data/*.json",
            "data/*.yaml",
            "templates/*.html",
            "static/*",
        ],
        "sanskrit_rewrite_engine": [
            "rules/*.yaml",
            "dictionaries/*.json",
        ],
    },
    keywords=[
        "sanskrit",
        "ai",
        "quantum",
        "consciousness",
        "nlp",
        "linguistics",
        "ancient-wisdom",
        "morphology",
        "panini",
        "grammar",
        "vedic",
        "fastapi",
        "react",
        "webgl",
        "visualization",
    ],
    zip_safe=False,
)