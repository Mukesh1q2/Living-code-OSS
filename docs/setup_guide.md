# Sanskrit Rewrite Engine - Setup and Installation Guide

## Overview

This comprehensive setup guide covers installation, configuration, and initial setup of the Sanskrit Rewrite Engine for different use cases and environments.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Development Setup](#development-setup)
4. [Configuration](#configuration)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)
7. [Environment-Specific Setup](#environment-specific-setup)
8. [Performance Optimization](#performance-optimization)

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)
- **Python**: Version 3.8 or higher
- **Memory**: 2GB RAM available
- **Storage**: 200MB free space
- **Network**: Internet connection for installation

### Recommended Requirements
- **Python**: Version 3.9 or higher
- **Memory**: 4GB+ RAM for large text processing
- **Storage**: 1GB+ free space for caches and models
- **CPU**: Multi-core processor for better performance

### Optional Requirements
- **GPU**: NVIDIA GPU with CUDA for ML features (optional)
- **Docker**: For containerized deployment
- **Node.js**: For frontend development (if using web interface)

## Installation Methods

### Method 1: PyPI Installation (Recommended)

#### Basic Installation
```bash
# Install core package
pip install sanskrit-rewrite-engine

# Verify installation
python -c "import sanskrit_rewrite_engine; print('Installation successful!')"
```

#### Installation with Extras
```bash
# For development work
pip install sanskrit-rewrite-engine[dev]

# For web interface
pip install sanskrit-rewrite-engine[web]

# For GPU acceleration
pip install sanskrit-rewrite-engine[gpu]

# For all features
pip install sanskrit-rewrite-engine[all]
```

### Method 2: Source Installation

#### Clone and Install
```bash
# Clone repository
git clone https://github.com/your-org/sanskrit-rewrite-engine.git
cd sanskrit-rewrite-engine

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e .[dev]
```

### Method 3: Docker Installation

#### Using Docker Hub
```bash
# Pull and run container
docker pull sanskrit-rewrite-engine:latest
docker run -p 8000:8000 sanskrit-rewrite-engine:latest
```

#### Build from Source
```bash
# Clone repository
git clone https://github.com/your-org/sanskrit-rewrite-engine.git
cd sanskrit-rewrite-engine

# Build Docker image
docker build -t sanskrit-rewrite-engine .

# Run container
docker run -p 8000:8000 sanskrit-rewrite-engine
```

## Development Setup

### Complete Development Environment

#### 1. Clone and Setup
```bash
# Clone with all branches
git clone --recurse-submodules https://github.com/your-org/sanskrit-rewrite-engine.git
cd sanskrit-rewrite-engine

# Create development virtual environment
python -m venv dev-env
source dev-env/bin/activate  # On Windows: dev-env\Scripts\activate

# Install all development dependencies
pip install -e .[dev-all]
```

#### 2. Install Development Tools
```bash
# Install pre-commit hooks
pre-commit install

# Install additional development tools
pip install jupyter notebook ipython

# Verify development setup
python -m pytest tests/ -v
```

#### 3. IDE Configuration

##### VS Code Setup
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./dev-env/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

##### PyCharm Setup
1. Open project directory
2. Configure Python interpreter to use virtual environment
3. Enable code inspections for flake8, mypy
4. Configure Black as code formatter
5. Set up run configurations for tests and server

### Frontend Development (Optional)

#### Setup React Frontend
```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Start development server
npm start

# Build for production
npm run build
```

## Configuration

### Basic Configuration

#### Engine Configuration
```python
# config.py or your application
from sanskrit_rewrite_engine.config import EngineConfig

config = EngineConfig(
    max_passes=20,
    enable_tracing=True,
    performance_mode=False,
    rule_directories=["data/rules"],
    default_rule_set="basic_sandhi"
)
```

#### Server Configuration
```python
# server_config.py
from sanskrit_rewrite_engine.server import create_app

app = create_app(
    host="0.0.0.0",
    port=8000,
    debug=False,
    cors_origins=["http://localhost:3000"]
)
```

### Environment Variables

#### Core Settings
```bash
# .env file
SANSKRIT_ENGINE_DEBUG=false
SANSKRIT_ENGINE_LOG_LEVEL=INFO
SANSKRIT_ENGINE_MAX_PASSES=20
SANSKRIT_ENGINE_ENABLE_TRACING=true
SANSKRIT_ENGINE_PERFORMANCE_MODE=false
```

#### Server Settings
```bash
# Server configuration
SANSKRIT_SERVER_HOST=0.0.0.0
SANSKRIT_SERVER_PORT=8000
SANSKRIT_SERVER_WORKERS=4
SANSKRIT_SERVER_CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

#### Development Settings
```bash
# Development environment
SANSKRIT_DEV_MODE=true
SANSKRIT_DEV_RELOAD=true
SANSKRIT_DEV_LOG_LEVEL=DEBUG
```

### Configuration Files

#### YAML Configuration
```yaml
# config.yaml
engine:
  max_passes: 20
  enable_tracing: true
  performance_mode: false
  rule_directories:
    - "data/rules"
    - "custom_rules"
  
server:
  host: "0.0.0.0"
  port: 8000
  cors_origins:
    - "http://localhost:3000"
    - "http://127.0.0.1:3000"
  
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

#### JSON Configuration
```json
{
  "engine": {
    "max_passes": 20,
    "enable_tracing": true,
    "performance_mode": false,
    "rule_directories": ["data/rules"]
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "cors_origins": ["http://localhost:3000"]
  }
}
```

## Verification

### Basic Verification

#### Test Installation
```python
# test_installation.py
from sanskrit_rewrite_engine import SanskritRewriteEngine

def test_basic_functionality():
    """Test basic engine functionality"""
    engine = SanskritRewriteEngine()
    result = engine.process("rāma + iti")
    
    assert result.success
    assert result.get_output_text() == "rāmeti"
    print("✓ Basic functionality working")

def test_server_import():
    """Test server import"""
    from sanskrit_rewrite_engine.server import create_app
    app = create_app()
    print("✓ Server import working")

def test_cli_import():
    """Test CLI import"""
    from sanskrit_rewrite_engine.cli import main
    print("✓ CLI import working")

if __name__ == "__main__":
    test_basic_functionality()
    test_server_import()
    test_cli_import()
    print("✅ All verification tests passed!")
```

#### Run Verification
```bash
# Run verification script
python test_installation.py

# Run unit tests
python -m pytest tests/unit/ -v

# Test CLI
sanskrit-cli --help

# Test server (in separate terminal)
sanskrit-web --port 8001
curl http://localhost:8001/health
```

### Advanced Verification

#### Performance Test
```python
# performance_test.py
import time
from sanskrit_rewrite_engine import SanskritRewriteEngine

def performance_test():
    """Test processing performance"""
    engine = SanskritRewriteEngine()
    
    # Test data
    test_texts = [
        "rāma + iti",
        "deva + indra + iti",
        "mahā + ātman + eva + ca"
    ]
    
    start_time = time.time()
    
    for text in test_texts * 100:  # Process 300 texts
        result = engine.process(text)
        assert result.success
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Processed 300 texts in {duration:.2f} seconds")
    print(f"Average: {duration/300*1000:.2f} ms per text")
    
    if duration < 5.0:  # Should process 300 texts in under 5 seconds
        print("✅ Performance test passed")
    else:
        print("⚠️ Performance may be suboptimal")

if __name__ == "__main__":
    performance_test()
```

## Troubleshooting

### Common Installation Issues

#### Issue 1: Python Version
```bash
# Error: Python version too old
ERROR: sanskrit-rewrite-engine requires Python '>=3.8'

# Solution: Update Python
# Check current version
python --version

# Install Python 3.8+ from python.org or use package manager
# On Ubuntu:
sudo apt update
sudo apt install python3.9 python3.9-venv python3.9-pip

# On macOS with Homebrew:
brew install python@3.9

# On Windows: Download from python.org
```

#### Issue 2: Dependency Conflicts
```bash
# Error: Dependency resolution failed
ERROR: pip's dependency resolver does not currently consider...

# Solution: Use virtual environment
python -m venv clean-env
source clean-env/bin/activate  # Windows: clean-env\Scripts\activate
pip install --upgrade pip
pip install sanskrit-rewrite-engine
```

#### Issue 3: Permission Errors
```bash
# Error: Permission denied
ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied

# Solution: Use user installation or virtual environment
pip install --user sanskrit-rewrite-engine
# OR
python -m venv user-env
source user-env/bin/activate
pip install sanskrit-rewrite-engine
```

#### Issue 4: Network Issues
```bash
# Error: Network timeout
ERROR: Could not fetch URL https://pypi.org/simple/sanskrit-rewrite-engine/

# Solution: Use different index or offline installation
pip install --index-url https://pypi.org/simple/ sanskrit-rewrite-engine
# OR
pip install --trusted-host pypi.org --trusted-host pypi.python.org sanskrit-rewrite-engine
```

### Runtime Issues

#### Issue 1: Import Errors
```python
# Error
ImportError: No module named 'sanskrit_rewrite_engine'

# Solution: Check installation and Python path
import sys
print(sys.path)
# Ensure package is installed in correct environment
```

#### Issue 2: Configuration Errors
```python
# Error
FileNotFoundError: Rule file not found

# Solution: Check configuration paths
from sanskrit_rewrite_engine.config import EngineConfig
config = EngineConfig()
print(f"Rule directories: {config.rule_directories}")
# Ensure rule files exist in specified directories
```

#### Issue 3: Performance Issues
```python
# Issue: Slow processing
# Solution: Enable performance mode
from sanskrit_rewrite_engine.config import EngineConfig

config = EngineConfig(
    performance_mode=True,
    enable_tracing=False,  # Disable for production
    max_passes=10  # Reduce if needed
)
```

## Environment-Specific Setup

### Production Environment

#### System Service Setup
```bash
# Create systemd service (Linux)
sudo tee /etc/systemd/system/sanskrit-engine.service > /dev/null <<EOF
[Unit]
Description=Sanskrit Rewrite Engine
After=network.target

[Service]
Type=simple
User=sanskrit
WorkingDirectory=/opt/sanskrit-engine
Environment=PATH=/opt/sanskrit-engine/venv/bin
ExecStart=/opt/sanskrit-engine/venv/bin/sanskrit-web --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable sanskrit-engine
sudo systemctl start sanskrit-engine
```

#### Nginx Configuration
```nginx
# /etc/nginx/sites-available/sanskrit-engine
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Development Environment

#### Docker Compose Setup
```yaml
# docker-compose.yml
version: '3.8'

services:
  sanskrit-engine:
    build: .
    ports:
      - "8000:8000"
    environment:
      - SANSKRIT_ENGINE_DEBUG=true
      - SANSKRIT_ENGINE_LOG_LEVEL=DEBUG
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    depends_on:
      - sanskrit-engine
```

### Testing Environment

#### CI/CD Setup (GitHub Actions)
```yaml
# .github/workflows/test.yml
name: Test Sanskrit Rewrite Engine

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]
    
    - name: Run tests
      run: |
        python -m pytest tests/ -v --cov=sanskrit_rewrite_engine
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Performance Optimization

### System-Level Optimization

#### Memory Settings
```bash
# Increase available memory for Python
export PYTHONMALLOC=malloc
export MALLOC_ARENA_MAX=2

# For large text processing
ulimit -v 4194304  # 4GB virtual memory limit
```

#### CPU Optimization
```python
# config.py
import os
from sanskrit_rewrite_engine.config import EngineConfig

# Use all available CPU cores
cpu_count = os.cpu_count()

config = EngineConfig(
    performance_mode=True,
    worker_processes=cpu_count,
    enable_parallel_processing=True
)
```

### Application-Level Optimization

#### Caching Configuration
```python
# Enable caching for better performance
from sanskrit_rewrite_engine.config import EngineConfig

config = EngineConfig(
    enable_caching=True,
    cache_size=1000,  # Cache 1000 most recent transformations
    cache_ttl=3600,   # Cache for 1 hour
)
```

#### Memory Management
```python
# Configure memory limits
config = EngineConfig(
    memory_limit_mb=1024,  # 1GB limit
    gc_threshold=1000,     # Garbage collection threshold
    enable_memory_monitoring=True
)
```

---

This setup guide provides comprehensive instructions for installing and configuring the Sanskrit Rewrite Engine in various environments. Follow the appropriate sections based on your use case and environment requirements.

*Last updated: January 15, 2024*