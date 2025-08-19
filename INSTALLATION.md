# Installation Guide

This guide explains how to install the Sanskrit Rewrite Engine with different feature sets based on your needs.

## Quick Start (Minimal Installation)

For basic functionality with minimal dependencies:

```bash
pip install sanskrit-rewrite-engine
```

Or from source:

```bash
git clone <repository-url>
cd sanskrit-rewrite-engine
pip install -e .
```

This installs only the core dependencies (~50MB):
- FastAPI web server
- CLI interface
- Basic text processing
- Configuration management

## Installation Options

### Development Setup

For contributors and developers:

```bash
pip install -e ".[dev]"
# OR
pip install -r requirements-dev.txt
```

Includes:
- Testing framework (pytest)
- Code formatting (black, isort)
- Type checking (mypy)
- Linting (flake8)
- Pre-commit hooks

### Web Interface

For enhanced web features:

```bash
pip install -e ".[web]"
```

Adds:
- Streamlit dashboard
- File upload support
- JWT authentication
- Enhanced templating

### Scientific Analysis

For researchers and linguists:

```bash
pip install -e ".[analysis]"
```

Adds:
- NumPy, SciPy, Pandas
- Matplotlib, Seaborn, Plotly
- Scikit-learn for ML analysis
- Statistical computing tools

### GPU/Machine Learning

For advanced ML features (⚠️ Large download ~2GB):

```bash
pip install -e ".[gpu]"
```

Adds:
- PyTorch ecosystem
- Transformers library
- GPU acceleration support
- Advanced tokenization

### System Monitoring

For production deployments:

```bash
pip install -e ".[monitoring]"
```

Adds:
- System resource monitoring
- Structured logging
- Prometheus metrics
- Performance profiling

### Database Support

For persistent storage:

```bash
pip install -e ".[storage]"
```

Adds:
- SQLAlchemy ORM
- Redis caching
- MongoDB support
- Async database drivers

### Jupyter Notebooks

For interactive development:

```bash
pip install -e ".[jupyter]"
```

Adds:
- JupyterLab environment
- Interactive widgets
- Notebook support
- Visualization tools

### Performance Optimizations

For high-performance deployments:

```bash
pip install -e ".[performance]"
```

Adds:
- uvloop (faster event loop)
- orjson (faster JSON)
- Compression libraries
- Async optimizations

### All Features

For complete installation (⚠️ Very large download ~5GB):

```bash
pip install -e ".[all]"
```

Includes all optional dependencies.

## Recommended Combinations

### For Developers
```bash
pip install -e ".[dev,web,analysis]"
```

### For Researchers
```bash
pip install -e ".[analysis,jupyter,monitoring]"
```

### For Production
```bash
pip install -e ".[web,monitoring,storage,performance]"
```

### For ML Research
```bash
pip install -e ".[gpu,analysis,jupyter,monitoring]"
```

## Platform-Specific Notes

### Windows
- GPU extras require CUDA toolkit
- Some monitoring tools may not be available
- Use PowerShell or Command Prompt

### macOS
- GPU extras use CPU-only PyTorch
- All features supported
- Use Terminal or iTerm2

### Linux
- Full GPU support with NVIDIA drivers
- All monitoring features available
- Recommended for production

## Verification

After installation, verify your setup:

```bash
# Check CLI
sanskrit-cli --help

# Check web server
python -c "from sanskrit_rewrite_engine.server import app; print('✅ Server ready')"

# Run tests (if dev extras installed)
pytest tests/

# Start web server
uvicorn sanskrit_rewrite_engine.server:app --reload
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you installed the package with `-e .`
2. **Missing dependencies**: Install the appropriate extras for your use case
3. **GPU issues**: Verify CUDA installation and PyTorch compatibility
4. **Permission errors**: Use virtual environments or `--user` flag

### Getting Help

- Check the [documentation](docs/)
- Review [examples](examples/)
- Open an [issue](https://github.com/your-org/sanskrit-rewrite-engine/issues)
- Join our [discussions](https://github.com/your-org/sanskrit-rewrite-engine/discussions)

## Virtual Environment (Recommended)

Always use a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install
pip install -e ".[dev]"

# Deactivate when done
deactivate
```