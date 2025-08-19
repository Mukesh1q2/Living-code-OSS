#!/usr/bin/env bash
set -euo pipefail
ruff check .
black --check .
mypy vidya_quantum_interface || true
pytest -q --maxfail=1 --disable-warnings --cov --cov-report=term-missing

