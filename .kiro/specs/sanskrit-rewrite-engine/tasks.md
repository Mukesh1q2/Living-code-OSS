# Implementation Plan

## Overview

This implementation plan refactors and enhances the Sanskrit Rewrite Engine through an incremental approach. Starting from the current basic implementation, we'll establish proper project structure, consolidate APIs, implement a minimal but functional transformation engine, and create a foundation for future sophisticated Sanskrit processing capabilities.

## Implementation Strategy

The plan is organized into three phases that build upon each other:

**Phase 1 - Foundation**: Establish proper project structure and consolidate existing code
**Phase 2 - Core Engine**: Implement basic transformation engine with rule-based processing  
**Phase 3 - Enhancement**: Add advanced features and prepare for future evolution

---

## Phase 1 — Foundation & Structure

- [x] 1. Create proper Python package structure

  - Update `pyproject.toml` to use src layout with correct package discovery
  - Fix console_scripts entry points to point to actual module functions
  - Create placeholder modules: `engine.py`, `tokenizer.py`, `rules.py`, `server.py`, `cli.py`
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 2. Consolidate server implementations into single FastAPI app

  - Analyze existing servers (`simple_server.py`, `robust_server.py`, `start_server.py`)
  - Create unified FastAPI application in `src/sanskrit_rewrite_engine/server.py`
  - Implement consistent endpoint definitions with proper request/response models
  - Add structured error handling with appropriate HTTP status codes
  - Include automatic OpenAPI documentation generation
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 3. Right-size dependencies and create optional extras

  - Audit current dependencies in `requirements.txt` and `pyproject.toml`
  - Move heavy/unused libraries (torch, transformers, etc.) to optional extras
  - Keep only essential dependencies for core functionality
  - Create `[dev]`, `[web]`, and `[gpu]` extras in pyproject.toml
  - Update installation documentation for different use cases
  - _Requirements: 3.1, 3.2, 3.3, 3.4_

- [x] 4. Organize tests for proper pytest discovery

  - Create `tests/` directory structure with unit and integration subdirectories
  - Move existing tests from root to appropriate test directories
  - Fix `pyproject.toml` testpaths configuration to point to `tests/`
  - Create test fixtures for common test data and mock objects
  - Add pytest configuration for test discovery and reporting
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 5. Implement security and CORS improvements

  - Replace wildcard CORS origins with specific allowed origins
  - Add request payload validation using Pydantic models
  - Implement content length limits and request size restrictions
  - Add structured logging with request IDs and timing information
  - Create middleware for security headers and request monitoring
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 6. Set up code quality tools and CI/CD

  - Create `.pre-commit-config.yaml` with black, isort, flake8, mypy
  - Add `.gitignore` to exclude build artifacts, node_modules, cache files
  - Create GitHub Actions workflow for automated testing and linting
  - Configure mypy, flake8, and black settings in pyproject.toml
  - Add pre-commit hooks installation to development setup
  - _Requirements: 6.1, 6.2, 6.3, 6.4_

## Phase 2 — Core Engine Implementation

- [x] 7. Implement basic Sanskrit transformation engine

  - Create `SanskritRewriteEngine` class with process() method
  - Implement basic tokenization for Sanskrit text processing
  - Add rule-based transformation pipeline with ordered rule application
  - Create transformation result objects with trace information
  - Add configuration management for engine settings
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 8. Create JSON-based rule definition system

  - Design JSON schema for rule definitions with pattern/replacement pairs
  - Implement `RuleRegistry` class for loading and managing rules
  - Create basic rule matching and application logic
  - Add rule priority ordering and conflict resolution
  - Create sample rule files for basic Sanskrit transformations
  - _Requirements: 7.4, 9.1, 9.2, 9.3_

- [x] 9. Build CLI interface with click

  - Create command-line interface in `src/sanskrit_rewrite_engine/cli.py`
  - Implement `process` command for text transformation
  - Add `serve` command to start the web server
  - Include options for rule selection, tracing, and configuration
  - Create help documentation and usage examples
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [x] 10. Implement comprehensive API endpoints

  - Create `/process` endpoint with request/response validation
  - Add `/health` endpoint for service monitoring
  - Implement `/rules` endpoint to list available rules
  - Create structured error responses with helpful messages
  - Add OpenAPI documentation with examples and schemas
  - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [x] 11. Create comprehensive test suite

  - Write unit tests for engine, tokenizer, and rule system components
  - Create integration tests for API endpoints using TestClient
  - Add CLI testing using subprocess and mock inputs
  - Create test fixtures with sample Sanskrit texts and expected outputs
  - Add performance tests for processing speed and memory usage
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 12. Add basic tokenization and text processing

  - Implement `BasicSanskritTokenizer` for text segmentation
  - Add Sanskrit character recognition and marker preservation
  - Create token objects with position and metadata tracking
  - Implement basic compound word detection and handling
  - Add support for morphological markers (+, _, :)
  - _Requirements: 7.1, 7.2, 9.4_

## Phase 3 — Enhancement & Future Preparation

- [x] 13. Enhance tokenization with linguistic awareness

  - Improve Sanskrit character recognition with compound vowel handling
  - Add morphological boundary detection for better segmentation
  - Implement context-aware tokenization for ambiguous cases
  - Create token metadata for grammatical and phonological information
  - Add support for multiple Sanskrit input formats (IAST, Devanagari)
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [x] 14. Implement advanced rule application logic

  - Add rule application guards to prevent infinite loops
  - Implement iterative processing until convergence
  - Create rule conflict resolution and priority handling
  - Add conditional rule activation based on context
  - Implement rule application tracing and debugging features
  - _Requirements: 7.3, 8.1, 8.2, 9.1_

- [x] 15. Create performance optimizations

  - Add caching for frequently used transformations
  - Implement lazy evaluation for large text processing
  - Create rule indexing for faster pattern matching
  - Add memory usage monitoring and optimization
  - Implement configurable processing limits and timeouts
  - _Requirements: 9.4, 8.4_

- [x] 16. Prepare architecture for future enhancements

  - Design interfaces to support future token-based processing
  - Add metadata support for future Pāṇini sūtra references
  - Create extensible rule format for complex linguistic rules
  - Implement plugin architecture for future linguistic components
  - Add hooks for future machine learning integration
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [x] 17. Improve frontend development workflow

  - Clean up frontend build artifacts from version control
  - Create proper .gitignore for frontend and Python artifacts
  - Document frontend development setup and build process
  - Fix frontend proxy configuration for backend communication
  - Add frontend build automation and deployment scripts
  - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [x] 18. Create comprehensive documentation

  - Write developer documentation with setup and contribution guidelines
  - Create API documentation with examples and use cases
  - Add user guides for different personas (developers, linguists, researchers)
  - Document the migration path from current to future architecture
  - Create troubleshooting guides and FAQ sections
  - _Requirements: 8.2, 8.4, 9.4_

---

## Implementation Notes

### Task Dependencies

**Phase 1** tasks can be executed in parallel after task 1 (package structure) is complete.

**Phase 2** requires Phase 1 completion, with tasks 7-8 as prerequisites for tasks 9-12.

**Phase 3** builds on Phase 2 and can be executed incrementally based on priorities.

### Success Criteria

Each task should result in:

- Working, testable code that can be executed immediately
- Proper test coverage for new functionality  
- Updated documentation where applicable
- No breaking changes to existing functionality

### Future Evolution

This implementation creates a solid foundation that can evolve toward:

- Sophisticated token-based processing with linguistic metadata
- Pāṇini sūtra encoding and rule hierarchies
- Integration with machine learning components for advanced Sanskrit processing
- Full computational linguistics capabilities as originally envisioned

The incremental approach ensures each phase delivers immediate value while building toward the ambitious long-term vision.
