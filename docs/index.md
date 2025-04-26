# Welcome to MLOps Project

This is the documentation for the MLOps project, containing homework implementations for the MLOps course.

## Project Overview

This project demonstrates modern MLOps practices including:

- Automated testing and linting
- Containerization with Docker
- CI/CD pipeline with GitHub Actions
- Documentation with MkDocs
- Dependency management with Poetry

## Quick Start

```bash
# Clone the repository
git clone https://github.com/EgorovM/itmo-mlops-2025
cd mlops-project

# Install dependencies with Poetry
poetry install

# Run tests
poetry run pytest

# Build and run with Docker
docker-compose up
```

## Project Structure

The project follows a standard Python package structure with additional MLOps components:

```
.
├── src/               # Source code
├── docs/              # Documentation
├── tests/             # Test files
├── .github/           # GitHub Actions workflows
├── Dockerfile        # Container configuration
└── docker-compose.yml # Local development setup
```

## API Documentation

For detailed API documentation, please see the [API Reference](api/core.md) section.
