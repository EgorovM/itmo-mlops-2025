# MLOps Project

This repository contains MLOps course homework implementations following modern DevOps practices.

## Project Structure

```
.
├── src/               # Source code
├── docs/              # Documentation
├── tests/             # Test files
├── .github/           # GitHub Actions workflows
├── Dockerfile        # Container configuration
├── docker-compose.yml # Local development setup
├── pyproject.toml    # Poetry dependency management
└── .pre-commit-config.yaml # Code quality tools
```

## Development Workflow

We follow GitHub Flow for this project:

1. **Main Branch**: The `main` branch contains production-ready code
2. **Feature Branches**: Create a new branch for each feature/fix
3. **Pull Requests**: All changes must go through PR review
4. **CI/CD**: Automated checks run on every PR and merge to main

### Branch Naming Convention

- Feature branches: `feature/description`
- Bug fixes: `fix/description`
- Documentation: `docs/description`

## Getting Started

### Prerequisites

- Python 3.8+
- Poetry
- Docker
- Git

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mlops-project.git
cd mlops-project
```

2. Install dependencies:
```bash
poetry install
```

3. Setup pre-commit hooks:
```bash
poetry run pre-commit install
```

4. Run tests:
```bash
poetry run pytest
```

### Using Docker

Build and run the project using Docker:

```bash
docker build -t mlops-project .
docker run mlops-project
```

Or use docker-compose:

```bash
docker-compose up
```

## CI/CD Pipeline

Our CI/CD pipeline includes:

1. **Linting**: Black and Flake8
2. **Testing**: PyTest
3. **Docker Build**: Building and pushing to GitHub Container Registry
4. **Documentation**: Auto-deployment to GitHub Pages

## Dependencies Management

We use Poetry for dependency management to ensure reproducible builds:

- Dependencies are specified in `pyproject.toml`
- Locked versions in `poetry.lock`
- Virtual environments are managed by Poetry

## Code Quality

We maintain code quality through:

- Pre-commit hooks
- Black code formatter
- Flake8 linter
- Type hints
- Automated testing

## Documentation

Documentation is built using MkDocs and automatically deployed to GitHub Pages on each merge to main. 