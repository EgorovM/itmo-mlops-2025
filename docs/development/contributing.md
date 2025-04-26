# Contributing Guide

Thank you for considering contributing to the MLOps project! This document provides guidelines and best practices for contributing.

## Code Style

We use several tools to maintain code quality:

- **Black**: For code formatting
- **Flake8**: For code linting
- **Pre-commit hooks**: For automated checks before commits

## Development Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Adding or modifying tests
- `chore:` Maintenance tasks

Example:
```
feat: add user authentication system
```

## Pull Request Process

1. Update documentation if needed
2. Add tests for new features
3. Ensure CI passes
4. Get review from maintainers

## Setting Up Development Environment

1. Install dependencies:
```bash
poetry install
```

2. Install pre-commit hooks:
```bash
poetry run pre-commit install
```

3. Run tests:
```bash
poetry run pytest
```

## Documentation

- Update documentation for any new features
- Use docstrings for Python functions
- Keep README.md up to date 