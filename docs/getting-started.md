# Getting Started

This guide will help you get up and running with the MLOps project.

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.11 or higher
- Poetry (Python package manager)
- Docker (for containerization)
- Git

## Installation

1. Clone the repository:
```bash
git clone https://github.com/EgorovM/itmo-mlops-2025.git
cd itmo-mlops-2025
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Set up pre-commit hooks:
```bash
poetry run pre-commit install
```

## Running Tests

Run the tests using pytest:
```bash
poetry run pytest
```

## Building Documentation

To build and serve the documentation locally:
```bash
poetry run mkdocs serve
```

Then visit http://127.0.0.1:8000 to view the documentation.

## Using Docker

Build and run the project using Docker:

```bash
docker build -t mlops-project .
docker run mlops-project
```

Or use docker-compose:

```bash
docker-compose up
```

## Development Workflow

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and commit them:
```bash
git add .
git commit -m "feat: your feature description"
```

3. Push your changes and create a pull request:
```bash
git push origin feature/your-feature-name
```

4. Wait for CI checks to pass and review 