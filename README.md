# MLOps Project

This repository contains MLOps course homework implementations following modern DevOps practices.

## Project Structure

```
.
├── src/               # Source code
├── docs/              # Documentation
├── tests/             # Test files
├── .github/           # GitHub Actions workflows
├── data/             # Data directory (DVC-tracked)
│   ├── raw/          # Raw data
│   ├── interim/      # Intermediate data
│   └── processed/    # Processed data
├── models/           # Model directory (DVC-tracked)
│   ├── trained/      # Trained models
│   └── evaluation/   # Model evaluation results
├── Dockerfile        # Container configuration
├── docker-compose.yml # Local development setup
├── pyproject.toml    # Poetry dependency management
├── dvc.yaml         # DVC pipeline configuration
├── params.yaml      # Model parameters
└── .pre-commit-config.yaml # Code quality tools
```

## Data Version Control

We use [DVC (Data Version Control)](https://dvc.org/) for managing our ML pipeline:

1. **Data Management**:
   - Raw data versioning
   - Processed datasets tracking
   - Model artifacts versioning

2. **ML Pipeline**:
   - Data processing pipeline
   - Model training pipeline
   - Evaluation pipeline

3. **Experiment Tracking**:
   - Parameter versioning
   - Metric tracking
   - Model comparison

### DVC Workflow

```bash
# Initialize DVC
dvc init

# Add and track data
dvc add data/raw/dataset.csv

# Run pipelines
dvc repro

# Push to remote storage
dvc push

# Pull from remote storage
dvc pull

# Compare experiments
dvc exp show
```

For detailed documentation about our ML pipeline and DVC usage, see our [documentation](https://egorovm.github.io/itmo-mlops-2025/).

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