# Model Training

## Overview

This section describes our model training pipeline. We use DVC to track model training experiments, including model parameters, metrics, and artifacts.

## Training Pipeline

Our model training pipeline includes:

1. **Model Selection**
   - Choose appropriate algorithms
   - Define model architecture
   - Set hyperparameters

2. **Training Process**
   - Load processed data
   - Train models
   - Save model artifacts
   - Track metrics

3. **Experiment Tracking**
   - Version model files
   - Track parameters
   - Record metrics
   - Compare experiments

## Pipeline Configuration

The training pipeline is defined in `dvc.yaml`:

```yaml
stages:
  train_model:
    cmd: python src/mlops/models/train.py
    deps:
      - data/processed/train.csv
      - src/mlops/models/train.py
    params:
      - model.type
      - model.learning_rate
      - model.max_depth
      - model.n_estimators
    outs:
      - models/trained/model.pkl
    metrics:
      - metrics/train_metrics.json:
          cache: false
```

## Model Parameters

Parameters are defined in `params.yaml`:

```yaml
model:
  type: "random_forest"
  learning_rate: 0.1
  max_depth: 6
  n_estimators: 100
```

## Usage

To train models:

```bash
# Run training pipeline
dvc repro train_model

# Check metrics
dvc metrics show

# Compare experiments
dvc exp diff

# Push model to remote storage
dvc push
```

## Experiment Management

Track different model versions:

```bash
# Create new experiment
dvc exp run -n "experiment_name" \
    -S model.learning_rate=0.2 \
    -S model.max_depth=8

# List experiments
dvc exp list

# Apply best experiment
dvc exp apply "experiment_name"

# Save model version
dvc add models/trained/model.pkl
git add models/trained/model.pkl.dvc
git commit -m "feat: add new model version"
``` 