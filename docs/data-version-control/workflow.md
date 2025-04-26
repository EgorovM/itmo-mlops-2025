# DVC Workflow

## Basic Commands

### Initialize DVC

Start using DVC in your project:
```bash
dvc init
git commit -m "Initialize DVC"
```

### Add Data

Track files or directories:
```bash
# Add raw data
dvc add data/raw/dataset.csv

# Add processed data
dvc add data/processed/train.csv
dvc add data/processed/test.csv

# Add models
dvc add models/trained/model.pkl
```

### Push/Pull Data

Share data with team:
```bash
# Configure remote storage
dvc remote add -d storage s3://mybucket/dvcstore

# Push to remote
dvc push

# Pull from remote
dvc pull
```

## Pipeline Management

### Define Pipeline

Create `dvc.yaml`:
```yaml
stages:
  process:
    cmd: python src/mlops/data/process.py
    deps:
      - data/raw/dataset.csv
      - src/mlops/data/process.py
    outs:
      - data/processed/train.csv
      - data/processed/test.csv
    
  train:
    cmd: python src/mlops/models/train.py
    deps:
      - data/processed/train.csv
      - src/mlops/models/train.py
    params:
      - model.learning_rate
      - model.max_depth
    outs:
      - models/trained/model.pkl
    metrics:
      - metrics.json:
          cache: false

  evaluate:
    cmd: python src/mlops/evaluation/evaluate.py
    deps:
      - data/processed/test.csv
      - models/trained/model.pkl
      - src/mlops/evaluation/evaluate.py
    metrics:
      - evaluation.json:
          cache: false
```

### Run Pipeline

Execute pipeline stages:
```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro train

# Show pipeline structure
dvc dag
```

### Track Parameters

Create `params.yaml`:
```yaml
model:
  learning_rate: 0.1
  max_depth: 6
  n_estimators: 100
```

### Compare Experiments

```bash
# Show metrics
dvc metrics show

# Compare experiments
dvc exp diff

# List experiments
dvc exp list
```

## Git Integration

### Commit Changes

```bash
# Add data
dvc add data/raw/dataset.csv
git add data/raw/dataset.csv.dvc

# Commit
git commit -m "Add raw dataset"
```

### Branch Workflow

```bash
# Create feature branch
git checkout -b feature/new-model

# Make changes
dvc add models/trained/new_model.pkl
git add models/trained/new_model.pkl.dvc
git commit -m "Add new model"

# Push changes
git push origin feature/new-model
dvc push
```

## Best Practices

1. **Version Control**
   - Always commit `.dvc` files to Git
   - Use meaningful commit messages
   - Keep data and code changes atomic

2. **Pipeline Organization**
   - Create modular pipeline stages
   - Use clear stage names
   - Document dependencies

3. **Experiment Tracking**
   - Use consistent metric names
   - Track all relevant parameters
   - Document experiment results

4. **Team Collaboration**
   - Push data changes regularly
   - Document remote storage setup
   - Share pipeline configurations 