# Data Processing

## Overview

This section describes our data processing pipeline that prepares raw data for model training. We use DVC to track all data transformations and ensure reproducibility.

## Data Pipeline

Our data processing pipeline consists of the following stages:

1. **Raw Data Collection**
   - Data is stored in `data/raw/`
   - Each dataset is versioned with DVC
   - Raw data is considered immutable

2. **Data Cleaning**
   - Handle missing values
   - Remove duplicates
   - Fix data types
   - Normalize values

3. **Feature Engineering**
   - Create new features
   - Encode categorical variables
   - Scale numerical features
   - Select relevant features

## Pipeline Configuration

The data processing pipeline is defined in `dvc.yaml`:

```yaml
stages:
  process_data:
    cmd: python src/mlops/data/process.py
    deps:
      - data/raw/dataset.csv
      - src/mlops/data/process.py
    params:
      - data.split_ratio
      - data.random_state
    outs:
      - data/processed/train.csv
      - data/processed/test.csv
```

## Usage

To run the data processing pipeline:

```bash
# Run data processing stage
dvc repro process_data

# Check pipeline status
dvc status

# Push processed data to remote storage
dvc push
```

## Data Versioning

Each dataset version is tracked in DVC:

```bash
# Add new raw data
dvc add data/raw/dataset.csv

# Add processed datasets
dvc add data/processed/train.csv
dvc add data/processed/test.csv

# Commit changes
git add data/raw/dataset.csv.dvc data/processed/*.dvc
git commit -m "feat: add new dataset version"
``` 