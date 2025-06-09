# Model Comparison Report

## Overview
This report compares the performance of three different models:
- Random Forest
- XGBoost
- LightGBM

## Performance Metrics

| Model        |   accuracy |   f1_score |   precision |   recall |
|:-------------|-----------:|-----------:|------------:|---------:|
| LightGBM     |   0.973684 |   0.979021 |    0.972222 | 0.985915 |
| RandomForest |   0.964912 |   0.972222 |    0.958904 | 0.985915 |
| XGBoost      |   0.95614  |   0.965035 |    0.958333 | 0.971831 |

## Analysis

### Best Performing Model

- accuracy: LightGBM (0.9737)
- f1_score: LightGBM (0.9790)
- precision: LightGBM (0.9722)
- recall: RandomForest (0.9859)

## Visualization
A bar plot comparing all metrics across models can be found in `model_comparison.png`.

## Conclusion

Based on overall accuracy, the LightGBM model performed best with an accuracy of 0.9737.