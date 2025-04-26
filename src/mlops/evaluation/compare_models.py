"""Compare model results and generate report."""
import json
from pathlib import Path
from typing import Any, Dict, List, cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_metrics(metrics_file: Path) -> List[Dict[str, Any]]:
    """Load metrics from JSON file.

    Args:
        metrics_file: Path to metrics file

    Returns:
        List of metrics dictionaries
    """
    with open(metrics_file, "r") as f:
        return cast(List[Dict[str, Any]], json.load(f))


def create_comparison_plots(
    metrics: List[Dict[str, Any]], dataset: str, output_dir: Path
) -> None:
    """Create comparison plots for model metrics.

    Args:
        metrics: List of metrics dictionaries
        dataset: Name of the dataset
        output_dir: Directory to save plots
    """
    df = pd.DataFrame(metrics)

    # Set style
    sns.set_theme(style="whitegrid")

    # Create bar plots for each metric
    metrics_to_plot = [col for col in df.columns if col != "model_name"]

    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x="model_name", y=metric)
        plt.title(f"{metric.upper()} by Model Type - {dataset}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset.capitalize()}_{metric}_comparison.png")
        plt.close()


def generate_report(
    titanic_metrics: List[Dict[str, Any]], house_metrics: List[Dict[str, Any]]
) -> str:
    """Generate comparison report.

    Args:
        titanic_metrics: List of Titanic metrics
        house_metrics: List of House Price metrics

    Returns:
        Markdown formatted report
    """
    report = """# Model Comparison Report

## Titanic Dataset Results

### Classification Metrics

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|---------|-----------|
"""

    for metric in titanic_metrics:
        report += f"| {metric['model_name']} | {metric['accuracy']:.3f} | {metric['precision']:.3f} | {metric['recall']:.3f} | {metric['f1']:.3f} |\n"

    report += """
## House Price Dataset Results

### Regression Metrics

| Model | MSE | RMSE | MAE | R² Score |
|-------|-----|------|-----|----------|
"""

    for metric in house_metrics:
        report += f"| {metric['model_name']} | {metric['mse']:.0f} | {metric['rmse']:.0f} | {metric['mae']:.0f} | {metric['r2']:.3f} |\n"

    report += """
## Analysis

### Titanic Dataset

The classification models were evaluated on accuracy, precision, recall, and F1 score:
"""

    # Find best model for Titanic
    best_titanic = max(titanic_metrics, key=lambda x: x["f1"])
    report += f"\n- Best performing model: **{best_titanic['model_name']}** with F1 score of {best_titanic['f1']:.3f}"

    report += """

### House Price Dataset

The regression models were evaluated on Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² score:
"""

    # Find best model for House Price
    best_house = max(house_metrics, key=lambda x: x["r2"])
    report += f"\n- Best performing model: **{best_house['model_name']}** with R² score of {best_house['r2']:.3f}"

    return report


def main() -> None:
    """Compare models and generate report."""
    # Setup paths
    metrics_dir = Path("metrics")
    docs_dir = Path("docs") / "model-comparison"
    docs_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics
    titanic_metrics = load_metrics(metrics_dir / "titanic_metrics.json")
    house_metrics = load_metrics(metrics_dir / "house_metrics.json")

    # Create comparison plots
    create_comparison_plots(titanic_metrics, "Titanic", docs_dir)
    create_comparison_plots(house_metrics, "House", docs_dir)

    # Generate report
    report = generate_report(titanic_metrics, house_metrics)

    # Save report
    with open(docs_dir / "model_comparison.md", "w") as f:
        f.write(report)
    print("Report generated successfully!")


if __name__ == "__main__":
    main()
