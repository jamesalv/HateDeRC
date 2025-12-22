"""
Visualization utilities for experiment comparison and analysis.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional
import pandas as pd
from ExperimentManager import ExperimentManager


def plot_training_curves(
    experiment_ids: List[str],
    base_dir: str = "./experiments",
    save_path: Optional[str] = None,
):
    """
    Plot training curves (loss, accuracy, F1) for multiple experiments.

    Args:
        experiment_ids: List of experiment IDs to plot
        base_dir: Base directory for experiments
        save_path: Optional path to save the plot
    """
    manager = ExperimentManager(base_dir=base_dir)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for exp_id in experiment_ids:
        exp_dir = manager.get_experiment_path(exp_id)
        if not exp_dir:
            print(f"Warning: Experiment {exp_id} not found")
            continue

        history_file = exp_dir / "metrics" / "training_history.json"
        if not history_file.exists():
            print(f"Warning: No training history for {exp_id}")
            continue

        with open(history_file, "r") as f:
            history = json.load(f)

        # Get experiment name
        for exp in manager.registry["experiments"]:
            if exp["experiment_id"] == exp_id:
                label = exp.get("custom_name", exp_id[:20])
                break
        else:
            label = exp_id[:20]

        epochs = range(1, len(history["train_loss"]) + 1)

        # Plot training loss
        axes[0].plot(
            epochs, history["train_loss"], "o-", label=f"{label} (train)", alpha=0.7
        )
        if "val_loss" in history:
            axes[0].plot(
                epochs, history["val_loss"], "s--", label=f"{label} (val)", alpha=0.7
            )

        # Plot accuracy
        if "val_accuracy" in history:
            axes[1].plot(epochs, history["val_accuracy"], "o-", label=label, alpha=0.7)

        # Plot F1
        if "val_f1" in history:
            axes[2].plot(epochs, history["val_f1"], "o-", label=label, alpha=0.7)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].set_title("Validation F1 Score")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_metrics_comparison(
    experiment_ids: List[str],
    base_dir: str = "./experiments",
    save_path: Optional[str] = None,
):
    """
    Create bar plots comparing final metrics across experiments.

    Args:
        experiment_ids: List of experiment IDs to compare
        base_dir: Base directory for experiments
        save_path: Optional path to save the plot
    """
    manager = ExperimentManager(base_dir=base_dir)

    data = []
    for exp_id in experiment_ids:
        exp_dir = manager.get_experiment_path(exp_id)
        if not exp_dir:
            continue

        metrics_file = exp_dir / "metrics" / "final_summary.json"
        if not metrics_file.exists():
            continue

        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        # Get experiment name
        for exp in manager.registry["experiments"]:
            if exp["experiment_id"] == exp_id:
                name = exp.get("custom_name", exp_id[:15])
                break
        else:
            name = exp_id[:15]

        data.append(
            {
                "Experiment": name,
                "F1": metrics.get("test_f1", 0),
                "Accuracy": metrics.get("test_accuracy", 0),
                "Loss": metrics.get("test_loss", 0),
            }
        )

    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # F1 Score
    axes[0].bar(df["Experiment"], df["F1"], color="steelblue", alpha=0.8)
    axes[0].set_ylabel("F1 Score")
    axes[0].set_title("Test F1 Score Comparison")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, alpha=0.3, axis="y")

    # Accuracy
    axes[1].bar(df["Experiment"], df["Accuracy"], color="forestgreen", alpha=0.8)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Test Accuracy Comparison")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, alpha=0.3, axis="y")

    # Loss
    axes[2].bar(df["Experiment"], df["Loss"], color="coral", alpha=0.8)
    axes[2].set_ylabel("Loss")
    axes[2].set_title("Test Loss Comparison")
    axes[2].tick_params(axis="x", rotation=45)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_bias_metrics(
    experiment_ids: List[str],
    base_dir: str = "./experiments",
    save_path: Optional[str] = None,
):
    """
    Plot bias metrics (GMB scores) across experiments.

    Args:
        experiment_ids: List of experiment IDs to compare
        base_dir: Base directory for experiments
        save_path: Optional path to save the plot
    """
    manager = ExperimentManager(base_dir=base_dir)

    data = []
    for exp_id in experiment_ids:
        exp_dir = manager.get_experiment_path(exp_id)
        if not exp_dir:
            continue

        bias_file = exp_dir / "metrics" / "bias_metrics.json"
        if not bias_file.exists():
            continue

        with open(bias_file, "r") as f:
            bias_data = json.load(f)

        # Get experiment name
        for exp in manager.registry["experiments"]:
            if exp["experiment_id"] == exp_id:
                name = exp.get("custom_name", exp_id[:15])
                break
        else:
            name = exp_id[:15]

        gmb_metrics = bias_data.get("gmb_metrics", {})

        data.append(
            {
                "Experiment": name,
                "GMB-SUBGROUP": gmb_metrics.get("GMB-SUBGROUP-AUC", 0),
                "GMB-BPSN": gmb_metrics.get("GMB-BPSN-AUC", 0),
                "GMB-BNSP": gmb_metrics.get("GMB-BNSP-AUC", 0),
                "GMB-COMBINED": gmb_metrics.get("GMB-COMBINED-AUC", 0),
            }
        )

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(df))
    width = 0.2

    ax.bar(
        [i - 1.5 * width for i in x],
        df["GMB-SUBGROUP"],
        width,
        label="Subgroup",
        alpha=0.8,
    )
    ax.bar([i - 0.5 * width for i in x], df["GMB-BPSN"], width, label="BPSN", alpha=0.8)
    ax.bar([i + 0.5 * width for i in x], df["GMB-BNSP"], width, label="BNSP", alpha=0.8)
    ax.bar(
        [i + 1.5 * width for i in x],
        df["GMB-COMBINED"],
        width,
        label="Combined",
        alpha=0.8,
    )

    ax.set_xlabel("Experiment")
    ax.set_ylabel("GMB AUC Score")
    ax.set_title("Bias Metrics Comparison (GMB Scores)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Experiment"], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


def plot_xai_metrics(
    experiment_ids: List[str],
    base_dir: str = "./experiments",
    save_path: Optional[str] = None,
):
    """
    Plot XAI/faithfulness metrics across experiments.

    Args:
        experiment_ids: List of experiment IDs to compare
        base_dir: Base directory for experiments
        save_path: Optional path to save the plot
    """
    manager = ExperimentManager(base_dir=base_dir)

    data = []
    for exp_id in experiment_ids:
        exp_dir = manager.get_experiment_path(exp_id)
        if not exp_dir:
            continue

        xai_file = exp_dir / "metrics" / "xai_metrics.json"
        if not xai_file.exists():
            continue

        with open(xai_file, "r") as f:
            xai_metrics = json.load(f)

        # Get experiment name
        for exp in manager.registry["experiments"]:
            if exp["experiment_id"] == exp_id:
                name = exp.get("custom_name", exp_id[:15])
                break
        else:
            name = exp_id[:15]

        data.append(
            {
                "Experiment": name,
                "AUPRC": xai_metrics.get("token_soft_metrics", 0).get('auprc', 0),
                "Token F1": xai_metrics.get("token_prf", 0).get("instance_macro", 0).get("f1", 0),
                "Comprehensiveness": xai_metrics.get("classification_scores", 0).get("comprehensiveness", 0),
                "Sufficiency": xai_metrics.get("classification_scores", 0).get("sufficiency", 0),
            }
        )

    df = pd.DataFrame(data)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # AUPRC
    axes[0, 0].bar(df["Experiment"], df["AUPRC"], color="steelblue", alpha=0.8)
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].set_title("AUPRC (Plausibility)")
    axes[0, 0].tick_params(axis="x", rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    # Token F1
    axes[0, 1].bar(df["Experiment"], df["Token F1"], color="forestgreen", alpha=0.8)
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_title("Token F1 (Plausibility)")
    axes[0, 1].tick_params(axis="x", rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # Comprehensiveness
    axes[1, 0].bar(df["Experiment"], df["Comprehensiveness"], color="coral", alpha=0.8)
    axes[1, 0].set_ylabel("Score")
    axes[1, 0].set_title("Comprehensiveness (Faithfulness)")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Sufficiency
    axes[1, 1].bar(df["Experiment"], df["Sufficiency"], color="orchid", alpha=0.8)
    axes[1, 1].set_ylabel("Score")
    axes[1, 1].set_title("Sufficiency (Faithfulness)")
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()


def create_experiment_report(
    experiment_id: str,
    base_dir: str = "./experiments",
    output_dir: Optional[str] = None,
):
    """
    Create a comprehensive HTML report for an experiment.

    Args:
        experiment_id: Experiment ID
        base_dir: Base directory for experiments
        output_dir: Optional directory to save report (defaults to experiment dir)
    """
    manager = ExperimentManager(base_dir=base_dir)
    exp_dir = manager.get_experiment_path(experiment_id)

    if not exp_dir:
        print(f"Experiment {experiment_id} not found")
        return

    # Load all data
    with open(exp_dir / "config.json", "r") as f:
        config = json.load(f)

    metrics_file = exp_dir / "metrics" / "final_summary.json"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
    else:
        metrics = {}

    # Create HTML report
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Experiment Report: {experiment_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            h1 {{ color: #333; }}
            h2 {{ color: #555; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ padding: 12px; text-align: left; border: 1px solid #ddd; }}
            th {{ background-color: #f4f4f4; font-weight: bold; }}
            .metric {{ font-size: 1.2em; color: #2c7; }}
        </style>
    </head>
    <body>
        <h1>Experiment Report</h1>
        <p><strong>Experiment ID:</strong> {experiment_id}</p>
        <p><strong>Directory:</strong> {exp_dir}</p>
        
        <h2>Configuration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Model</td><td>{config.get('model_name', 'N/A')}</td></tr>
            <tr><td>Learning Rate</td><td>{config.get('learning_rate', 'N/A')}</td></tr>
            <tr><td>Batch Size</td><td>{config.get('batch_size', 'N/A')}</td></tr>
            <tr><td>Epochs</td><td>{config.get('num_epochs', 'N/A')}</td></tr>
            <tr><td>Multi-layer Loss</td><td>{config.get('use_multi_layer_loss', 'N/A')}</td></tr>
        </table>
        
        <h2>Results</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Test F1</td><td class="metric">{metrics.get('test_f1', 'N/A'):.4f}</td></tr>
            <tr><td>Test Accuracy</td><td class="metric">{metrics.get('test_accuracy', 'N/A'):.4f}</td></tr>
            <tr><td>Test Loss</td><td>{metrics.get('test_loss', 'N/A'):.4f}</td></tr>
        </table>
        
        <p><em>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
    </body>
    </html>
    """

    # Save report
    if output_dir:
        output_path = Path(output_dir) / f"{experiment_id}_report.html"
    else:
        output_path = exp_dir / "experiment_report.html"

    with open(output_path, "w") as f:
        f.write(html)

    print(f"Report saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Example: Plot training curves for specific experiments
    experiment_ids = [
        "20241216_120000_baseline_a1b2c3d4",
        "20241216_130000_multilayer_e5f6g7h8",
    ]

    # Uncomment to test
    # plot_training_curves(experiment_ids)
    # plot_metrics_comparison(experiment_ids)
    # plot_bias_metrics(experiment_ids)
    # plot_xai_metrics(experiment_ids)
    # create_experiment_report(experiment_ids[0])

    print("Visualization utilities loaded. Use functions to create plots.")
