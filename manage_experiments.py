#!/usr/bin/env python
"""
Command-line interface for managing experiments.
Run: python manage_experiments.py [command] [options]
"""

import argparse
import json
from pathlib import Path
from ExperimentManager import ExperimentManager
from tabulate import tabulate


def list_experiments(args):
    """List all experiments."""
    manager = ExperimentManager(base_dir=args.base_dir)
    experiments = manager.list_experiments(status=args.status)

    if not experiments:
        print("No experiments found.")
        return

    if args.format == "table":
        # Format as table
        headers = ["ID", "Status", "Name", "Model", "F1", "Accuracy", "Date"]
        rows = []

        for exp in experiments:
            exp_dir = Path(exp["directory"])
            metrics_file = exp_dir / "metrics" / "final_summary.json"

            f1 = "N/A"
            accuracy = "N/A"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)
                    f1 = (
                        f"{metrics.get('test_f1', 'N/A'):.4f}"
                        if isinstance(metrics.get("test_f1"), (int, float))
                        else "N/A"
                    )
                    accuracy = (
                        f"{metrics.get('test_accuracy', 'N/A'):.4f}"
                        if isinstance(metrics.get("test_accuracy"), (int, float))
                        else "N/A"
                    )

            rows.append(
                [
                    (
                        exp["experiment_id"][:30] + "..."
                        if len(exp["experiment_id"]) > 30
                        else exp["experiment_id"]
                    ),
                    exp.get("status", "unknown"),
                    exp.get("custom_name", "N/A")[:20],
                    exp["config"]["model_name"].split("/")[-1][:20],
                    f1,
                    accuracy,
                    exp["timestamp"][:10],
                ]
            )

        print(tabulate(rows, headers=headers, tablefmt="grid"))
        print(f"\nTotal: {len(experiments)} experiments")
    else:
        # Format as JSON
        print(json.dumps(experiments, indent=2))


def show_experiment(args):
    """Show details of a specific experiment."""
    manager = ExperimentManager(base_dir=args.base_dir)
    exp_dir = manager.get_experiment_path(args.experiment_id)

    if not exp_dir:
        print(f"Experiment '{args.experiment_id}' not found.")
        return

    # Load all data
    with open(exp_dir / "config.json", "r") as f:
        config = json.load(f)

    metrics_file = exp_dir / "metrics" / "final_summary.json"
    metrics = {}
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            metrics = json.load(f)

    print("=" * 80)
    print(f"EXPERIMENT: {args.experiment_id}")
    print("=" * 80)
    print(f"Directory: {exp_dir}")
    print()

    print("Configuration:")
    print(f"  Model: {config['model_name']}")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Batch Size: {config['batch_size']}")
    print(f"  Epochs: {config['num_epochs']}")
    print(f"  Multi-layer Loss: {config['use_multi_layer_loss']}")
    print()

    if metrics:
        print("Results:")
        print(f"  Test F1: {metrics.get('test_f1', 'N/A')}")
        print(f"  Test Accuracy: {metrics.get('test_accuracy', 'N/A')}")
        print(f"  Test Loss: {metrics.get('test_loss', 'N/A')}")

        if "gmb_metrics" in metrics:
            print(f"\n  GMB Metrics:")
            for key, value in metrics["gmb_metrics"].items():
                print(f"    {key}: {value}")

        if "xai_metrics" in metrics:
            print(f"\n  XAI Metrics:")
            for key, value in metrics["xai_metrics"].items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")
    else:
        print("No final metrics found (experiment may be incomplete)")

    print("=" * 80)


def compare_experiments(args):
    """Compare multiple experiments."""
    manager = ExperimentManager(base_dir=args.base_dir)
    comparison = manager.compare_experiments(args.experiment_ids)

    if not comparison["experiments"]:
        print("No experiments found.")
        return

    print("=" * 80)
    print("EXPERIMENT COMPARISON")
    print("=" * 80)

    # Configuration comparison
    print("\nConfigurations:")
    headers = ["Experiment", "Model", "LR", "BS", "Epochs", "Multi-Layer"]
    rows = []
    for exp in comparison["experiments"]:
        config = exp["config"]
        rows.append(
            [
                exp["experiment_id"][:30],
                config.get("model_name", "N/A").split("/")[-1][:15],
                config.get("learning_rate", "N/A"),
                config.get("batch_size", "N/A"),
                config.get("num_epochs", "N/A"),
                "Yes" if config.get("use_multi_layer_loss") else "No",
            ]
        )
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Results comparison
    print("\nResults:")
    headers = ["Experiment", "F1", "Accuracy", "Loss"]
    rows = []
    for exp in comparison["experiments"]:
        metrics = exp["metrics"]
        rows.append(
            [
                exp["experiment_id"][:30],
                (
                    f"{metrics.get('test_f1', 'N/A'):.4f}"
                    if isinstance(metrics.get("test_f1"), (int, float))
                    else "N/A"
                ),
                (
                    f"{metrics.get('test_accuracy', 'N/A'):.4f}"
                    if isinstance(metrics.get("test_accuracy"), (int, float))
                    else "N/A"
                ),
                (
                    f"{metrics.get('test_loss', 'N/A'):.4f}"
                    if isinstance(metrics.get("test_loss"), (int, float))
                    else "N/A"
                ),
            ]
        )
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    print("=" * 80)


def export_experiment(args):
    """Export an experiment."""
    manager = ExperimentManager(base_dir=args.base_dir)

    try:
        manager.export_experiment(args.experiment_id, args.output_dir)
        print(
            f"Successfully exported experiment '{args.experiment_id}' to '{args.output_dir}'"
        )
    except Exception as e:
        print(f"Error exporting experiment: {e}")


def delete_experiment(args):
    """Delete an experiment (with confirmation)."""
    manager = ExperimentManager(base_dir=args.base_dir)
    exp_dir = manager.get_experiment_path(args.experiment_id)

    if not exp_dir:
        print(f"Experiment '{args.experiment_id}' not found.")
        return

    if not args.force:
        response = input(
            f"Are you sure you want to delete '{args.experiment_id}'? (yes/no): "
        )
        if response.lower() != "yes":
            print("Deletion cancelled.")
            return

    import shutil

    shutil.rmtree(exp_dir)

    # Update registry
    manager.registry["experiments"] = [
        exp
        for exp in manager.registry["experiments"]
        if exp["experiment_id"] != args.experiment_id
    ]
    manager._save_registry()

    print(f"Deleted experiment '{args.experiment_id}'")


def main():
    parser = argparse.ArgumentParser(
        description="Manage ML experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python manage_experiments.py list
  python manage_experiments.py list --status completed
  python manage_experiments.py show 20241216_143052_baseline_a1b2c3d4
  python manage_experiments.py compare exp1 exp2 exp3
  python manage_experiments.py export exp1 --output-dir ./backup
        """,
    )

    parser.add_argument(
        "--base-dir", default="./experiments", help="Base experiments directory"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument(
        "--status", choices=["running", "completed", "failed"], help="Filter by status"
    )
    list_parser.add_argument(
        "--format", choices=["table", "json"], default="table", help="Output format"
    )
    list_parser.set_defaults(func=list_experiments)

    # Show command
    show_parser = subparsers.add_parser("show", help="Show experiment details")
    show_parser.add_argument("experiment_id", help="Experiment ID")
    show_parser.set_defaults(func=show_experiment)

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare experiments")
    compare_parser.add_argument(
        "experiment_ids", nargs="+", help="Experiment IDs to compare"
    )
    compare_parser.set_defaults(func=compare_experiments)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export experiment")
    export_parser.add_argument("experiment_id", help="Experiment ID")
    export_parser.add_argument(
        "--output-dir", default="./exported_experiments", help="Output directory"
    )
    export_parser.set_defaults(func=export_experiment)

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete experiment")
    delete_parser.add_argument("experiment_id", help="Experiment ID")
    delete_parser.add_argument("--force", action="store_true", help="Skip confirmation")
    delete_parser.set_defaults(func=delete_experiment)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Try to use tabulate, but gracefully degrade
    global tabulate
    try:
        from tabulate import tabulate
    except ImportError:
        print(
            "Warning: 'tabulate' package not found. Install with: pip install tabulate"
        )
        print("Falling back to basic formatting.\n")

        def tabulate(rows, headers, tablefmt):
            # Simple fallback
            print(" | ".join(headers))
            print("-" * 80)
            for row in rows:
                print(" | ".join(str(cell) for cell in row))
            return ""

    args.func(args)


if __name__ == "__main__":
    main()
