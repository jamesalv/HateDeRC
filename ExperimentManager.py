import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import shutil
from TrainingConfig import TrainingConfig


class ExperimentManager:
    """
    Manages experiment tracking and systematic organization of training outputs.
    Creates unique experiment directories and maintains a registry of all experiments.
    """

    def __init__(
        self,
        base_dir: str = "./experiments",
        registry_file: str = "experiment_registry.json",
    ):
        """
        Initialize the experiment manager.

        Args:
            base_dir: Base directory where all experiments will be stored
            registry_file: JSON file to track all experiments
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.registry_file = self.base_dir / registry_file
        self.registry = self._load_registry()
        self.current_experiment_dir = None
        self.current_experiment_id = None

    def _load_registry(self) -> Dict:
        """Load the experiment registry from disk."""
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                return json.load(f)
        return {"experiments": []}

    def _save_registry(self):
        """Save the experiment registry to disk."""
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2)

    def _generate_experiment_id(
        self, config: TrainingConfig, custom_name: Optional[str] = None
    ) -> str:
        """
        Generate a unique experiment ID based on timestamp and config hash.

        Args:
            config: Training configuration
            custom_name: Optional custom name to append to experiment ID

        Returns:
            Unique experiment ID string
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create config hash (short version of config parameters)
        config_dict = {
            "model": config.model_name.split("/")[-1],  # Just model name
            "lr": config.learning_rate,
            "bs": config.batch_size,
            "ep": config.num_epochs,
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        if custom_name:
            return f"{timestamp}_{custom_name}_{config_hash}"
        else:
            return f"{timestamp}_{config_hash}"

    def create_experiment(
        self,
        config: TrainingConfig,
        custom_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Path:
        """
        Create a new experiment directory and register it.

        Args:
            config: Training configuration
            custom_name: Optional custom name for the experiment
            description: Optional description of the experiment

        Returns:
            Path to the experiment directory
        """
        # Generate unique experiment ID
        self.current_experiment_id = self._generate_experiment_id(config, custom_name)
        self.current_experiment_dir = self.base_dir / self.current_experiment_id

        # Create experiment directory structure
        self.current_experiment_dir.mkdir(parents=True, exist_ok=True)
        (self.current_experiment_dir / "checkpoints").mkdir(exist_ok=True)
        (self.current_experiment_dir / "results").mkdir(exist_ok=True)
        (self.current_experiment_dir / "metrics").mkdir(exist_ok=True)
        (self.current_experiment_dir / "logs").mkdir(exist_ok=True)

        # Save configuration
        config_dict = self._config_to_dict(config)
        with open(self.current_experiment_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

        # Register experiment
        experiment_entry = {
            "experiment_id": self.current_experiment_id,
            "timestamp": datetime.now().isoformat(),
            "config": config_dict,
            "description": description or "No description provided",
            "custom_name": custom_name,
            "directory": str(self.current_experiment_dir),
            "status": "running",
        }
        self.registry["experiments"].append(experiment_entry)
        self._save_registry()

        # Update config save_dir to point to this experiment
        config.save_dir = str(self.current_experiment_dir / "checkpoints")

        print(f"=" * 80)
        print(f"Created new experiment: {self.current_experiment_id}")
        print(f"Directory: {self.current_experiment_dir}")
        if description:
            print(f"Description: {description}")
        print(f"=" * 80)

        return self.current_experiment_dir

    def _config_to_dict(self, config: TrainingConfig) -> Dict[str, Any]:
        """Convert TrainingConfig to dictionary."""
        return {
            # Model Parameters
            "model_name": config.model_name,
            "num_labels": config.num_labels,
            # Dataset Parameters
            "max_length": config.max_length,
            "class_weighting": config.class_weighting,
            # Training Parameters
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "num_epochs": config.num_epochs,
            "train_attention": config.train_attention,
            "lambda_attn": config.lambda_attn,
            "ranking_margin": config.ranking_margin,
            "ranking_threshold": config.ranking_threshold,
            "train_entropy": config.train_entropy,
            "alpha_entropy": config.alpha_entropy,
            "hidden_dropout_prob": config.hidden_dropout_prob,
            # Multi-Stage Training Parameters
            "use_multistage_training": config.use_multistage_training,
            "entropy_only_epochs": config.entropy_only_epochs,
            "attention_alignment_epochs": config.attention_alignment_epochs,
            "model_rationale_topk": config.model_rationale_topk,
            "model_rationale_threshold": config.model_rationale_threshold,
            "lambda_attn_alignment": config.lambda_attn_alignment,
            "ranking_margin_alignment": config.ranking_margin_alignment,
            # Optimization Parameters
            "use_amp": config.use_amp,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "max_grad_norm": config.max_grad_norm,
            "num_workers": config.num_workers,
            "pin_memory": config.pin_memory,
            "use_compile": config.use_compile,
            # Other Parameters
            "seed": config.seed,
            "save_dir": config.save_dir,
        }

    def save_training_history(self, history: Dict[str, list]):
        """Save training history to the experiment directory."""
        if not self.current_experiment_dir:
            raise RuntimeError("No active experiment. Call create_experiment() first.")

        save_path = self.current_experiment_dir / "metrics" / "training_history.json"
        with open(save_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved to: {save_path}")

    def save_predictions(self, results: Dict, filename: str = "test_results.pkl"):
        """Save prediction results to the experiment directory."""
        if not self.current_experiment_dir:
            raise RuntimeError("No active experiment. Call create_experiment() first.")

        import pickle

        save_path = self.current_experiment_dir / "results" / filename
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Predictions saved to: {save_path}")

    def save_bias_metrics(self, gmb_metrics: Dict, bias_details: Dict):
        """Save bias evaluation metrics."""
        if not self.current_experiment_dir:
            raise RuntimeError("No active experiment. Call create_experiment() first.")

        save_path = self.current_experiment_dir / "metrics" / "bias_metrics.json"
        with open(save_path, "w") as f:
            json.dump(
                {"gmb_metrics": gmb_metrics, "bias_details": bias_details}, f, indent=2
            )
        print(f"Bias metrics saved to: {save_path}")

    def save_xai_metrics(self, xai_metrics: Dict):
        """Save XAI/faithfulness metrics."""
        if not self.current_experiment_dir:
            raise RuntimeError("No active experiment. Call create_experiment() first.")

        save_path = self.current_experiment_dir / "metrics" / "xai_metrics.json"
        with open(save_path, "w") as f:
            json.dump(xai_metrics, f, indent=2)
        print(f"XAI metrics saved to: {save_path}")

    def save_eraser_results(self, results_path: str):
        """Copy ERASER formatted results to experiment directory."""
        if not self.current_experiment_dir:
            raise RuntimeError("No active experiment. Call create_experiment() first.")

        dest_path = (
            self.current_experiment_dir / "results" / "eraser_formatted_results.jsonl"
        )
        shutil.copy(results_path, dest_path)
        print(f"ERASER results copied to: {dest_path}")

    def save_final_metrics(self, metrics: Dict[str, Any]):
        """Save comprehensive final metrics summary."""
        if not self.current_experiment_dir:
            raise RuntimeError("No active experiment. Call create_experiment() first.")

        save_path = self.current_experiment_dir / "metrics" / "final_summary.json"

        # Add timestamp
        metrics["timestamp"] = datetime.now().isoformat()

        with open(save_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Final metrics summary saved to: {save_path}")

    def mark_complete(self, status: str = "completed", notes: Optional[str] = None):
        """Mark the current experiment as complete."""
        if not self.current_experiment_id:
            raise RuntimeError("No active experiment. Call create_experiment() first.")

        # Update registry
        for exp in self.registry["experiments"]:
            if exp["experiment_id"] == self.current_experiment_id:
                exp["status"] = status
                exp["completed_at"] = datetime.now().isoformat()
                if notes:
                    exp["notes"] = notes
                break

        self._save_registry()
        print(f"Experiment {self.current_experiment_id} marked as: {status}")

    def get_experiment_path(self, experiment_id: str) -> Optional[Path]:
        """Get the path to a specific experiment."""
        for exp in self.registry["experiments"]:
            if exp["experiment_id"] == experiment_id:
                return Path(exp["directory"])
        return None

    def list_experiments(self, status: Optional[str] = None) -> list:
        """
        List all experiments, optionally filtered by status.

        Args:
            status: Filter by status ('running', 'completed', 'failed', etc.)

        Returns:
            List of experiment dictionaries
        """
        experiments = self.registry["experiments"]
        if status:
            experiments = [exp for exp in experiments if exp.get("status") == status]
        return experiments

    def print_experiment_summary(self):
        """Print a summary of all experiments."""
        print("\n" + "=" * 80)
        print("EXPERIMENT REGISTRY")
        print("=" * 80)

        if not self.registry["experiments"]:
            print("No experiments found.")
            return

        for exp in self.registry["experiments"]:
            print(f"\nExperiment ID: {exp['experiment_id']}")
            print(f"  Status: {exp.get('status', 'unknown')}")
            print(f"  Timestamp: {exp['timestamp']}")
            if exp.get("custom_name"):
                print(f"  Name: {exp['custom_name']}")
            print(f"  Description: {exp.get('description', 'N/A')}")
            print(f"  Model: {exp['config']['model_name']}")
            print(
                f"  Epochs: {exp['config']['num_epochs']}, LR: {exp['config']['learning_rate']}, BS: {exp['config']['batch_size']}"
            )
            print(f"  Directory: {exp['directory']}")

        print("\n" + "=" * 80)

    def compare_experiments(self, experiment_ids: list) -> Dict:
        """
        Compare configurations and results of multiple experiments.

        Args:
            experiment_ids: List of experiment IDs to compare

        Returns:
            Dictionary with comparison data
        """
        comparison = {"experiments": [], "config_diff": {}}

        for exp_id in experiment_ids:
            exp_dir = self.get_experiment_path(exp_id)
            if not exp_dir:
                print(f"Warning: Experiment {exp_id} not found")
                continue

            # Load config
            config_path = exp_dir / "config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
            else:
                config = {}

            # Load final metrics if available
            metrics_path = exp_dir / "metrics" / "final_summary.json"
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
            else:
                metrics = {}

            comparison["experiments"].append(
                {"experiment_id": exp_id, "config": config, "metrics": metrics}
            )

        return comparison

    def export_experiment(self, experiment_id: str, export_path: Path):
        """Export an entire experiment directory to a specified path."""
        exp_dir = self.get_experiment_path(experiment_id)
        if not exp_dir:
            raise ValueError(f"Experiment {experiment_id} not found")

        export_path = Path(export_path)
        shutil.copytree(exp_dir, export_path / experiment_id)
        print(f"Experiment {experiment_id} exported to: {export_path / experiment_id}")
