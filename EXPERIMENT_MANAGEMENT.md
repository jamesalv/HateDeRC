# Experiment Management System

A robust system for tracking, organizing, and comparing machine learning experiments based on `TrainingConfig`.

## Overview

The `ExperimentManager` class provides a systematic way to:
- 📁 Organize all experiment outputs in structured directories
- 🔖 Track experiment configurations and metadata
- 📊 Save and compare results across experiments
- 🔍 Query and filter experiments by status
- 📤 Export experiments for sharing or backup

## Quick Start

### 1. Basic Usage

```python
from ExperimentManager import ExperimentManager

# Initialize the manager
experiment_manager = ExperimentManager(base_dir="./experiments")

# Create a new experiment
experiment_dir = experiment_manager.create_experiment(
    config=config,
    custom_name="baseline_model",
    description="Baseline distilbert with standard hyperparameters"
)

# Your training code here...
model = HateClassifier(config)
history = model.train(train_loader, val_loader)

# Save results
experiment_manager.save_training_history(history)
experiment_manager.save_predictions(test_results)
experiment_manager.mark_complete(status="completed")
```

### 2. Directory Structure

Each experiment creates the following structure:

```
experiments/
└── 20241216_143052_baseline_a1b2c3d4/
    ├── config.json                    # Training configuration
    ├── checkpoints/                   # Model checkpoints
    │   ├── best_model/
    │   ├── final_model/
    │   └── training_state.pt
    ├── results/                       # Prediction outputs
    │   ├── test_predictions.pkl
    │   └── eraser_formatted_results.jsonl
    ├── metrics/                       # All metrics
    │   ├── training_history.json
    │   ├── bias_metrics.json
    │   ├── xai_metrics.json
    │   └── final_summary.json
    └── logs/                          # Optional logs
```

### 3. Full Pipeline Integration

```python
# ============================================================================
# COMPLETE EXPERIMENT PIPELINE
# ============================================================================

# 1. CREATE EXPERIMENT
experiment_manager = ExperimentManager()
experiment_dir = experiment_manager.create_experiment(
    config=config,
    custom_name="multi_layer_test",
    description="Testing multi-layer loss approach"
)

# 2. TRAIN
model = HateClassifier(config)  # config.save_dir auto-updated
history = model.train(train_loader, val_loader)
experiment_manager.save_training_history(history)

# 3. EVALUATE
results = model.predict(test_loader, return_attentions=True)
experiment_manager.save_predictions(results)

# 4. BIAS METRICS
gmb_metrics, bias_details = calculate_gmb_metrics(...)
experiment_manager.save_bias_metrics(gmb_metrics, bias_details)

# 5. XAI METRICS
xai_results = calculator.compute_all_metrics(...)
experiment_manager.save_xai_metrics(xai_results)

# 6. FINAL SUMMARY
final_summary = {
    "test_accuracy": float(results['accuracy']),
    "test_f1": float(results['f1']),
    "gmb_metrics": gmb_metrics,
    "xai_metrics": xai_results,
}
experiment_manager.save_final_metrics(final_summary)

# 7. MARK COMPLETE
experiment_manager.mark_complete(status="completed")
```

## Features

### Experiment ID Generation

Experiment IDs are automatically generated with:
- **Timestamp**: `YYYYMMDD_HHMMSS` format
- **Custom name**: Your provided name (optional)
- **Config hash**: Short hash of key config parameters (8 chars)

Example: `20241216_143052_baseline_a1b2c3d4`

### Experiment Registry

All experiments are tracked in `experiment_registry.json`:

```json
{
  "experiments": [
    {
      "experiment_id": "20241216_143052_baseline_a1b2c3d4",
      "timestamp": "2024-12-16T14:30:52",
      "status": "completed",
      "custom_name": "baseline_model",
      "description": "Baseline with standard hyperparameters",
      "config": { ... },
      "directory": "./experiments/20241216_143052_baseline_a1b2c3d4",
      "completed_at": "2024-12-16T15:45:30"
    }
  ]
}
```

### Query Experiments

```python
# List all experiments
experiment_manager.print_experiment_summary()

# Filter by status
completed = experiment_manager.list_experiments(status="completed")
running = experiment_manager.list_experiments(status="running")
failed = experiment_manager.list_experiments(status="failed")

# Get specific experiment path
exp_dir = experiment_manager.get_experiment_path("20241216_143052_baseline_a1b2c3d4")
```

### Compare Experiments

```python
# Compare multiple experiments
comparison = experiment_manager.compare_experiments([
    "20241216_143052_baseline_a1b2c3d4",
    "20241216_153045_multilayer_e5f6g7h8"
])

# Access comparison data
for exp in comparison["experiments"]:
    print(f"ID: {exp['experiment_id']}")
    print(f"F1: {exp['metrics']['test_f1']}")
    print(f"Accuracy: {exp['metrics']['test_accuracy']}")
```

### Export Experiments

```python
# Export for sharing or backup
experiment_manager.export_experiment(
    experiment_id="20241216_143052_baseline_a1b2c3d4",
    export_path="./shared_experiments"
)
```

## API Reference

### ExperimentManager

#### `__init__(base_dir="./experiments", registry_file="experiment_registry.json")`
Initialize the experiment manager.

#### `create_experiment(config, custom_name=None, description=None)`
Create a new experiment and return its directory path.

#### `save_training_history(history)`
Save training history (loss, accuracy, etc.).

#### `save_predictions(results, filename="test_predictions.pkl")`
Save model predictions.

#### `save_bias_metrics(gmb_metrics, bias_details)`
Save bias evaluation metrics.

#### `save_xai_metrics(xai_metrics)`
Save XAI/faithfulness metrics.

#### `save_eraser_results(results_path)`
Copy ERASER formatted results to experiment directory.

#### `save_final_metrics(metrics)`
Save comprehensive final metrics summary.

#### `mark_complete(status="completed", notes=None)`
Mark experiment as complete with optional notes.

#### `list_experiments(status=None)`
List all experiments, optionally filtered by status.

#### `print_experiment_summary()`
Print a formatted summary of all experiments.

#### `compare_experiments(experiment_ids)`
Compare configurations and results of multiple experiments.

#### `export_experiment(experiment_id, export_path)`
Export entire experiment directory to specified path.

#### `get_experiment_path(experiment_id)`
Get the path to a specific experiment.

## Best Practices

### 1. Naming Conventions

Use descriptive custom names that indicate what you're testing:
- `baseline_distilbert`
- `multilayer_loss_v1`
- `high_lr_experiment`
- `class_weighted_v2`

### 2. Descriptions

Provide clear descriptions of what each experiment tests:
```python
description="Testing impact of higher learning rate (5e-5) on convergence speed"
```

### 3. Status Management

Use consistent status values:
- `"running"` - Experiment in progress
- `"completed"` - Successfully completed
- `"failed"` - Failed during execution
- `"interrupted"` - Manually stopped

### 4. Notes

Add notes when marking experiments complete:
```python
experiment_manager.mark_complete(
    status="completed",
    notes="Best F1 score so far: 0.89. Model converged after 3 epochs."
)
```

## Troubleshooting

### Issue: Config save_dir not updating

**Solution**: Ensure you create the experiment BEFORE initializing the model:
```python
# ✓ CORRECT
experiment_dir = experiment_manager.create_experiment(config=config)
model = HateClassifier(config)

# ✗ WRONG
model = HateClassifier(config)
experiment_dir = experiment_manager.create_experiment(config=config)
```

### Issue: Cannot find old experiments

**Solution**: Check the `experiment_registry.json` file is not corrupted. You can manually inspect it:
```python
import json
with open("./experiments/experiment_registry.json", 'r') as f:
    registry = json.load(f)
print(json.dumps(registry, indent=2))
```

### Issue: Experiment directory exists

**Solution**: Each experiment gets a unique timestamp-based ID, so collisions are rare. If needed, add a more specific custom name.

## Migration from Old System

If you have existing experiments in `./checkpoints`:

1. Keep your old checkpoints as-is
2. Start using ExperimentManager for new experiments
3. Optionally, manually organize old experiments:

```python
# Example: Import old checkpoint
import shutil
old_checkpoint = "./checkpoints/best_model"
new_exp_dir = experiment_manager.create_experiment(
    config=old_config,
    custom_name="legacy_model",
    description="Migrated from old checkpoint system"
)
shutil.copytree(old_checkpoint, new_exp_dir / "checkpoints" / "best_model")
```

## Advanced Usage

### Custom Metrics

Add your own metrics to final summary:
```python
final_summary = {
    "test_accuracy": 0.89,
    "test_f1": 0.87,
    "custom_metric": my_custom_calculation(),
    "inference_time_ms": 45.2,
    "memory_usage_mb": 512.0
}
experiment_manager.save_final_metrics(final_summary)
```

### Programmatic Analysis

Load and analyze experiments programmatically:
```python
import pandas as pd

# Collect all experiments
all_experiments = experiment_manager.list_experiments(status="completed")

# Create comparison DataFrame
data = []
for exp in all_experiments:
    exp_dir = Path(exp['directory'])
    with open(exp_dir / "metrics" / "final_summary.json") as f:
        metrics = json.load(f)
    
    data.append({
        'experiment_id': exp['experiment_id'],
        'model': exp['config']['model_name'],
        'lr': exp['config']['learning_rate'],
        'f1': metrics['test_f1'],
        'accuracy': metrics['test_accuracy']
    })

df = pd.DataFrame(data)
print(df.sort_values('f1', ascending=False))
```

## Integration with Git

Add to `.gitignore`:
```
# Experiment outputs (too large for git)
experiments/*/checkpoints/
experiments/*/results/*.pkl

# Keep configs and metrics (small, important)
!experiments/*/config.json
!experiments/*/metrics/
experiments/experiment_registry.json
```

This allows you to track configurations and metrics while excluding large model files.
