from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    # Model Parameters
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 2

    # Dataset Parameters
    max_length: int = 128
    class_weighting: bool = False

    # Training Parameters
    learning_rate: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 2
    train_attention: bool = False
    lambda_attn: float = 0.1
    ranking_margin: float = 0.1        # Minimum margin between token pairs
    ranking_threshold: float = 0.05    # Min difference to consider pairs significant
    use_multi_layer_loss: bool = False
    hidden_dropout_prob: float = 0

    # Optimization Parameters
    use_amp: bool = True  # Automatic Mixed Precision (2-3x speedup on GPU)
    gradient_accumulation_steps: int = (
        1  # Effective batch_size = batch_size * accumulation_steps
    )
    max_grad_norm: float = 1.0  # Gradient clipping threshold
    num_workers: int = 4  # DataLoader workers (0 for Windows, 4+ for Linux)
    pin_memory: bool = True  # Pin memory for faster GPU transfer
    use_compile: bool = False  # PyTorch 2.0+ compile (can provide 10-30% speedup)

    # Other Parameters
    seed: int = 42
    save_dir: str = "./checkpoints"
