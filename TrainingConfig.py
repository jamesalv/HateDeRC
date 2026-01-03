from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    # Model Parameters
    model_name: str = "distilbert-base-uncased"
    num_labels: int = 2
    classification_mode: str = (
        "binary"  # "binary" or "multiclass" (3-way: normal, hatespeech, offensive)
    )

    # Dataset Parameters
    max_length: int = 128
    class_weighting: bool = False

    # Training Parameters
    learning_rate: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 2
    early_stopping_patience: int = (
        2  # Stop if no improvement for N epochs (0 = disabled)
    )
    hidden_dropout_prob: float = 0.1

    # Attention-based Ranking Loss Parameters
    train_attention: bool = False
    lambda_attn: float = 0.1
    ranking_margin: float = 0.1  # Minimum margin between token pairs
    ranking_threshold: float = 0.05  # Min difference to consider pairs significant

    # Entropy Regularization Parameters
    train_entropy: bool = False
    alpha_entropy: float = 0.01  # Weight for entropy loss component

    # Multi-Stage Training Parameters
    use_multistage_training: bool = False  # Enable exploration → discovery → alignment
    entropy_only_epochs: int = 3  # Stage 1: Entropy exploration epochs
    attention_alignment_epochs: int = 2  # Stage 2: Attention alignment epochs
    model_rationale_topk: int = (
        2  # Top-k model-discovered tokens (reduced for stability)
    )
    model_rationale_threshold: float = (
        0.2  # Scale factor for model rationales (reduced for stability)
    )
    lambda_attn_alignment: float = (
        0.01  # Lighter weight for alignment phase (soft constraint)
    )
    ranking_margin_alignment: float = 0.05  # Softer margin for alignment phase

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
