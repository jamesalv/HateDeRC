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
    train_attention: bool = True
    use_multi_layer_loss = True
    hidden_dropout_prob=0
    
    # Other Parameters
    seed: int = 42
    save_dir: str = "./checkpoints"
