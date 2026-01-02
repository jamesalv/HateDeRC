import torch
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast, GradScaler  # type: ignore
from transformers import (
    AutoModelForSequenceClassification,  # pyright: ignore[reportPrivateImportUsage]
    AutoConfig,  # pyright: ignore[reportPrivateImportUsage]
)
from tqdm import tqdm
import numpy as np
from TrainingConfig import TrainingConfig
from sklearn.metrics import f1_score, accuracy_score
import json
import os
from pathlib import Path


class HateClassifier:
    """
    HAXE: Hate Speech Detection with Debiasing Residual Connections.

    This classifier implements a novel architecture for binary hate speech detection
    that reduces dependency on target-sensitive words (e.g., race, religion, gender).

    Key Components:
    ---------------
    **Ranking-Based Attention Supervision**:
       - Uses human token-level annotations as supervision signal
       - Employs pairwise ranking loss instead of cross-entropy (respects independent annotations)
       - Enforces: tokens with higher human importance should receive higher attention
       - Helps model focus on contextually relevant hate indicators

    Loss Function:
    --------------
    Total Loss = CLS Loss + λ × attention_ranking_loss

    Where:
    - λ (lambda_attn): Weight for attention supervision loss
    """

    def __init__(self, config: TrainingConfig, **kwargs):
        self.config = config

        # Initialize device:
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Configure & initialize the model with classification head
        model_config = AutoConfig.from_pretrained(
            config.model_name,
            num_labels=config.num_labels,
            output_attentions=True,
            hidden_dropout_prob=config.hidden_dropout_prob,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name, config=model_config
        )

        self.lambda_attn = getattr(
            config, "lambda_attn", 0.1
        )  # λ: attention loss weight

        # Attention Training Configuration
        self.train_attention = getattr(config, "train_attention", False)
        self.ranking_margin = getattr(
            config, "ranking_margin", 0.1
        )  # Margin for pairwise ranking
        self.ranking_threshold = getattr(
            config, "ranking_threshold", 0.05
        )  # Threshold for significant pairs

        # Move model to device
        self.model.to(self.device)

        # Configure loss function
        if config.class_weighting:
            class_weight = kwargs.get("class_weight")
            print("Using class weighting for loss function.")
            if class_weight is not None:
                class_weight = class_weight.to(self.device)
                self.cls_criterion = CrossEntropyLoss(weight=class_weight)
            # else:
            #     raise Exception("class_weight not found!")
        else:
            self.cls_criterion = CrossEntropyLoss()

        # Configure optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)

        # Learning rate scheduler
        self.scheduler = None

        # Mixed precision training
        self.use_amp = config.use_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None

        # Gradient accumulation
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        self.max_grad_norm = config.max_grad_norm

        # Torch compile (PyTorch 2.0+)
        if hasattr(config, "use_compile") and config.use_compile:
            try:
                self.model = torch.compile(self.model)
                print("✓ Model compiled with torch.compile")
            except Exception as e:
                print(f"Warning: torch.compile failed: {e}")

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
        }

    def train_epoch(self, train_dataloader):
        """
        Train for one epoch.

        Returns:
            float: Average total loss for the epoch
        """
        self.model.train()

        # Track individual loss components for monitoring
        total_loss = 0
        total_cls_loss = 0
        total_attn_loss = 0
        num_batches = 0

        progress_bar = tqdm(train_dataloader, desc="Training", unit="batch")
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            # Mixed precision context
            with autocast(device_type="cuda", enabled=self.use_amp):
                # Forward pass through model
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=self.train_attention,
                )

                # Get logits from model
                logits = outputs.logits

                # Calculate loss
                loss_dict = self._calculate_loss(
                    logits=logits,
                    labels=labels,
                    attention_mask=attention_mask,
                    attentions=outputs.attentions if self.train_attention else None,
                    human_rationales=(
                        batch.get("rationales") if self.train_attention else None
                    ),
                )

                loss = loss_dict["total_loss"]

                # Track individual loss components
                total_cls_loss += loss_dict["cls_loss"]
                if "attn_loss" in loss_dict:
                    total_attn_loss += loss_dict["attn_loss"]

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Backward pass with gradient scaling
            if self.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation: only step optimizer every N batches
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()

                if self.scheduler:
                    self.scheduler.step()

            # Update progress bar with relevant loss components
            postfix = {"total": total_loss / num_batches}
            postfix["cls"] = total_cls_loss / num_batches
            if self.train_attention and total_attn_loss > 0:
                postfix["attn"] = total_attn_loss / num_batches

            progress_bar.set_postfix(postfix)

        return total_loss / num_batches

    def evaluate(self, val_dataloader):
        self.model.eval()

        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Evaluating", unit="batch"):
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(
                    self.device, non_blocking=True
                )
                labels = batch["labels"].to(self.device, non_blocking=True)

                # Mixed precision context for evaluation
                with autocast(device_type="cuda", enabled=self.use_amp):
                    # Forward pass through model
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                    # Get logits
                    logits = outputs.logits

                    # Calculate loss
                    loss = self.cls_criterion(logits, labels)
                    total_loss += loss.item()

                # Get predictions
                preds = torch.argmax(logits, dim=1).cpu().numpy()

                # Store predictions and labels
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")

        # Calculate loss
        avg_loss = total_loss / len(val_dataloader)

        return avg_loss, accuracy, f1

    def train(self, train_dataloader, val_dataloader):
        """
        Train the HAXE model with multi-component loss.
        
        Training Configuration:
        - Attention supervision: Ranking-based loss from human annotations
        
        Loss Weights:
        - Attention: {:.2f}
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            
        Returns:
            dict: Training history with loss and metrics per epoch
        """.format(
            self.lambda_attn
        )
        print(f"Training on device: {self.device}")
        print(f"Model: {self.config.model_name}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(
            f"Effective batch size: {self.config.batch_size * self.gradient_accumulation_steps}"
        )
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Mixed precision (AMP): {self.use_amp}")
        print(f"Gradient clipping: {self.max_grad_norm}")
        print(f"\nLoss Configuration:")
        print(f"  Attention supervision: {self.train_attention}")
        if self.train_attention:
            print(f"    - Ranking loss: λ={self.lambda_attn}")
            print(
                f"    - Margin: {self.ranking_margin}, Threshold: {self.ranking_threshold}"
            )
        print("=" * 60)

        best_f1 = 0.0
        patience_counter = 0
        early_stopping_patience = getattr(self.config, "early_stopping_patience", 0)

        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")

            # Train for one epoch
            train_loss = self.train_epoch(train_dataloader)

            # Evaluate on validation set
            val_loss, val_accuracy, val_f1 = self.evaluate(val_dataloader)

            # Store metrics in history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_accuracy)
            self.history["val_f1"].append(val_f1)

            # Print epoch summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val Acc:    {val_accuracy:.4f}")
            print(f"  Val F1:     {val_f1:.4f}")

            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0  # Reset patience counter
                self.save_model("best_model")
                print(f"  ✓ New best model saved! (F1: {best_f1:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement (patience: {patience_counter}/{early_stopping_patience})")
                
                # Early stopping check
                if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
                    print(f"\n  ⚠ Early stopping triggered! No improvement for {early_stopping_patience} epochs.")
                    print(f"  Best F1: {best_f1:.4f} (Epoch {epoch + 1 - patience_counter})")
                    break

        self.save_history()

        print("\n" + "=" * 60)
        print(f"Training completed!")
        if patience_counter >= early_stopping_patience and early_stopping_patience > 0:
            print(f"Stopped early at epoch {epoch + 1}/{self.config.num_epochs}")
        print(f"Best F1 Score: {best_f1:.4f}")
        print(
            f"Training history saved to: {self.config.save_dir}/training_history.json"
        )

        return self.history

    def save_model(self, name: str):
        """Save model checkpoint."""
        save_path = Path(self.config.save_dir) / name
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model (includes classification head)
        self.model.save_pretrained(save_path)

        # Save optimizer state and config
        torch.save(
            {
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            save_path / "training_state.pt",
        )

    def load_model(self, name: str):
        """Load model checkpoint."""
        load_path = Path(self.config.save_dir) / name

        # Load model (includes classification head)
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        self.model.to(self.device)

        # Load optimizer state
        # Note: weights_only=False is safe for your own checkpoints
        checkpoint = torch.load(load_path / "training_state.pt", weights_only=False)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        print(f"Model loaded from: {load_path}")

    def save_history(self):
        """Save training history to JSON file."""
        save_path = Path(self.config.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        history_path = save_path / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def load_history(self):
        """Load training history from JSON file."""
        history_path = Path(self.config.save_dir) / "training_history.json"

        if history_path.exists():
            with open(history_path, "r") as f:
                self.history = json.load(f)
            print(f"Training history loaded from: {history_path}")
        else:
            print(f"No training history found at: {history_path}")

    def predict(self, test_dataloader, return_attentions=False):
        """
        Run inference on test data and return predictions with metrics.

        Args:
            test_dataloader: DataLoader for test data
            return_attentions: If True, returns attention weights

        Returns:
            dict: Contains predictions, true labels, probabilities, loss, accuracy, and F1 score
        """
        self.model.eval()

        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        all_attention_weights = []
        all_post_ids = []

        print(f"Running inference on {len(test_dataloader)} batches...")

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Testing", unit="batch"):
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(
                    self.device, non_blocking=True
                )
                labels = batch["labels"].to(self.device, non_blocking=True)
                post_id = batch["post_id"]

                # Mixed precision context for inference
                with autocast(device_type="cuda", enabled=self.use_amp):
                    # Forward pass through model
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_attentions=return_attentions,
                    )

                    logits = outputs.logits

                    # Calculate loss
                    loss = self.cls_criterion(logits, labels)
                    total_loss += loss.item()

                    # Get predictions and probabilities
                    probs = F.softmax(logits, dim=1)
                    preds = torch.argmax(logits, dim=1)

                # Store predictions, labels, and probabilities
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_post_ids.extend(post_id)
                # Store attentions if requested
                if return_attentions:
                    attention_result = self.extract_attention(outputs.attentions)
                    if attention_result is not None:
                        all_attention_weights.extend(attention_result)
                    else:
                        print("No attention weights extracted.")

        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")
        avg_loss = total_loss / len(test_dataloader)

        # Print summary
        print("\n" + "=" * 60)
        print("Test Results:")
        print(f"  Test Loss:     {avg_loss:.4f}")
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  Test F1:       {f1:.4f}")
        print("=" * 60)

        results = {
            "post_ids": all_post_ids,
            "predictions": all_preds,
            "labels": all_labels,
            "probabilities": all_probs,
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1": f1,
        }

        if return_attentions:
            results["attentions"] = all_attention_weights

        return results

    def save_predictions(self, results: dict, filename: str = "test_results.json"):
        """
        Save prediction results to file.

        Args:
            results: Dictionary returned from predict() method
            filename: Name of the file to save results
        """
        save_path = Path(self.config.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {
            "predictions": results["predictions"].tolist(),
            "labels": results["labels"].tolist(),
            "probabilities": results["probabilities"].tolist(),
            "loss": float(results["loss"]),
            "accuracy": float(results["accuracy"]),
            "f1": float(results["f1"]),
        }

        results_path = save_path / filename
        with open(results_path, "w") as f:
            json.dump(results_serializable, f, indent=2)

        print(f"Test results saved to: {results_path}")

    def extract_attention(self, attentions, return_tensor=False):
        """
        Extract CLS token attention from last layer, averaged across all heads.

        Args:
            attentions: Tuple of attention tensors from model
            return_tensor: If True, returns tensor (for training). If False, returns numpy (for inference)

        Returns:
            (batch_size, seq_len) attention weights as tensor or numpy array
        """
        if attentions is None or len(attentions) == 0:
            print("WARNING: No attention data available.")
            return None

        # Take CLS representation from the last layer's attentions
        last_layer_attentions = attentions[
            -1
        ]  # Shape: (batch_size, num_heads, seq_len, seq_len)
        cls_attentions = last_layer_attentions[
            :, :, 0, :
        ]  # Shape: (batch_size, num_heads, seq_len)
        # Average over all heads
        avg_cls_attention = cls_attentions.mean(dim=1)  # Shape: (batch_size, seq_len)

        if return_tensor:
            return (
                avg_cls_attention  # Keep as tensor for training (preserves gradients)
            )
        else:
            return avg_cls_attention.cpu().numpy()  # Convert to numpy for inference

    def _calculate_loss(
        self,
        logits,
        labels,
        attention_mask,
        attentions=None,
        human_rationales=None,
    ):
        """
        Calculate unified loss with configurable component weights.

        This method computes the total loss as a combination of:
        1. Classification loss (cross-entropy)
        2. Attention ranking loss (if attention supervision enabled)

        Args:
            logits: Logits from classifier (batch_size, num_labels)
            labels: Ground truth labels (batch_size,)
            attention_mask: Attention mask for padding (batch_size, seq_len)
            attentions: Tuple of attention tensors from model (optional)
            human_rationales: Human token annotations (batch_size, seq_len) (optional)

        Returns:
            dict: Dictionary containing:
                - 'total_loss': Sum of all loss components
                - 'cls_loss': Classification loss
                - 'attn_loss': Attention ranking loss (if attention training enabled)
        """
        loss_dict = {}

        # Classification loss
        cls_loss = self.cls_criterion(logits, labels)
        loss_dict["cls_loss"] = cls_loss.item()
        total_loss = cls_loss

        # Attention Ranking Loss
        if (
            self.train_attention
            and attentions is not None
            and human_rationales is not None
        ):
            if len(attentions) > 0:
                # Extract model attention from last layer
                model_attention = self.extract_attention(attentions, return_tensor=True)

                # Move rationales to device
                human_rationales = human_rationales.to(self.device, non_blocking=True)

                # Calculate ranking loss (already weighted by lambda_attn internally)
                attn_loss = self.calculate_attention_loss(
                    human_rationales, model_attention, attention_mask
                )

                total_loss = total_loss + attn_loss
                loss_dict["attn_loss"] = attn_loss.item()

        loss_dict["total_loss"] = total_loss
        return loss_dict

    def calculate_attention_loss(
        self, human_rationales, models_attentions, attention_mask
    ):
        """
        Calculate pairwise margin ranking loss for attention supervision.

        For each pair of tokens (i, j) where human_score[i] > human_score[j],
        we enforce: attention[i] - attention[j] >= margin

        This respects the independent nature of human annotations and focuses on
        relative importance rather than absolute values.

        Args:
            human_rationales: (batch_size, seq_len) - Independent token importance scores [0-1]
            models_attentions: (batch_size, seq_len) - Model attention weights (softmax tensor)
            attention_mask: (batch_size, seq_len) - Mask for padding tokens

        Returns:
            Scalar ranking loss
        """
        # Ensure models_attentions is a tensor (should be from extract_attention with return_tensor=True)
        if isinstance(models_attentions, np.ndarray):
            models_attentions = torch.from_numpy(models_attentions).to(self.device)

        batch_size, seq_len = human_rationales.shape

        # Mask out padding positions
        human_rationales = human_rationales * attention_mask
        models_attentions = models_attentions * attention_mask

        total_loss = 0.0
        total_pairs = 0

        for b in range(batch_size):
            # Get valid (non-padding) positions for this sample
            valid_mask = attention_mask[b].bool()
            valid_indices = torch.where(valid_mask)[0]

            if len(valid_indices) < 2:
                continue  # Skip if less than 2 valid tokens

            # Get human scores and model attentions for valid tokens
            human_scores = human_rationales[b, valid_indices]  # (num_valid,)
            model_attn = models_attentions[b, valid_indices]  # (num_valid,)

            # Create all pairs: (num_valid, num_valid)
            # human_i: (num_valid, 1), human_j: (1, num_valid)
            human_i = human_scores.unsqueeze(1)  # (num_valid, 1)
            human_j = human_scores.unsqueeze(0)  # (1, num_valid)

            model_i = model_attn.unsqueeze(1)  # (num_valid, 1)
            model_j = model_attn.unsqueeze(0)  # (1, num_valid)

            # Find pairs where human_i > human_j (should have model_i > model_j)
            human_diff = human_i - human_j  # (num_valid, num_valid)
            model_diff = model_i - model_j  # (num_valid, num_valid)

            # Only consider pairs where there's a clear difference in human scores
            # (avoid pairs with very similar scores)
            significant_pairs = (
                human_diff > self.ranking_threshold
            ).float()  # Threshold to avoid noise

            # Margin ranking loss: max(0, margin - (model_i - model_j)) when human_i > human_j
            # We want: model_i - model_j >= margin when human_i > human_j
            ranking_loss = torch.relu(self.ranking_margin - model_diff)

            # Apply mask to only consider significant pairs
            ranking_loss = ranking_loss * significant_pairs

            # Accumulate
            num_pairs = significant_pairs.sum()
            if num_pairs > 0:
                total_loss += ranking_loss.sum() / num_pairs
                total_pairs += 1

        if total_pairs == 0:
            return torch.tensor(0.0, device=human_rationales.device)

        avg_loss = total_loss / total_pairs
        return self.lambda_attn * avg_loss
