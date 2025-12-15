import torch
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast, GradScaler  # type: ignore
from transformers import (
    AutoModel,  # pyright: ignore[reportPrivateImportUsage]
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
    def __init__(self, config: TrainingConfig, **kwargs):
        self.config = config

        # Initialize device:
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Configure & initialize the base model
        model_config = AutoConfig.from_pretrained(
            config.model_name, output_attentions=True, output_hidden_states=True
        )
        self.base_model = AutoModel.from_pretrained(
            config.model_name, config=model_config
        )

        # Get model dimensions
        hidden_size = self.base_model.config.hidden_size
        self.num_layers = self.base_model.config.num_hidden_layers

        # Multi-layer classifier heads (one for each transformer layer)
        self.classifier_list = nn.ModuleList(
            [nn.Linear(hidden_size, config.num_labels) for _ in range(self.num_layers)]
        )

        # Dropout layer
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Layer configuration for debiasing (similar to BertDistill)
        self.debias_layer = 3  # Layer index for auxiliary loss
        self.use_multi_layer_loss = getattr(config, "use_multi_layer_loss", False)

        # Move models to device
        self.base_model.to(self.device)
        self.classifier_list.to(self.device)

        # Configure loss function
        if config.class_weighting:
            class_weight = kwargs.get("class_weight")
            if class_weight:
                class_weight = class_weight.to(self.device)
                self.cls_criterion = CrossEntropyLoss(weight=class_weight)
            else:
                raise Exception("class_weight not found!")
        else:
            self.cls_criterion = CrossEntropyLoss()

        # Configure optimizer (for base model and all classifiers)
        params = list(self.base_model.parameters()) + list(
            self.classifier_list.parameters()
        )
        self.optimizer = AdamW(params, lr=config.learning_rate)

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
                self.base_model = torch.compile(self.base_model)
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
        self.base_model.train()
        for classifier in self.classifier_list:
            classifier.train()

        total_loss = 0
        total_cls_loss = 0
        total_lower_loss = 0
        num_batches = 0

        progress_bar = tqdm(train_dataloader, desc="Training", unit="batch")
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)

            # Mixed precision context
            with autocast(device_type="cuda", enabled=self.use_amp):
                # Forward Pass through base model (get all hidden states)
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

                # Get hidden states from all layers
                hidden_states = outputs.hidden_states  # Tuple of (num_layers+1) tensors

                # Apply pooling to each layer's hidden state (extract CLS token)
                # Note: hidden_states[0] is embeddings, hidden_states[1:] are transformer layers
                pooled_outputs = []
                for i in range(1, len(hidden_states)):  # Skip embeddings layer
                    cls_token = hidden_states[i][:, 0, :]  # Get CLS token
                    pooled_outputs.append(cls_token)

                # Get logits from all classifier heads
                logits_list = []
                for i, pooled_output in enumerate(pooled_outputs):
                    if i == len(pooled_outputs) - 1 and self.use_multi_layer_loss:
                        # Last layer: add residual connection from debias layer
                        combined = (
                            self.dropout(pooled_output)
                            + self.dropout(pooled_outputs[self.debias_layer]).detach()
                        )
                        logits = self.classifier_list[i](combined)
                    else:
                        logits = self.classifier_list[i](self.dropout(pooled_output))
                    logits_list.append(logits)

                # Main logits from final layer
                final_logits = logits_list[-1]

                if self.use_multi_layer_loss:
                    # Multi-layer loss (similar to BertDistill)
                    # Lower layer loss (auxiliary)
                    lower_loss = self.cls_criterion(
                        logits_list[self.debias_layer], labels
                    )

                    # Upper layer loss (final)
                    upper_loss = self.cls_criterion(logits_list[-1], labels)

                    # Combined loss
                    loss = 0.5 * lower_loss + 0.5 * upper_loss

                    total_lower_loss += lower_loss.item()
                    total_cls_loss += upper_loss.item()
                else:
                    # Single loss from final layer only
                    loss = self.cls_criterion(final_logits, labels)
                    total_cls_loss += loss.item()

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
                        self.base_model.parameters(), self.max_grad_norm
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.classifier_list.parameters(), self.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.base_model.parameters(), self.max_grad_norm
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.classifier_list.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()

                self.optimizer.zero_grad()

                if self.scheduler:
                    self.scheduler.step()

            if self.use_multi_layer_loss:
                progress_bar.set_postfix(
                    {
                        "total_loss": total_loss / num_batches,
                        "upper_loss": total_cls_loss / num_batches,
                        "lower_loss": total_lower_loss / num_batches,
                    }
                )
            else:
                progress_bar.set_postfix({"loss": total_loss / num_batches})

        return total_loss / num_batches

    def evaluate(self, val_dataloader):
        self.base_model.eval()
        for classifier in self.classifier_list:
            classifier.eval()

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
                    # Forward pass through base model (no hidden states needed - more efficient)
                    outputs = self.base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                    # Get final layer's CLS token (for evaluation, only use final classifier)
                    cls_token = outputs.last_hidden_state[:, 0, :]

                    # Get logits from final classifier (no dropout in eval mode)
                    logits = self.classifier_list[-1](cls_token)

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
        """Train the model and track metrics history."""
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
        print("=" * 60)

        best_f1 = 0.0

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
                self.save_model("best_model")
                print(f"  ✓ New best model saved! (F1: {best_f1:.4f})")

            # Save checkpoint every epoch
            self.save_model(f"checkpoint_epoch_{epoch + 1}")

        # Save final model and training history
        self.save_model("final_model")
        self.save_history()

        print("\n" + "=" * 60)
        print(f"Training completed!")
        print(f"Best F1 Score: {best_f1:.4f}")
        print(
            f"Training history saved to: {self.config.save_dir}/training_history.json"
        )

        return self.history

    def save_model(self, name: str):
        """Save model checkpoint."""
        save_path = Path(self.config.save_dir) / name
        save_path.mkdir(parents=True, exist_ok=True)

        # Save base model
        self.base_model.save_pretrained(save_path)

        # Save all classifier heads and optimizer state
        torch.save(
            {
                "classifier_list_state_dict": self.classifier_list.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            save_path / "training_state.pt",
        )

    def load_model(self, name: str):
        """Load model checkpoint."""
        load_path = Path(self.config.save_dir) / name

        # Load base model
        self.base_model = AutoModel.from_pretrained(load_path)
        self.base_model.to(self.device)

        # Load all classifier heads and optimizer state
        checkpoint = torch.load(load_path / "training_state.pt")
        self.classifier_list.load_state_dict(checkpoint["classifier_list_state_dict"])
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

    def predict(
        self, test_dataloader, return_layer_outputs=False, return_attentions=False
    ):
        """
        Run inference on test data and return predictions with metrics.

        Args:
            test_dataloader: DataLoader for test data
            return_layer_outputs: If True, returns predictions from all layers

        Returns:
            dict: Contains predictions, true labels, probabilities, loss, accuracy, and F1 score
        """
        self.base_model.eval()
        for classifier in self.classifier_list:
            classifier.eval()

        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        all_attention_weights = []
        all_layer_preds = (
            [[] for _ in range(self.num_layers)] if return_layer_outputs else None
        )

        print(f"Running inference on {len(test_dataloader)} batches...")

        with torch.no_grad():
            for batch in tqdm(test_dataloader, desc="Testing", unit="batch"):
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(
                    self.device, non_blocking=True
                )
                labels = batch["labels"].to(self.device, non_blocking=True)

                # Mixed precision context for inference
                with autocast(device_type="cuda", enabled=self.use_amp):
                    # Forward pass through base model (only request hidden states if needed)
                    outputs = self.base_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=return_layer_outputs,
                        output_attentions=return_attentions,
                    )

                    cls_token = outputs.last_hidden_state[:, 0, :]
                    logits = self.classifier_list[-1](cls_token)

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
                # Store attentions if requested
                if return_attentions:
                    attention_result = self.extract_attention(outputs.attentions)
                    if attention_result is not None:
                        all_attention_weights.extend(attention_result)
                    else:
                        print("No attention weights extracted.")

                # Optionally get predictions from all layers
                if return_layer_outputs and all_layer_preds is not None:
                    hidden_states = outputs.hidden_states[1:]  # Skip embeddings
                    for i, hidden_state in enumerate(hidden_states):
                        layer_cls = hidden_state[:, 0, :]
                        layer_logits = self.classifier_list[i](layer_cls)
                        layer_preds = torch.argmax(layer_logits, dim=1)
                        all_layer_preds[i].extend(layer_preds.cpu().numpy())

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
            "predictions": all_preds,
            "labels": all_labels,
            "probabilities": all_probs,
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1": f1,
        }

        if return_layer_outputs and all_layer_preds is not None:
            results["layer_predictions"] = [
                np.array(preds) for preds in all_layer_preds
            ]

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

    def extract_attention(self, attentions) -> np.ndarray | None:
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
        return avg_cls_attention.cpu().numpy()
