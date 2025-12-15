import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, auc, f1_score, precision_score, recall_score
from typing import List, Dict, Tuple


class HateInterpreter:
    """
    Minimal implementation of ERASER metrics for research
    
    Implements:
    1. Plausibility: AUPRC (soft), Token F1 (hard)
    2. Faithfulness: Comprehensiveness, Sufficiency
    """
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def compute_all_metrics(
        self, 
        input_ids_list: List[torch.Tensor],  # Already tokenized!
        attention_masks_list: List[torch.Tensor],  # Attention masks
        attention_scores: List[np.ndarray],  # Soft scores per token
        human_rationales: List[torch.Tensor],  # Binary mask per token
        predicted_classes: List[int],  # Hard predictions
        original_probs: List[np.ndarray]  # [prob_class_0, prob_class_1, ...]
    ) -> Dict[str, float]:
        """
        Compute all metrics for a dataset
        
        Args:
            input_ids_list: List of input_ids tensors (already tokenized)
            attention_masks_list: List of attention_mask tensors
            attention_scores: Attention weights per token (soft scores)
            human_rationales: Binary masks indicating human rationales
            predicted_classes: Model's predicted class for each instance
            original_probs: Model's original class probabilities
            
        Returns:
            Dictionary with all metrics
        """
        # 1. Calculate average rationale length (for top-k selection)
        k = self._calculate_average_rationale_length(human_rationales, attention_masks_list)
        print(f"Using k={k} (average human rationale length)")
        
        # 2. PLAUSIBILITY METRICS
        auprc = self._compute_auprc(attention_scores, human_rationales, attention_masks_list)
        
        # Extract top-k tokens as hard predictions
        hard_predictions = self._extract_top_k_tokens(attention_scores, attention_masks_list, k)
        token_f1, token_prec, token_rec = self._compute_token_f1(
            hard_predictions, human_rationales, attention_masks_list
        )
        
        # 3. FAITHFULNESS METRICS
        comp_scores, suff_scores = self._compute_faithfulness(
            input_ids_list, attention_masks_list, hard_predictions, 
            predicted_classes, original_probs
        )
        
        comprehensiveness = np.mean(comp_scores)
        sufficiency = np.mean(suff_scores)
        
        return {
            # Plausibility (alignment with human rationales)
            'auprc': auprc,
            'token_f1': token_f1,
            'token_precision': token_prec,
            'token_recall': token_rec,
            
            # Faithfulness (model actually used these rationales)
            'comprehensiveness': comprehensiveness,  # Higher is better
            'sufficiency': sufficiency,  # Lower is better
            
            # Additional info
            'avg_rationale_length': k,
        }
    
    def _calculate_average_rationale_length(
        self, 
        human_rationales: List[torch.Tensor],
        attention_masks_list: List[torch.Tensor]
    ) -> int:
        """Calculate average number of rationale tokens (for top-k), excluding padding"""
        lengths = []
        for rat, mask in zip(human_rationales, attention_masks_list):
            # Only count non-padding tokens
            valid_positions = mask.bool()
            rat_count = (rat[valid_positions] == 1).sum().item()
            lengths.append(rat_count)
        
        return max(1, int(np.mean(lengths)))  # At least 1
    
    def _compute_auprc(
        self,
        attention_scores: List[np.ndarray],
        human_rationales: List[torch.Tensor],
        attention_masks_list: List[torch.Tensor]
    ) -> float:
        """
        Compute Area Under Precision-Recall Curve for soft scores
        
        This measures: "If I rank tokens by attention, do I recover human rationales?"
        """
        all_scores = []
        all_labels = []
        
        for attn, rat, mask in zip(attention_scores, human_rationales, attention_masks_list):
            # Only consider non-padding tokens
            valid_positions = mask.bool().cpu().numpy()
            
            all_scores.extend(attn[valid_positions].tolist())
            all_labels.extend(rat[valid_positions].cpu().numpy().tolist())
        
        # Convert to numpy
        all_scores = np.array(all_scores)
        all_labels = np.array(all_labels)
        
        # Compute precision-recall curve
        precision, recall, _ = precision_recall_curve(all_labels, all_scores)
        
        # Compute area under curve
        auprc_score = auc(recall, precision)
        
        return auprc_score
    
    def _extract_top_k_tokens(
        self,
        attention_scores: List[np.ndarray],
        attention_masks_list: List[torch.Tensor],
        k: int
    ) -> List[np.ndarray]:
        """
        Extract top-k tokens with highest attention as hard predictions
        
        Returns binary masks (1 = rationale, 0 = not rationale)
        """
        hard_predictions = []
        
        for attn, mask in zip(attention_scores, attention_masks_list):
            # Create binary mask
            pred_mask = np.zeros_like(attn, dtype=int)
            
            # Only consider non-padding tokens
            valid_positions = mask.bool().cpu().numpy()
            valid_attn = attn[valid_positions]
            
            if k > 0 and len(valid_attn) > 0:
                k_actual = min(k, len(valid_attn))
                # Get top-k indices within valid positions
                top_k_within_valid = np.argsort(valid_attn)[-k_actual:]
                
                # Map back to original positions
                valid_indices = np.where(valid_positions)[0]
                top_k_indices = valid_indices[top_k_within_valid]
                
                pred_mask[top_k_indices] = 1
            
            hard_predictions.append(pred_mask)
        
        return hard_predictions
    
    def _compute_token_f1(
        self,
        hard_predictions: List[np.ndarray],
        human_rationales: List[torch.Tensor],
        attention_masks_list: List[torch.Tensor]
    ) -> Tuple[float, float, float]:
        """
        Compute token-level F1, Precision, Recall
        
        Treats each token as binary classification problem
        """
        all_preds = []
        all_labels = []
        
        for pred, rat, mask in zip(hard_predictions, human_rationales, attention_masks_list):
            # Only consider non-padding tokens
            valid_positions = mask.bool().cpu().numpy()
            
            all_preds.extend(pred[valid_positions].tolist())
            all_labels.extend(rat[valid_positions].cpu().numpy().tolist())
        
        # Compute metrics
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        
        return f1, precision, recall
    
    def _compute_faithfulness(
        self,
        input_ids_list: List[torch.Tensor],
        attention_masks_list: List[torch.Tensor],
        hard_predictions: List[np.ndarray],
        predicted_classes: List[int],
        original_probs: List[np.ndarray]
    ) -> Tuple[List[float], List[float]]:
        """
        Compute comprehensiveness and sufficiency scores
        
        Comprehensiveness: P(original) - P(text without rationales)
            - HIGH is good (removing rationales hurt prediction)
            
        Sufficiency: P(original) - P(only rationales)
            - LOW is good (rationales alone are sufficient)
        """
        comprehensiveness_scores = []
        sufficiency_scores = []
        
        for input_ids, mask, rationale_mask, pred_class, orig_prob in zip(
            input_ids_list, attention_masks_list, hard_predictions, 
            predicted_classes, original_probs
        ):
            # Get valid tokens (non-padding)
            valid_positions = mask.bool().squeeze()
            valid_input_ids = input_ids.squeeze()[valid_positions]
            
            # Get original probability for predicted class
            orig_class_prob = orig_prob[pred_class]
            
            # === COMPREHENSIVENESS: Remove rationales ===
            # Keep only tokens where rationale_mask is 0
            rationale_positions = rationale_mask[valid_positions.cpu().numpy()] == 0
            remaining_ids = valid_input_ids[torch.tensor(rationale_positions)]
            
            if len(remaining_ids) > 0:
                prob_without_rat = self._get_prediction_prob_from_ids(
                    remaining_ids, pred_class
                )
            else:
                prob_without_rat = 1.0 / len(orig_prob)  # Uniform distribution
            
            comprehensiveness = orig_class_prob - prob_without_rat
            
            # === SUFFICIENCY: Keep only rationales ===
            # Keep only tokens where rationale_mask is 1
            rationale_positions = rationale_mask[valid_positions.cpu().numpy()] == 1
            rationale_ids = valid_input_ids[torch.tensor(rationale_positions)]
            
            if len(rationale_ids) > 0:
                prob_only_rat = self._get_prediction_prob_from_ids(
                    rationale_ids, pred_class
                )
            else:
                prob_only_rat = 1.0 / len(orig_prob)  # Uniform distribution
            
            sufficiency = orig_class_prob - prob_only_rat
            
            comprehensiveness_scores.append(comprehensiveness)
            sufficiency_scores.append(sufficiency)
        
        return comprehensiveness_scores, sufficiency_scores
    
    def _get_prediction_prob_from_ids(
        self, 
        token_ids: torch.Tensor, 
        target_class: int,
        max_length: int = 128
    ) -> float:
        """
        Get model's probability for target class given token IDs
        
        Args:
            token_ids: Tensor of token IDs (no padding, no special tokens except CLS/SEP)
            target_class: Target class index
            max_length: Max sequence length
        """
        self.model.eval()
        
        # Handle empty input
        if len(token_ids) == 0:
            return 1.0 / self.model.config.num_labels  # Uniform
        
        # Ensure we have CLS and SEP tokens
        cls_token_id = self.tokenizer.cls_token_id
        sep_token_id = self.tokenizer.sep_token_id
        
        # Remove existing CLS/SEP if present
        token_ids = token_ids[(token_ids != cls_token_id) & (token_ids != sep_token_id)]
        
        # Add CLS at start and SEP at end
        token_ids = torch.cat([
            torch.tensor([cls_token_id]),
            token_ids[:max_length - 2],  # Leave room for CLS and SEP
            torch.tensor([sep_token_id])
        ])
        
        # Create attention mask
        attention_mask = torch.ones_like(token_ids)
        
        # Pad to max_length
        padding_length = max_length - len(token_ids)
        if padding_length > 0:
            token_ids = torch.cat([
                token_ids,
                torch.zeros(padding_length, dtype=torch.long)
            ])
            attention_mask = torch.cat([
                attention_mask,
                torch.zeros(padding_length, dtype=torch.long)
            ])
        
        # Add batch dimension and move to device
        token_ids = token_ids.unsqueeze(0).to(self.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_ids=token_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            class_prob = probs[0, target_class].item()
        
        return class_prob


# ============================================
# USAGE EXAMPLE WITH YOUR DATA FORMAT
# ============================================

# def example_usage_with_real_data():
#     """
#     Example of how to use the metrics with your actual data format
#     """
#     # Your test data format
#     test_data = [
#         {
#             'input_ids': torch.tensor([[101, 5310, 2273, ..., 102]]),
#             'attention_mask': torch.tensor([[1, 1, 1, ..., 0]]),
#             'rationales': torch.tensor([[0., 0., 0., ..., 0.]]),
#             'hard_label': 0,
#             'soft_label': 0.0,
#         },
#         # ... more instances
#     ]
    
#     # Your model predictions
#     attention_scores = [
#         np.array([0.1, 0.2, 0.8, ...]),  # Attention for instance 1
#         # ... more
#     ]
    
#     predicted_classes = [0, 1, 0, ...]  # Your model's predictions
    
#     original_probs = [
#         np.array([0.9, 0.1]),  # Probs for instance 1
#         # ... more
#     ]
    
#     # Initialize metrics
#     metrics_calculator = HateInterpreter(model, tokenizer, device)
    
#     # Prepare lists
#     input_ids_list = [d['input_ids'] for d in test_data]
#     attention_masks_list = [d['attention_mask'] for d in test_data]
#     human_rationales = [d['rationales'] for d in test_data]
    
#     # Compute all metrics
#     results = metrics_calculator.compute_all_metrics(
#         input_ids_list=input_ids_list,
#         attention_masks_list=attention_masks_list,
#         attention_scores=attention_scores,
#         human_rationales=human_rationales,
#         predicted_classes=predicted_classes,
#         original_probs=original_probs
#     )
    
#     # Print results
#     print("=== PLAUSIBILITY (Alignment with Human Rationales) ===")
#     print(f"AUPRC (soft):       {results['auprc']:.4f}")
#     print(f"Token F1 (hard):    {results['token_f1']:.4f}")
#     print(f"Token Precision:    {results['token_precision']:.4f}")
#     print(f"Token Recall:       {results['token_recall']:.4f}")
#     print()
#     print("=== FAITHFULNESS (Model Actually Used Rationales) ===")
#     print(f"Comprehensiveness:  {results['comprehensiveness']:.4f}  (higher is better)")
#     print(f"Sufficiency:        {results['sufficiency']:.4f}  (lower/negative is better)")
#     print()
#     print(f"Average rationale length: {results['avg_rationale_length']} tokens")
    
#     return results