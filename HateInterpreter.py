import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple
import json
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
)
from preprocessing import find_ranges

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple
import json
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    f1_score,
    precision_score,
    recall_score,
)


class HateInterpreter:
    """
    Compute faithfulness metrics using the model's existing predict() method.
    Creates modified datasets and uses DataLoader for efficient batched processing.
    """

    def __init__(self, model, tokenizer, dataset_class, batch_size=32):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_class = dataset_class
        self.batch_size = batch_size

        # Get special token IDs
        self.special_token_ids = {
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
        }
        self.special_token_ids = {x for x in self.special_token_ids if x is not None}

    def compute_all_metrics(
        self,
        test_data: List[Dict],  # Your original test data
        test_results: Dict,  # Results from prediction
        k: int = 5,  # Number of top tokens to consider
        eraser_save_path: str = "Data/eraser_formatted_results.jsonl",
    ) -> Dict[str, float]:
        """
        Compute all ERASER metrics efficiently using DataLoader approach

        Args:
            test_data: List of test instances (each with input_ids, attention_mask, rationales, labels)
            test_results: List of dictionaries containing attention scores for each instance

        Returns:
            Dictionary with all metrics
        """
        print("Computing ERASER metrics using DataLoader approach...")

        # Extract lists for easier processing
        input_ids_list = [item["input_ids"] for item in test_data]
        attention_masks_list = [item["attention_mask"] for item in test_data]
        human_rationales = [item["rationales"] for item in test_data]
        attention_scores = [item for item in test_results["attentions"]]

        # 1. Extract top-k as hard predictions
        print("[1/5] Extracting top-k tokens for hard rationale predictions...")
        hard_predictions = self._extract_top_k_tokens(
            attention_scores, attention_masks_list, input_ids_list, k
        )

        hard_rationale_predictions, soft_rationale_predictions = self._convert_attention_to_evidence_format(input_ids_list, attention_scores, hard_predictions)

        # 1. FAITHFULNESS METRICS
        print("[2/5] Computing comprehensiveness scores...")
        raw_comprehensiveness, comprehensiveness_scores = (
            self._compute_comprehensiveness(test_data, test_results, hard_predictions)
        )

        print("[3/5] Computing sufficiency scores...")
        raw_sufficiency, sufficiency_scores = self._compute_sufficiency(
            test_data, test_results, hard_predictions
        )

        # 4. Convert to eraser format
        print("[4/5] Converting results to ERASER format and saving...")
        results_eraser = self._convert_result_to_eraser_format(test_results, hard_rationale_predictions, soft_rationale_predictions, raw_sufficiency, raw_comprehensiveness)
        # Convert to JSONL format
        jsonl_output = '\n'.join([json.dumps(entry) for entry in results_eraser])
        with open(eraser_save_path, 'w') as f:
            f.write(jsonl_output)
        
        # 5. Calculate ERASER metrics
        print("[5/5] Calculating ERASER metrics...")
        
        return {
            'comprehensiveness': float(np.average(comprehensiveness_scores)),
            'sufficiency': float(np.average(sufficiency_scores)),
            
        }

    def _convert_attention_to_evidence_format(self, input_ids_list, attention_scores, hard_predictions):
        # 2. Collect evidence
        hard_rationale_predictions = []
        for idx, hp in enumerate(hard_predictions):
            evidences = []
            indexes = sorted([i for i, each in enumerate(hp.tolist()) if each == 1])
            for span in find_ranges(indexes):
                if isinstance(span, int):
                    start, end = span, span + 1
                else:
                    start, end = span[0], span[1] + 1

                evidences.append({
                    "start_token": start,
                    "end_token": end,
                })
            hard_rationale_predictions.append(evidences)

        soft_rationale_predictions = []
        for att in attention_scores:
            pred = [x for x in att if x > 0]
            soft_rationale_predictions.append(pred)

        return hard_rationale_predictions, soft_rationale_predictions

    def _convert_result_to_eraser_format(
        self,
        test_result: Dict,
        hard_rationale_predictions,
        soft_rationale_predictions,
        sufficiency_scores: np.ndarray,
        comprehensiveness_scores: np.ndarray,
    ):
        all_entries = []
        for idx, data in enumerate(test_result["post_id"]):
            entry = {
            'annotation_id': data,
            'classification': str(int(test_result["predictions"][idx])),
            'classification_scores': {
                0: float(test_result["probabilities"][idx][0]),
                1: float(test_result["probabilities"][idx][1]),
            },
            'rationales': [
                {
                    "docid": data,
                    "hard_rationale_predictions": hard_rationale_predictions[idx],
                    "soft_rationale_predictions": [float(x) for x in soft_rationale_predictions[idx]],
                }
            ],
            'sufficiency_classification_scores': {
                0: float(sufficiency_scores[idx][0]),
                1: float(sufficiency_scores[idx][1])
            },
            'comprehensiveness_classification_scores': {
                0: float(comprehensiveness_scores[idx][0]),
                1: float(comprehensiveness_scores[idx][1])
            }
            }
            all_entries.append(entry)

        return all_entries

    def _extract_top_k_tokens(
        self,
        attention_scores: List[np.ndarray],
        attention_masks_list: List[torch.Tensor],
        input_ids_list: List[torch.Tensor],
        k: int,
    ) -> List[np.ndarray]:
        """Extract top-k content tokens as hard predictions"""
        hard_predictions = []

        for idx, (attn, mask) in enumerate(zip(attention_scores, attention_masks_list)):
            pred_mask = np.zeros_like(attn, dtype=int)
            valid_positions = mask.bool().cpu().numpy().flatten()

            # Exclude special tokens
            input_ids = input_ids_list[idx].cpu().numpy().flatten()
            is_special = np.isin(input_ids, list(self.special_token_ids))
            content_positions = valid_positions & ~is_special

            content_attn = attn[content_positions]

            if k > 0 and len(content_attn) > 0:
                k_actual = min(k, len(content_attn))
                top_k_within_content = np.argsort(content_attn)[-k_actual:]
                content_indices = np.where(content_positions)[0]
                top_k_indices = content_indices[top_k_within_content]
                pred_mask[top_k_indices] = 1

            hard_predictions.append(pred_mask)

        return hard_predictions
    
    def _compute_comprehensiveness(
        self,
        test_data: List[Dict],
        test_results: Dict,
        hard_predictions: List[np.ndarray],
    ) -> Tuple[float, List[float]]:
        """
        Compute comprehensiveness: how much does REMOVING rationales hurt?
        Uses DataLoader approach for efficiency
        """
        # Create modified dataset (remove rationales from attention mask)
        modified_data = []
        for item, rationale_mask in zip(test_data, hard_predictions):
            modified_item = self._create_comprehensiveness_instance(
                item, rationale_mask
            )
            modified_data.append(modified_item)

        # Create DataLoader
        modified_dataset = self.dataset_class(modified_data)
        modified_loader = DataLoader(
            modified_dataset, batch_size=self.batch_size, shuffle=False
        )

        # Get predictions using model's predict method
        results = self.model.predict(modified_loader, return_attentions=False)
        modified_probs = results["probabilities"]

        # Calculate comprehensiveness scores
        comprehensiveness_scores = []
        for idx, (prob, label) in enumerate(zip(test_results["probabilities"], test_results['predictions'])):
            original_prob = prob[
                label
            ]  # Probability from normal prediction process for the label
            modified_prob = modified_probs[idx][label]

            # Comprehensiveness = original - modified (higher is better)
            comp_score = original_prob - modified_prob
            comprehensiveness_scores.append(comp_score)

        return modified_probs, comprehensiveness_scores

    def _compute_sufficiency(
        self,
        test_data: List[Dict],
        test_results: Dict,
        hard_predictions: List[np.ndarray],
    ) -> Tuple[List[float], List[float]]:
        """
        Compute sufficiency: how well do ONLY rationales predict?
        Uses DataLoader approach for efficiency
        """
        # Create modified dataset (keep only rationales in attention mask)
        modified_data = []
        for item, rationale_mask in zip(test_data, hard_predictions):
            modified_item = self._create_sufficiency_instance(item, rationale_mask)
            modified_data.append(modified_item)

        # Create DataLoader
        modified_dataset = self.dataset_class(modified_data)
        modified_loader = DataLoader(
            modified_dataset, batch_size=self.batch_size, shuffle=False
        )

        # Get predictions using model's predict method
        results = self.model.predict(modified_loader, return_attentions=False)
        modified_probs = results["probabilities"]

        # Calculate sufficiency scores
        sufficiency_scores = []
        for idx, (prob, label) in enumerate(zip(test_results["probabilities"], test_results['predictions'])):
            original_prob = prob[
                label
            ]  # Probability from normal prediction process for the label
            modified_prob = modified_probs[idx][label]

            # Sufficiency = original - modified (lower/negative is better)
            suff_score = original_prob - modified_prob
            sufficiency_scores.append(suff_score)

        return modified_probs, sufficiency_scores

    def _create_comprehensiveness_instance(
        self, item: Dict, rationale_mask: np.ndarray
    ) -> Dict:
        """
        Create instance for comprehensiveness: REMOVE rationales from attention
        Keep: CLS + non-rationale content tokens + SEP
        """
        input_ids = item["input_ids"].cpu().numpy().flatten()
        orig_mask = item["attention_mask"].cpu().numpy().flatten()

        # Start with original mask
        new_mask = orig_mask.copy()

        # Zero out rationale positions (except CLS and SEP)
        for i in range(len(new_mask)):
            if rationale_mask[i] == 1:  # This is a rationale
                # Don't mask if it's CLS or SEP
                if input_ids[i] not in self.special_token_ids:
                    new_mask[i] = 0

        return {
            "post_id": item["post_id"],
            "input_ids": torch.tensor(input_ids).unsqueeze(0),
            "attention_mask": torch.tensor(new_mask).unsqueeze(0),
            "rationales": item["rationales"],
            "hard_label": item["hard_label"],
        }

    def _create_sufficiency_instance(
        self, item: Dict, rationale_mask: np.ndarray
    ) -> Dict:
        """
        Create instance for sufficiency: Keep ONLY rationales in attention
        Keep: CLS + rationale tokens + SEP
        """
        input_ids = item["input_ids"].cpu().numpy().flatten()
        orig_mask = item["attention_mask"].cpu().numpy().flatten()

        # Start with zeros
        new_mask = np.zeros_like(orig_mask)

        # Always keep CLS and SEP
        for i in range(len(new_mask)):
            if input_ids[i] in self.special_token_ids:
                new_mask[i] = 1

        # Keep rationale positions
        for i in range(len(new_mask)):
            if rationale_mask[i] == 1 and orig_mask[i] == 1:
                new_mask[i] = 1

        return {
            "post_id": item["post_id"],
            "input_ids": torch.tensor(input_ids).unsqueeze(0),
            "attention_mask": torch.tensor(new_mask).unsqueeze(0),
            "rationales": item["rationales"],
            "hard_label": item["hard_label"]
        }