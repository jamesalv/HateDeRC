from typing import Any, Dict, List, Tuple
import numpy as np


def get_bias_evaluation_samples(data, method, group):
    """
    Get positive and negative sample IDs for bias evaluation based on method and group

    Args:
        data: list of data entries
        method: Bias evaluation method ('subgroup', 'bpsn', or 'bnsp')
        group: Target group to evaluate

    Returns:
        Tuple of (positive_ids, negative_ids)
    """
    positive_ids = []
    negative_ids = []

    for idx, row in enumerate(data):
        target_groups = row["target_groups"]
        if target_groups is None:
            continue

        is_in_group = group in target_groups

        # Convert various label formats to binary toxic/non-toxic
        if "hard_label" in row:
            is_toxic = row["hard_label"] == 1
        else:
            continue

        if method == "subgroup":
            # Only consider samples mentioning the group
            if is_in_group:
                if is_toxic:
                    positive_ids.append(idx)
                else:
                    negative_ids.append(idx)

        elif method == "bpsn":
            # Compare non-toxic posts mentioning the group with toxic posts NOT mentioning the group
            if is_in_group and not is_toxic:
                negative_ids.append(idx)
            elif not is_in_group and is_toxic:
                positive_ids.append(idx)

        elif method == "bnsp":
            # Compare toxic posts mentioning the group with non-toxic posts NOT mentioning the group
            if is_in_group and is_toxic:
                positive_ids.append(idx)
            elif not is_in_group and not is_toxic:
                negative_ids.append(idx)

    return positive_ids, negative_ids


from collections import defaultdict
from sklearn.metrics import roc_auc_score


def calculate_gmb_metrics(
    test_data: List[Dict[str, Any]],
    probabilities: np.ndarray,
    target_groups: List[str],
    classification_mode: str = "binary",
):
    """
    Calculate GMB (Generalized Mean of Bias) AUC metrics from model predictions

    Note: Bias evaluation always uses binary classification (toxic vs non-toxic).
    If classification_mode is "multiclass", hatespeech and offensive are combined into one toxic class.

    Args:
        probabilities: Model's probability outputs (shape: [n_samples, n_classes])
        test_data: List of test data entries
        target_groups: List of target groups to evaluate
        classification_mode: "binary" or "multiclass"

    Returns:
        Dictionary with GMB metrics
    """
    # Create mappings from post_id to predictions and ground truth
    prediction_scores = defaultdict(lambda: defaultdict(dict))
    ground_truth = {}

    for idx, row in enumerate(test_data):
        # Convert predictions to binary (toxic vs non-toxic)
        if classification_mode == "multiclass":
            # For 3-class: P(toxic) = P(hatespeech) + P(offensive)
            binary_prob = probabilities[idx, 1] + probabilities[idx, 2]
            prediction_scores[idx] = binary_prob
            # Convert label: 0=normal, 1/2=toxic
            ground_truth[idx] = 1 if row["hard_label"] > 0 else 0
        else:
            # For binary: use probability of class 1 (toxic)
            prediction_scores[idx] = probabilities[idx, 1]
            ground_truth[idx] = row["hard_label"]

    # Calculate metrics for each target group and method
    bias_metrics = {}
    methods = ["subgroup", "bpsn", "bnsp"]

    for method in methods:
        bias_metrics[method] = {}  # Initialize nested dictionary for each method
        for group in target_groups:
            # Get positive and negative samples based on the method
            positive_ids, negative_ids = get_bias_evaluation_samples(
                test_data, method, group
            )

            if len(positive_ids) == 0 or len(negative_ids) == 0:
                print(f"Skipping {method} for group {group}: no samples found")
                continue  # Skip if no samples for this group/method

            # Collect ground truth and predictions
            y_true = []
            y_score = []

            for post_id in positive_ids:
                if post_id in ground_truth and post_id in prediction_scores:
                    y_true.append(ground_truth[post_id])
                    y_score.append(prediction_scores[post_id])

            for post_id in negative_ids:
                if post_id in ground_truth and post_id in prediction_scores:
                    y_true.append(ground_truth[post_id])
                    y_score.append(prediction_scores[post_id])

            # Calculate AUC if we have enough samples with both classes
            if len(y_true) > 10 and len(set(y_true)) > 1:
                try:
                    auc = roc_auc_score(y_true, y_score)
                    bias_metrics[method][group] = auc
                except ValueError:
                    print(
                        f"Could not compute AUC for {method} and group {group} due to ValueError"
                    )
                    pass

    # Calculate GMB for each method
    gmb_metrics = {}
    power = -5  # Power parameter for generalized mean

    for method in methods:
        if not bias_metrics[method]:
            continue

        scores = list(bias_metrics[method].values())
        if not scores:
            continue

        # Calculate generalized mean with p=-5
        power_mean = np.mean([score**power for score in scores]) ** (1 / power)
        gmb_metrics[f"GMB-{method.upper()}-AUC"] = power_mean

    # Calculate a combined GMB score that includes all methods
    all_scores = []
    for method in methods:
        all_scores.extend(list(bias_metrics[method].values()))

    if all_scores:
        gmb_metrics["GMB-COMBINED-AUC"] = np.mean(
            [score**power for score in all_scores]
        ) ** (1 / power)

    return gmb_metrics, bias_metrics
