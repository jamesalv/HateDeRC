import re
import string
import numpy as np
import torch


def deobfuscate_text(text):
    """
    Normalize common text obfuscation patterns to reveal original words.
    Useful for hate speech detection and content analysis.

    Args:
        text (str): Input text with potential obfuscations

    Returns:
        str: Text with obfuscations normalized
    """
    if not isinstance(text, str):
        return text

    # Make a copy to work with
    result = text.lower()

    # 1. Handle asterisk/symbol replacements
    symbol_patterns = {
        # Common profanity
        r"f\*+c?k": "fuck",
        r"f\*+": "fuck",
        r"s\*+t": "shit",
        r"b\*+ch": "bitch",
        r"a\*+s": "ass",
        r"d\*+n": "damn",
        r"h\*+l": "hell",
        r"c\*+p": "crap",
        # Slurs and hate speech terms (be comprehensive for detection)
        r"n\*+g+[aer]+": "nigger",  # Various n-word obfuscations
        r"f\*+g+[ot]*": "faggot",
        r"r\*+[dt]ard": "retard",
        r"sp\*+c": "spic",
        # Other symbols
        r"@ss": "ass",
        r"b@tch": "bitch",
        r"sh!t": "shit",
        r"f#ck": "fuck",
        r"d@mn": "damn",
    }

    for pattern, replacement in symbol_patterns.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # 2. Handle character spacing (f u c k -> fuck)
    spacing_patterns = {
        r"\bf\s+u\s+c\s+k\b": "fuck",
        r"\bs\s+h\s+i\s+t\b": "shit",
        r"\bd\s+a\s+m\s+n\b": "damn",
        r"\bh\s+e\s+l\s+l\b": "hell",
        r"\ba\s+s\s+s\b": "ass",
        r"\bc\s+r\s+a\s+p\b": "crap",
    }

    for pattern, replacement in spacing_patterns.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # 3. Handle number/letter substitutions
    leet_patterns = {
        # Basic leet speak
        r"\b3\s*1\s*1\s*3\b": "elle",  # 3113 -> elle
        r"\bf4g\b": "fag",
        r"\bf4gg0t\b": "faggot",
        r"\bn00b\b": "noob",
        r"\bl33t\b": "leet",
        r"\bh4t3\b": "hate",
        r"\b5h1t\b": "shit",
        r"\bf0ck\b": "fock",
    }

    for pattern, replacement in leet_patterns.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # 4. Handle repeated characters and separators
    # Remove excessive punctuation between letters
    result = re.sub(r"([a-z])[^\w\s]+([a-z])", r"\1\2", result)

    # Handle underscore separation
    result = re.sub(r"([a-z])_+([a-z])", r"\1\2", result)

    # Handle dot separation
    result = re.sub(r"([a-z])\.+([a-z])", r"\1\2", result)

    # 5. Handle common misspellings/variations used for evasion
    evasion_patterns = {
        r"\bfuk\b": "fuck",
        r"\bfuq\b": "fuck",
        r"\bfck\b": "fuck",
        r"\bshyt\b": "shit",
        r"\bshit\b": "shit",
        r"\bbiatch\b": "bitch",
        r"\bbeatch\b": "bitch",
        r"\basshole\b": "asshole",
        r"\ba55hole\b": "asshole",
        r"\btard\b": "retard",
        r"\bfagg\b": "fag",
    }

    for pattern, replacement in evasion_patterns.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # 6. Clean up multiple spaces
    result = re.sub(r"\s+", " ", result).strip()

    return result


def aggregate_rationales(rationales, labels, post_length, drop_abnormal=False):
    """
    If all 3 annotators are normal → 3 zero spans → average (all zeros).
    If k annotators are non-normal and k spans exist → average the k spans (no added zeros).
    If k non-normal but fewer than k spans:
        If the missing annotators are non-normal → do not fill with zeros; average only existing spans and record rationale_support = #spans.
        If the missing annotators are normal (e.g., 2 hate + 1 normal + 2 spans) → append one zero span for the normal.
    """
    count_normal = labels.count(0)
    count_hate = labels.count(1)
    count_rationales = len(rationales)
    pad = np.zeros(post_length, dtype="int").tolist()

    # If there are hate labels but no rationales, something is wrong
    if count_hate > 0 and count_rationales == 0:
        if drop_abnormal:
            return None

        # Else just fill with 0
        return np.zeros(post_length).tolist()

    # If all annotators are normal, return all zeros
    if count_normal == 3:
        return np.zeros(post_length).tolist()

    # If we have hate annotators
    if count_hate > 0:
        # Case 1: Number of rationales matches number of hate annotators
        if count_rationales == count_hate:
            return np.average(rationales, axis=0).tolist()

        # Case 2: Fewer rationales than hate annotators
        elif count_rationales < count_hate:
            # Add zero padding for normal annotators only
            rationales_copy = rationales.copy()
            zeros_to_add = count_normal
            for _ in range(zeros_to_add):
                rationales_copy.append(pad)
            return np.average(rationales_copy, axis=0).tolist()

        # Case 3: More rationales than hate annotators (shouldn't happen normally)
        else:
            # Just average what we have
            return np.average(rationales, axis=0).tolist()

    # Fallback: return zeros if no clear case matches
    return np.zeros(post_length).tolist()


from typing import List, Tuple


def preprocess_text(raw_text):
    preprocessed_text = raw_text
    # # Remove HTML tags <>
    preprocessed_text = re.sub(r'<[^>]*>', '', preprocessed_text)
    # # De-Obsfucate Patterns
    preprocessed_text = deobfuscate_text(preprocessed_text)

    return preprocessed_text


def create_text_segment(
    text_tokens: List[str], rationale_mask: List[int]
) -> List[Tuple[List[str], int]]:
    """
    Process a rationale mask to identify contiguous segments of highlighted text.
    Then create a segmented representation of the tokens

    Args:
        text_tokens: Original text tokens
        mask: Binary mask where 1 indicates a highlighted token (this consists of mask from 3 annotators)

    Returns:
        A list of tuples (text segment, mask value)
    """
    # Handle case where mask is empty (no rationale provided), usually this is normal classification
    mask = rationale_mask

    # for mask in all_rationale_mask:
    # Find breakpoints (transitions between highlighted/1 and non-highlighted/0)
    breakpoints = []
    mask_values = []

    # Always start with position 0
    breakpoints.append(0)
    mask_values.append(mask[0])

    # Find transitions in the mask
    for i in range(1, len(mask)):
        if mask[i] != mask[i - 1]:
            breakpoints.append(i)
            mask_values.append(mask[i])

    # Always end with the length of the text
    if breakpoints[-1] != len(mask):
        breakpoints.append(len(mask))

    # Create segments based on breakpoints
    segments = []
    for i in range(len(breakpoints) - 1):
        start = breakpoints[i]
        end = breakpoints[i + 1]
        segments.append((text_tokens[start:end], mask_values[i]))

    return segments


def align_rationales(tokens, rationales, tokenizer, max_length=128):
    """
    Align rationales with tokenized text while handling different tokenizer formats.

    Args:
        tokens: Original text tokens
        rationales: Original rationale masks
        tokenizer: The tokenizer to use
        max_length: Maximum sequence length

    Returns:
        Dictionary with tokenized inputs and aligned rationale masks
    """
    segments = create_text_segment(tokens, rationales)
    all_human_rationales = []
    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_rationales = []
    for text_segment, rationale_value in segments:
        inputs = {}
        concatenated_text = " ".join(text_segment)
        processed_segment = preprocess_text(concatenated_text)
        tokenized = tokenizer(
            processed_segment, add_special_tokens=False, return_tensors="pt"
        )

        # Extract the relevant data
        segment_input_ids = tokenized["input_ids"][0]
        segment_attention_mask = tokenized["attention_mask"][0]
        # Handle token_type_ids if present
        if "token_type_ids" in tokenized:
            segment_token_type_ids = tokenized["token_type_ids"][0]
            all_token_type_ids.extend(segment_token_type_ids)

        # Add input IDs and attention mask
        all_input_ids.extend(segment_input_ids)
        all_attention_mask.extend(segment_attention_mask)

        # Add rationales (excluding special tokens)
        segment_rationales = [rationale_value] * len(segment_input_ids)
        all_rationales.extend(segment_rationales)
    # Get special token IDs
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id

    # Add special tokens at the beginning and end
    all_input_ids = [cls_token_id] + all_input_ids + [sep_token_id]
    all_attention_mask = [1] + all_attention_mask + [1]

    # Handle token_type_ids if the model requires it
    if hasattr(tokenizer, "create_token_type_ids_from_sequences"):
        all_token_type_ids = tokenizer.create_token_type_ids_from_sequences(
            all_input_ids[1:-1]
        )
    elif all_token_type_ids:
        all_token_type_ids = [0] + all_token_type_ids + [0]
    else:
        all_token_type_ids = [0] * len(all_input_ids)

    # Check tokenized vs rationales length
    if len(all_input_ids) != len(all_attention_mask):
        print("Warning: length of tokens and rationales do not match")

    # Add zero rationale values for special tokens
    all_rationales = [0] + all_rationales + [0]

    # Truncate to max length if needed
    if len(all_input_ids) > max_length:
        print("WARNING: NEED TO TRUNCATE")
        all_input_ids = all_input_ids[:max_length]
        all_attention_mask = all_attention_mask[:max_length]
        all_token_type_ids = all_token_type_ids[:max_length]
        all_rationales = all_rationales[:max_length]

    # Pad to max_length if needed
    pad_token_id = tokenizer.pad_token_id
    padding_length = max_length - len(all_input_ids)

    if padding_length > 0:
        all_input_ids = all_input_ids + [pad_token_id] * padding_length
        all_attention_mask = all_attention_mask + [0] * padding_length
        all_token_type_ids = all_token_type_ids + [0] * padding_length
        all_rationales = all_rationales + [0] * padding_length

    # Convert lists to tensors
    inputs = {
        "input_ids": torch.tensor([all_input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([all_attention_mask], dtype=torch.long),
        "token_type_ids": (
            torch.tensor([all_token_type_ids], dtype=torch.long)
            if "token_type_ids" in tokenizer.model_input_names
            else None
        ),
        "rationales": torch.tensor([all_rationales], dtype=torch.float32),
    }

    # Remove None values
    inputs = {k: v for k, v in inputs.items() if v is not None}
    return inputs


import re
import json
import os
import string
from collections import Counter
from tqdm import tqdm
import more_itertools as mit


def find_ranges(iterable):
    """Yield range of consecutive numbers."""
    for group in mit.consecutive_groups(iterable):
        group = list(group)
        if len(group) == 1:
            yield group[0]
        else:
            yield group[0], group[-1]


def process_and_convert_data(
    data,
    tokenizer,
    post_id_divisions,
    save_path="Data/explanations/",
    drop_abnormal=False,
    classification_mode="binary",
):
    """
    Combined function that processes raw entries and converts to ERASER format in one pass.
    Also splits data into train/val/test sets.

    Args:
        classification_mode: "binary" (normal vs toxic) or "multiclass" (normal vs hatespeech vs offensive)
    """
    print("Processing and converting data...")

    # Initialize outputs
    train_data = []
    val_data = []
    test_data = []
    dropped = 0

    # Create directories if saving splits
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, "docs"), exist_ok=True)
        train_fp = open(os.path.join(save_path, "train.jsonl"), "w")
        val_fp = open(os.path.join(save_path, "val.jsonl"), "w")
        test_fp = open(os.path.join(save_path, "test.jsonl"), "w")

    for key, value in tqdm(data.items()):
        try:
            # Extract labels based on classification mode
            if classification_mode == "binary":
                # Binary: 0=normal, 1=toxic (hatespeech or offensive)
                labels = [
                    1 if annot["label"] in ["hatespeech", "offensive"] else 0
                    for annot in value["annotators"]
                ]
            else:  # multiclass
                # 3-class: 0=normal, 1=hatespeech, 2=offensive
                label_map = {"normal": 0, "hatespeech": 1, "offensive": 2}
                labels = [
                    label_map.get(annot["label"], 0) for annot in value["annotators"]
                ]

            # Process rationales
            rationales = value.get("rationales", [])
            aggregated_rationale = aggregate_rationales(
                rationales,
                labels,
                len(value["post_tokens"]),
                drop_abnormal=drop_abnormal,
            )

            if aggregated_rationale is None:
                dropped += 1
                continue

            inputs = align_rationales(
                value["post_tokens"], aggregated_rationale, tokenizer
            )

            # Calculate labels
            hard_label = Counter(labels).most_common(1)[0][0]
            soft_label = sum(labels) / len(labels)

            # Determine target groups (mentioned at least 2 times)
            target_groups = [
                t for annot in value["annotators"] for t in annot["target"]
            ]
            filtered_targets = [k for k, v in Counter(target_groups).items() if v > 1]

            # Create processed entry
            processed_entry = {
                "post_id": key,
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "rationales": inputs["rationales"],
                "raw_text": " ".join(value["post_tokens"]),
                "hard_label": hard_label,
                "soft_label": soft_label,
                "target_groups": filtered_targets,
            }

            # Convert to ERASER format if it's hateful/offensive content
            if (hard_label == 1 or hard_label == 2) and save_path:
                input_ids_list = inputs["input_ids"].squeeze().tolist()
                rationales_list = inputs["rationales"].squeeze().ceil().int().tolist()

                # Build evidences
                evidences = []
                indexes = sorted(
                    [i for i, each in enumerate(rationales_list) if each == 1]
                )
                for span in find_ranges(indexes):
                    if isinstance(span, int):
                        start, end = span, span + 1
                    else:
                        start, end = span[0], span[1] + 1

                    evidences.append(
                        {
                            "docid": key,
                            "end_sentence": -1,
                            "end_token": end,
                            "start_sentence": -1,
                            "start_token": start,
                            "text": " ".join(
                                [str(x) for x in input_ids_list[start:end]]
                            ),
                        }
                    )

                eraser_entry = {
                    "annotation_id": key,
                    "classification": str(hard_label),
                    "evidences": [evidences],
                    "query": "What is the class?",
                    "query_type": None,
                }

                # Save document
                with open(os.path.join(save_path, "docs", key), "w") as fp:
                    fp.write(" ".join([str(x) for x in input_ids_list if x > 0]))

                # Write to appropriate split
                if key in post_id_divisions["train"]:
                    train_fp.write(json.dumps(eraser_entry) + "\n")
                elif key in post_id_divisions["val"]:
                    val_fp.write(json.dumps(eraser_entry) + "\n")
                elif key in post_id_divisions["test"]:
                    test_fp.write(json.dumps(eraser_entry) + "\n")

            # Add to appropriate split list
            if key in post_id_divisions["train"]:
                train_data.append(processed_entry)
            elif key in post_id_divisions["val"]:
                val_data.append(processed_entry)
            elif key in post_id_divisions["test"]:
                test_data.append(processed_entry)

        except Exception as e:
            dropped += 1
            print(f"Error processing {key}: {e}")

    if save_path:
        train_fp.close()
        val_fp.close()
        test_fp.close()

    print(
        f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}, Dropped: {dropped}"
    )

    return {"train": train_data, "val": val_data, "test": test_data}
