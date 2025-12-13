from typing import List, Dict
import torch
from torch.utils.data import Dataset

class HateDataset(Dataset):
    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        return {
            "input_ids": item["input_ids"].flatten(),
            "attention_mask": item["attention_mask"].flatten(),
            "rationales": item["rationales"].flatten(),
            "labels": torch.tensor(item["hard_label"], dtype=torch.long),
        }