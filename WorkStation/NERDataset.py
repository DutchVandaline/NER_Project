import torch
from torch.utils.data import Dataset

class NERDataset(Dataset):
    def __init__(self, input_ids, attention_masks, label_ids):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.label_ids = label_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_masks[idx], dtype=torch.long),
            'labels': torch.tensor(self.label_ids[idx], dtype=torch.long)
        }
