import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class MedsDataset(Dataset):
    """Medications mention dataset"""

    def __init__(self, data_path, max_seq_length, num_lines=None):
        """
        Args:
            data_path (string): Path to the file containg texts with labels.
        """
        self.df = pd.read_csv(data_path, header=None)
        if num_lines:
            self.df = self.df[:num_lines]
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
            Output:
            text: 
            attention_mask: 
            label: 
        """
        label = self.df.iloc[idx, 0]
        text = self.df.iloc[idx, 1]
        tokens = tokenizer.encode(text, add_special_tokens=True)
        padded = np.array(tokens + [0]*(self.max_seq_length-len(tokens)))
        text = torch.tensor(padded)
        attention_mask = np.where(padded != 0, 1, 0)
        attention_mask = torch.tensor(attention_mask)
        return text, attention_mask, label


