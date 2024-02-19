import torch
from torch.utils.data import Dataset
import numpy as np

class LocalDataset(Dataset):
    def __init__(self, file_path, context_size):
        self.data = np.memmap(file_path, dtype=np.uint16, mode='r')
        self.context_size = context_size

    def __len__(self):
        # Adjust for the context size to avoid overflow
        return len(self.data) - self.context_size - 1
    
    def __getitems__(self, idxs):
        elements = []
        for idx in idxs:
            x = torch.from_numpy((self.data[idx : idx + self.context_size]).astype(np.int64))
            y = torch.from_numpy((self.data[idx + 1 : idx + self.context_size + 1]).astype(np.int64))
            elements.append((x, y))
        return elements