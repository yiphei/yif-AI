from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler


# A DistributedSampler that works with both distributed and non-distributed training
class CustomDistributedSampler(DistributedSampler):
    # there is prob a better way to do this, but too lazy. The only new param is batch_size
    def __init__(
        self,
        dataset: Dataset,
        batch_size,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.batch_size = batch_size

    def __iter__(self):
        idx_range_per_process = len(self.dataset) // self.num_replicas
        start_idx = idx_range_per_process * self.rank
        end_idx = min(start_idx + idx_range_per_process, len(self.dataset))
        rand_idxs_iter = iter(
            torch.randint(start_idx, end_idx, (self.batch_size,)).tolist()
        )
        curr = 0
        while True:
            curr += 1
            next_idx = next(rand_idxs_iter)
            if curr >= self.batch_size:
                curr = 0
                rand_idxs_iter = iter(
                    torch.randint(start_idx, end_idx, (self.batch_size,)).tolist()
                )
            yield next_idx


class MapLocalDataset(Dataset):
    def __init__(self, file_path, context_size):
        self.data = np.memmap(file_path, dtype=np.uint16, mode="r")
        self.context_size = context_size

    def __len__(self):
        return len(self.data) - self.context_size - 1

    def __getitems__(self, idxs):
        x = torch.stack([torch.from_numpy(
            (self.data[idx : idx + self.context_size]).astype(np.int64)
        ) for idx in idxs])
        y = torch.stack([torch.from_numpy(
            (self.data[idx + 1 : idx + self.context_size + 1]).astype(np.int64)
        ) for idx in idxs])
        return x, y

    @classmethod
    def create_with_distributed_sampler(
        cls, file_path, context_size, batch_size, using_DDP
    ):
        dataset = cls(file_path, context_size)
        additional_args = {"rank": 0, "num_replicas": 1} if not using_DDP else {}
        sampler = CustomDistributedSampler(dataset, batch_size, **additional_args)
        return dataset, sampler
