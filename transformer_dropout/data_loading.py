from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DistributedSampler, IterableDataset
from torchdata.datapipes.iter import Shuffler


class CustomDistributedSampler(DistributedSampler):

    # there is a better way to do this, but too lazy. The only new param is batch_size
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
        idx_window_per_process = len(self.dataset) // self.num_replicas
        start_idx = idx_window_per_process * self.rank
        end_idx = min(start_idx + idx_window_per_process, len(self.dataset))
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
        elements = []
        for idx in idxs:
            x = torch.from_numpy(
                (self.data[idx : idx + self.context_size]).astype(np.int64)
            )
            y = torch.from_numpy(
                (self.data[idx + 1 : idx + self.context_size + 1]).astype(np.int64)
            )
            elements.append((x, y))
        return elements

    @classmethod
    def create_with_distributed_sampler(
        cls, file_path, context_size, batch_size, using_DDP
    ):
        dataset = cls(file_path, context_size)
        additional_args = {"rank": 0, "num_replicas": 1} if not using_DDP else {}
        sampler = CustomDistributedSampler(dataset, batch_size, **additional_args)
        return dataset, sampler


class IterableLocalDataset(IterableDataset):
    def __init__(self, file_path, context_size):
        self.data = np.memmap(file_path, dtype=np.uint16, mode="r")
        self.context_size = context_size

    def __len__(self):
        return len(self.data) - self.context_size

    def __iter__(self):
        for idx in range(len(self.data) - self.context_size):
            x = torch.from_numpy(
                (self.data[idx : idx + self.context_size]).astype(np.int64)
            )
            y = torch.from_numpy(
                (self.data[idx + 1 : idx + self.context_size + 1]).astype(np.int64)
            )
            yield x, y

    @classmethod
    def create_with_shuffler(cls, file_path, context_size, buffer_size):
        dataset = cls(file_path, context_size)
        dataset = Shuffler(dataset, buffer_size=buffer_size)
        return dataset


class DistributedIterableLocalDataset(IterableDataset):
    def __init__(self, file_path, context_size):
        self.data = np.memmap(file_path, dtype=np.uint16, mode="r")
        self.context_size = context_size

    def __iter__(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        total_size = len(self.data) - self.context_size
        per_process = int(np.ceil(total_size / float(world_size)))
        worker_start = rank * per_process
        worker_end = min(worker_start + per_process, total_size)

        for idx in range(worker_start, worker_end):
            x = torch.from_numpy(
                self.data[idx : idx + self.context_size].astype(np.int64)
            )
            y = torch.from_numpy(
                self.data[idx + 1 : idx + self.context_size + 1].astype(np.int64)
            )
            yield x, y

    @classmethod
    def create_with_shuffler(cls, file_path, context_size, buffer_size):
        dataset = cls(file_path, context_size)
        dataset = Shuffler(dataset, buffer_size=buffer_size)
        return dataset
