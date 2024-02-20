import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DistributedSampler, IterableDataset
from torchdata.datapipes.iter import Shuffler


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
    def create_with_distributed_sampler(cls, file_path, context_size, using_DDP):
        dataset = cls(file_path, context_size)
        sampler = None
        if using_DDP:
            sampler = DistributedSampler(dataset)
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
