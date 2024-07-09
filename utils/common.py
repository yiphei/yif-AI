import random
from contextlib import contextmanager, nullcontext
from enum import StrEnum
from dataclasses import dataclass, fields
import numpy as np
import torch

class AutoMappedEnum(StrEnum):
    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.int_mapping = {i: member for i, member in enumerate(cls, start=1)}

# this doesnt work
def auto_enum_dataclass(cls):
    cls = dataclass(cls)  # Apply the dataclass decorator first
    original_post_init = getattr(cls, '__post_init__', None)

    def __post_init__(self):
        for f in fields(self):
            if issubclass(f.type, AutoMappedEnum):
                value = getattr(self, f.name)
                if isinstance(value, int):
                    setattr(self, f.name, f.type.from_int(value))
                elif not isinstance(value, f.type):
                    raise TypeError(f"Expected int or {f.type.__name__}, got {type(value).__name__}")
        
        if original_post_init is not None:
            original_post_init(self)

    setattr(cls, '__post_init__', __post_init__)
    return cls


def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    # NB: "mps" introduces non-deterministic behavior, despite explicitly setting random seeds.
    return "mps" if torch.backends.mps.is_available() else "cpu"


def set_random_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def create_autocast_context(device_type, ptdtype):
    @contextmanager
    def autocast_context():
        ctx = (
            nullcontext()
            if device_type == "cpu"
            else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        )
        with ctx:
            yield

    return autocast_context
