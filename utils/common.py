import random
from contextlib import contextmanager, nullcontext
from enum import StrEnum
from dataclasses import dataclass
import numpy as np
import torch
from typing import Type, get_type_hints

class AutoMappedEnum(StrEnum):
    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.int_mapping = {i: member for i, member in enumerate(cls, start=1)}


    @classmethod
    def from_int(cls, value: int):
        return cls.int_mapping[value]

class EnumFieldDescriptor:
    def __init__(self, enum_class: Type[AutoMappedEnum]):
        self.enum_class = enum_class

    def __set__(self, instance, value):
        if isinstance(value, int):
            value = self.enum_class.from_int(value)
        elif isinstance(value, str):
            value = self.enum_class(value)
        elif not isinstance(value, self.enum_class):
            raise ValueError(f"Value must be an integer, string, or an instance of {self.enum_class}")
        instance.__dict__[self.name] = value

def adapted_dataclass(_cls=None, **kwargs):
    def wrap(cls):
        cls = dataclass(cls, **kwargs)
        hints = get_type_hints(cls)
        
        for name, hint in hints.items():
            if isinstance(hint, type) and issubclass(hint, AutoMappedEnum):
                setattr(cls, name, EnumFieldDescriptor(hint))
        
        return cls

    if _cls is None:
        return wrap
    return wrap(_cls)

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
