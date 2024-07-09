import random
from contextlib import contextmanager, nullcontext
from enum import StrEnum
from dataclasses import dataclass, fields
import numpy as np
import torch
from typing import Type, get_type_hints

class AutoMappedEnum(StrEnum):
    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.int_mapping = {i: member for i, member in enumerate(cls, start=1)}

class EnumFieldDescriptor:
    def __init__(self, enum_class: Type[AutoMappedEnum]):
        self.enum_class = enum_class
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name)

    def __set__(self, instance, value):
        if isinstance(value, int):
            value = self.enum_class.int_mapping[value]
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
