import random
from contextlib import contextmanager, nullcontext
from enum import StrEnum, EnumMeta
from dataclasses import field
import numpy as np
import torch
from typing import Type, Any

class AutoMappedEnumMeta(EnumMeta):
    def __new__(metacls, cls, bases, classdict):
        enum_class = super().__new__(metacls, cls, bases, classdict)
        enum_class._int_to_enum = {i: item for i, item in enumerate(enum_class, start=1)}
        return enum_class

    def from_int(cls, num):
        if num in cls._int_to_enum:
            return cls._int_to_enum[num]
        else:
            raise ValueError(f"Invalid integer for {cls.__name__}: {num}")

class AutoMappedEnum(StrEnum, metaclass=AutoMappedEnumMeta):
    pass


def dataclass_enum_field(enum_class: Type[AutoMappedEnum]):
    class EnumConverter:
        _value: Any

        def __get__(self, instance, owner):
            return self._value

        def __set__(self, instance, value):
            if isinstance(value, int):
                self._value = enum_class.from_int(value)
            elif isinstance(value, enum_class):
                self._value = value
            else:
                raise TypeError(f"Expected int or {enum_class.__name__}, got {type(value).__name__}")

    return field(default=enum_class(next(iter(enum_class._int_to_enum.values()))), metadata={'converter': EnumConverter()})


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
