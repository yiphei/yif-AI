import random
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum, EnumMeta
from typing import get_args, get_origin, get_type_hints

import numpy as np
import torch


class IntMappedEnumMeta(EnumMeta):
    def __init__(cls, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cls.int_mapping = {i: member for i, member in enumerate(cls, start=1)}


class IntMappedEnum(Enum, metaclass=IntMappedEnumMeta):
    """A custom enum that serves custom_dataclass logic"""

    def __str__(self):
        return str(self.value)

    @classmethod
    def from_int(cls, value: int):
        if value not in cls.int_mapping:
            raise ValueError(f"Invalid integer value for {cls.__name__}: {value}")

        return cls.int_mapping[value]


def custom_dataclass(_cls=None, **kwargs):
    """A dataclass wrapper that allows IntMappedEnum fields to be initialized, in addition to
    an enum member, with either an integer or string, and casts all of them to the corresponding
    enum member. This flexibility, especially the integer values, is useful for visualizing
    field values in wandb.
    """

    def wrap(cls):
        sub_post_init = getattr(cls, "__post_init__", None)

        def enum_post_init(self):
            hints = get_type_hints(cls)

            for field in fields(self):
                hint = hints[field.name]
                origin = get_origin(hint)
                actual_type = hint

                if origin is not None:
                    actual_args = get_args(hint)
                    if actual_args:
                        # Naively assume the first argument is the type of interest
                        actual_type = actual_args[0]

                if isinstance(actual_type, type) and issubclass(
                    actual_type, IntMappedEnum
                ):
                    value = getattr(self, field.name)
                    if isinstance(value, int):
                        setattr(self, field.name, actual_type.from_int(value))
                    elif isinstance(value, str):
                        setattr(self, field.name, actual_type(value))
                elif is_dataclass(actual_type):
                    value = getattr(self, field.name)
                    if isinstance(value, dict):
                        setattr(self, field.name, actual_type(**value))

            if sub_post_init:
                sub_post_init(self)

        setattr(cls, "__post_init__", enum_post_init)
        cls = dataclass(cls, **kwargs)
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
