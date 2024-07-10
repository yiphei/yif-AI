import random
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, fields
from enum import StrEnum
from typing import get_args, get_origin, get_type_hints

import numpy as np
import torch


class IntMappedEnum(StrEnum):
    """A custom enum that serves custom_dataclass logic"""

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.int_mapping = {i: member for i, member in enumerate(cls, start=1)}

    @classmethod
    def from_int(cls, value: int):
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
