# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import Tuple, Optional, Iterable

from torch import randperm
from torch.utils import data
from torch.utils.data import Dataset

from bann.b_data_functions.pytorch.shared_memory_interface import DataSetSharedMemoryA


def _generate_split(data_sets: Tuple[Dataset, ...], subset_size: int) -> Iterable[Dataset]:
    for data_set in data_sets:
        if isinstance(data_set, DataSetSharedMemoryA):
            # based on random_split from torch
            indices = randperm(len(data_set)).tolist()
            yield data_set.create_subsets(indices[0:subset_size])
        else:
            yield data.random_split(
                data_set, [subset_size, len(data_set) - subset_size]
            )[0]


def create_subsets(subset_size: Optional[int],
                   data_sets: Tuple[Dataset, ...], /) -> Tuple[Dataset, ...]:
    if subset_size is None:
        return data_sets
    len_sub = 1 if subset_size < 2 else subset_size
    for data_set in data_sets:
        if len_sub > len(data_set):
            len_sub = len(data_set)
    return tuple(_generate_split(data_sets, len_sub))


def check_subset_size(subset: str, /) -> Optional[int]:
    try:
        res = int(subset)
        return res if res >= 1 else None
    except TypeError:
        return None
