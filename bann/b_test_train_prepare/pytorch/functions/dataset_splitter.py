# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import Tuple, Iterable, List

from torch import randperm
from torch.utils.data import Dataset, random_split

from bann.b_data_functions.pytorch.shared_memory_interface import DataSetSharedMemoryA
from bann.b_test_train_prepare.errors.custom_errors import KnownSplitterError


def test_k_folds_data_sets(k_folds: int, data: Tuple[Dataset, ...], /) -> None:
    if k_folds < 2:
        raise KnownSplitterError("k_folds can not be smaller than two!")
    for data_set in data:
        if len(data_set) < k_folds:
            raise KnownSplitterError("k_folds can not be bigger than the dataset!")


def _generate_split(data_set: Dataset, subset_sizes: List[int]) -> Iterable[Dataset]:
    if isinstance(data_set, DataSetSharedMemoryA):
        # based on random_split from torch
        indices = randperm(len(data_set)).tolist()
        run_sum = 0
        for step_size in subset_sizes:
            yield data_set.create_subsets(indices[run_sum:run_sum+step_size])
            run_sum += step_size
    else:
        yield from random_split(data_set, subset_sizes)


def split_data_set(k_folds: int, data: Dataset, /) -> Tuple[Dataset, ...]:
    data_len = len(data)
    if data_len < k_folds or k_folds < 2:
        raise KnownSplitterError("k_folds can not be smaller than two or bigger than the dataset!")

    step_size = int(data_len / k_folds)
    data_split_len = [step_size for _ in range(k_folds)]
    if data_len % k_folds != 0:
        data_split_len.append(data_len % k_folds)
    return tuple(_generate_split(data, data_split_len))
