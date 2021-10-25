# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import abc
from typing import Tuple

import torch


class SPPInterface(abc.ABC):
    @abc.abstractmethod
    def get_nodes_cnt(self) -> int:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def get_fm_cnt(self) -> int:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def get_bin_size_c(self) -> Tuple[int, ...]:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def spp(self, in_data: torch.Tensor, /) -> torch.Tensor:
        raise NotImplementedError('Interface!')
