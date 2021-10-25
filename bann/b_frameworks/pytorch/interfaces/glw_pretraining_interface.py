# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import abc
from typing import Iterable, Tuple

import torch
from torch import nn


class GLWPNetInterface(abc.ABC):

    @abc.abstractmethod
    def prepare_input(self, layer: int, in_data: Tuple[torch.Tensor, ...], /) \
            -> Tuple[torch.Tensor, ...]:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def prepare_output(self, layer: int, in_data: Tuple[torch.Tensor, ...], /) -> torch.Tensor:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def prepare_target(self, layer: int, in_data: torch.Tensor, /) -> torch.Tensor:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def get_stack(self) -> Iterable[nn.Module]:
        raise NotImplementedError('Interface!')

    @property
    @abc.abstractmethod
    def get_stack_first(self) -> nn.Module:
        raise NotImplementedError('Interface!')
