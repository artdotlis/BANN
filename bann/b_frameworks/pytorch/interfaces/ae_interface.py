# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import abc
from typing import Tuple

import torch
from torch import nn


class AutoEncoderInterface(nn.Module, abc.ABC):
    @abc.abstractmethod
    def encode(self, *input_args: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def decode(self, *input_args: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('Interface!')

    @property
    @abc.abstractmethod
    def encoder(self) -> Tuple[Tuple[nn.Module, ...], Tuple[nn.Module, ...]]:
        raise NotImplementedError('Interface!')

    @property
    @abc.abstractmethod
    def decoder(self) -> Tuple[Tuple[nn.Module, ...], Tuple[nn.Module, ...]]:
        raise NotImplementedError('Interface!')
