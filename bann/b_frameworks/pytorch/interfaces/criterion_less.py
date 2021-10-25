# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import abc
from typing import TypeVar, Generic

import torch

_CritIn = TypeVar('_CritIn')
_CritOut = TypeVar('_CritOut')


class CriterionLess(Generic[_CritIn, _CritOut], abc.ABC):
    @abc.abstractmethod
    def criterion(self, output_d: _CritOut, input_d: _CritIn, /) -> torch.Tensor:
        raise NotImplementedError('Interface!')

    @staticmethod
    @abc.abstractmethod
    def criterion_str() -> str:
        raise NotImplementedError('Interface!')
