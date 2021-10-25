# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import abc
from typing import TypeVar, Generic

import torch

from bann.b_frameworks.pytorch.p_truth.p_truth_fun_arg_types import TruthFunArgCon, \
    TruthFunNoDevCon

_TruthKey = TypeVar('_TruthKey', TruthFunArgCon, TruthFunNoDevCon)
_TruthReturn = TypeVar('_TruthReturn', int, float)


class TruthClassInterface(Generic[_TruthKey, _TruthReturn], abc.ABC):

    @abc.abstractmethod
    def calc_truth(self, input_val: _TruthKey, /) -> _TruthReturn:
        raise NotImplementedError("Interface")

    @abc.abstractmethod
    def cr_truth_container(self, output: torch.Tensor, target: torch.Tensor,
                           device: torch.device, /) -> _TruthKey:
        raise NotImplementedError("Interface")

    @staticmethod
    @abc.abstractmethod
    def truth_name() -> str:
        raise NotImplementedError("Interface")
