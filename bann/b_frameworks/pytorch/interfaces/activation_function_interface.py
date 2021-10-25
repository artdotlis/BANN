# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import abc
from dataclasses import dataclass
from typing import Optional, final, Dict

import torch


@final
@dataclass
class AfcDataTC:
    data: torch.Tensor
    dim: Optional[int] = None
    e_params: Optional[Dict] = None


class AFClassInterface(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def act(input_val: AfcDataTC, /) -> torch.Tensor:
        raise NotImplementedError("Interface")

    @staticmethod
    @abc.abstractmethod
    def activation_fun_name() -> str:
        raise NotImplementedError("Interface")
