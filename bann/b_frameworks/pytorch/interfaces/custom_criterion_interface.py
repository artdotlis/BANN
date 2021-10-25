# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import abc
from typing import Generic, TypeVar, Any

from torch.nn import Module

TCo = TypeVar('TCo', covariant=True)


class PCustomLoss(Generic[TCo], Module, abc.ABC):
    def __init__(self, reduction: str) -> None:
        super().__init__()
        self.reduction = reduction

    @abc.abstractmethod
    def forward(self, *input_args: Any) -> TCo:
        raise NotImplementedError("Abstract method!")
