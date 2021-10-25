# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import abc
from dataclasses import dataclass
from typing import TypeVar, Any, final, Tuple
from torch import nn

from pan.public.interfaces.pub_net_interface import NetSavInterface

_TypeBuffer = TypeVar('_TypeBuffer')


@final
@dataclass
class CurrentNetData:
    fitness: float
    com: nn.Module
    lego: nn.Module


@dataclass(init=False)
class InitContainer:
    def __init__(self, **_: Any) -> None:  # type: ignore
        super().__init__()


class NetModelInterface(
    NetSavInterface[nn.Module, CurrentNetData, _TypeBuffer, InitContainer], abc.ABC
):
    @abc.abstractmethod
    def redraw_current_net(self) -> None:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def merge_net_model(self, model: 'NetModelInterface', /) -> None:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def re_copy_current_net(self) -> None:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def re_init_current_net(self, new_net: CurrentNetData, /) -> None:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def hyper_update(self, data_up: Tuple[float, ...], /) -> None:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def update_current_net(self, fitness: float, /) -> None:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def reset_current_net(self) -> None:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def set_best_net(self) -> None:
        raise NotImplementedError('Interface!')
