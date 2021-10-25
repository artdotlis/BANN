# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import abc
from typing import Tuple, Union, Type, Dict, Generator, List, final, Final
from dataclasses import dataclass

from bann.b_container.functions.compare_min_max import CompareNumElem, AlwaysCompareNumElem
from bann.b_container.states.general.interface.hyper_optim_state import OptimHyperState
from rewowr.public.interfaces.logger_interface import SyncStdoutInterface


class HyperParamInterface(abc.ABC):

    @abc.abstractmethod
    def get_hyper_param(self) -> Tuple[float, ...]:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def set_new_hyper_param(self, params: Tuple[float, ...], /) -> None:
        raise NotImplementedError('Interface!')

    @property
    @abc.abstractmethod
    def min_values(self) -> Tuple[CompareNumElem, ...]:
        raise NotImplementedError('Interface!')

    @property
    @abc.abstractmethod
    def max_values(self) -> Tuple[CompareNumElem, ...]:
        raise NotImplementedError('Interface!')

    @property
    @abc.abstractmethod
    def type_values(self) -> Tuple[Union[Type[int], Type[float]], ...]:
        raise NotImplementedError('Interface!')

    @property
    @abc.abstractmethod
    def hyper_min_values(self) -> Tuple[AlwaysCompareNumElem, ...]:
        raise NotImplementedError('Interface!')

    @property
    @abc.abstractmethod
    def hyper_max_values(self) -> Tuple[AlwaysCompareNumElem, ...]:
        raise NotImplementedError('Interface!')


@final
@dataclass
class HyperOptimInterfaceArgs:
    state_type: Dict[str, str]
    hyper_args: Dict[str, Tuple[float, ...]]
    hyper_max_args: Dict[str, Tuple[AlwaysCompareNumElem, ...]]
    hyper_min_args: Dict[str, Tuple[AlwaysCompareNumElem, ...]]
    min_max_types: Dict[str, Tuple[Union[Type[float], Type[int]], ...]]


@final
@dataclass
class HyperOptimReturnElem:
    param: Tuple[float, ...]
    state_type: str


HGenTA: Final = Generator[
    List[Dict[str, HyperOptimReturnElem]],
    List[Tuple[float, Dict[str, HyperOptimReturnElem]]],
    None
]


class HyperOptimInterface(abc.ABC):

    @abc.abstractmethod
    def hyper_optim(self, sync_out: SyncStdoutInterface, args: HyperOptimInterfaceArgs, /) \
            -> HGenTA:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def set_hyper_state(self, state: OptimHyperState, /) -> None:
        raise NotImplementedError('Interface!')
