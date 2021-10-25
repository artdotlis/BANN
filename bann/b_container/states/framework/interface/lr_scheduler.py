# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import abc
import re
from dataclasses import dataclass
from typing import Any, TypeVar, Generic, Type, Pattern, Dict, Final, final

from bann.b_container.constants.fr_string import StateNLib
from bann.b_hyper_optim.hyper_optim_interface import HyperParamInterface


@dataclass(init=False)
class LRSchedulerKwargs:
    def __init__(self, **_: Any) -> None:  # type: ignore
        super().__init__()

    @property
    def get_scheduler_dict(self) -> Dict:
        return {}


_LRSchedulerType = TypeVar('_LRSchedulerType', bound=LRSchedulerKwargs)
_LrSchPrePattern: Final[Pattern[str]] = re.compile(f"^{StateNLib.LRSCH.value}(.+)$")


class LRSchedulerState(Generic[_LRSchedulerType], HyperParamInterface, abc.ABC):

    @abc.abstractmethod
    def get_kwargs_repr(self, index: int, /) -> str:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def get_kwargs(self) -> _LRSchedulerType:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def set_kwargs(self, args_dict: Dict, /) -> None:
        raise NotImplementedError('Interface!')

    @staticmethod
    @final
    def get_pre_arg() -> str:
        return StateNLib.LRSCH.value

    @staticmethod
    @final
    def parse_dict(data_con: Type[_LRSchedulerType], args_dict: Dict, /) -> _LRSchedulerType:
        new_dict = {}
        for key, value in args_dict.items():
            found = _LrSchPrePattern.search(key)
            if found is not None:
                new_dict[found.group(1)] = value

        return data_con(**new_dict)
