# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import abc
import re
from dataclasses import dataclass
from typing import Any, Type, TypeVar, Generic, Dict, Pattern, Final, final

from bann.b_container.constants.fr_string import StateNLib
from bann.b_hyper_optim.hyper_optim_interface import HyperParamInterface


@dataclass(init=False)
class TrainStateKwargs:
    def __init__(self, **_: Any) -> None:  # type: ignore
        super().__init__()


_TrainStateType = TypeVar('_TrainStateType', bound=TrainStateKwargs)
_TrainPrePattern: Final[Pattern[str]] = re.compile(f"^{StateNLib.TR.value}(.+)$")


class TrainState(Generic[_TrainStateType], HyperParamInterface, abc.ABC):

    @abc.abstractmethod
    def get_kwargs_repr(self) -> str:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def get_kwargs(self) -> _TrainStateType:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def set_kwargs(self, args_dict: Dict, /) -> None:
        raise NotImplementedError('Interface!')

    @staticmethod
    @final
    def get_pre_arg() -> str:
        return StateNLib.TR.value

    @staticmethod
    @final
    def parse_dict(data_con: Type[_TrainStateType], args_dict: Dict, /) -> _TrainStateType:
        new_dict = {}
        for key, value in args_dict.items():
            found = _TrainPrePattern.search(key)
            if found is not None:
                new_dict[found.group(1)] = value

        return data_con(**new_dict)
