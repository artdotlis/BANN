# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import abc
import re
from dataclasses import dataclass
from typing import Any, Type, TypeVar, Generic, Pattern, Dict, Final, final

from bann.b_container.errors.custom_erors import KnownOptimStateError
from bann.b_container.constants.fr_string import StateNLib
from bann.b_hyper_optim.hyper_optim_interface import HyperParamInterface


@dataclass(init=False)
class OptimStateKwargs:
    def __init__(self, **_: Any) -> None:  # type: ignore
        super().__init__()

    @property
    def get_optim_dict(self) -> Dict:
        return {}


_OptimStateType = TypeVar('_OptimStateType', bound=OptimStateKwargs)
_OptimPrePattern: Final[Pattern[str]] = re.compile(f"^{StateNLib.OPTIM.value}(.+)$")


class MainOptimSt(Generic[_OptimStateType], HyperParamInterface, abc.ABC):

    @abc.abstractmethod
    def get_kwargs(self) -> _OptimStateType:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def get_kwargs_repr(self, index: int, /) -> str:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def set_kwargs(self, args_dict: Dict, /) -> None:
        raise NotImplementedError('Interface!')

    @staticmethod
    @final
    def get_pre_arg() -> str:
        return StateNLib.OPTIM.value

    @staticmethod
    @final
    def parse_str(str_to_parse: str, /) -> str:
        found = _OptimPrePattern.search(str_to_parse)
        if found is None:
            raise KnownOptimStateError(f"Optim could not parse {str_to_parse}")
        return found.group(1)

    @staticmethod
    @final
    def parse_dict(data_con: Type[_OptimStateType], args_dict: Dict, /) -> _OptimStateType:
        new_dict = {}
        for key, value in args_dict.items():
            found = _OptimPrePattern.search(key)
            if found is not None:
                new_dict[found.group(1)] = value

        return data_con(**new_dict)
