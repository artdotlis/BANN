# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import abc
import re
from dataclasses import dataclass
from typing import Any, Final, TypeVar, Pattern, Generic, Dict, final, Type

from bann.b_container.constants.fr_string import StateNLib


@dataclass(init=False)
class PrepareStateKwargs:
    def __init__(self, **_: Any) -> None:  # type: ignore
        super().__init__()


_PrepareStateType = TypeVar('_PrepareStateType', bound=PrepareStateKwargs)
_PreparePrePattern: Final[Pattern[str]] = re.compile(f"^{StateNLib.PR.value}(.+)$")


class PrepareState(Generic[_PrepareStateType], abc.ABC):
    @abc.abstractmethod
    def get_kwargs_repr(self) -> str:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def get_kwargs(self) -> _PrepareStateType:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def set_kwargs(self, args_dict: Dict, /) -> None:
        raise NotImplementedError('Interface!')

    @staticmethod
    @final
    def get_pre_arg() -> str:
        return StateNLib.PR.value

    @staticmethod
    @final
    def parse_dict(data_con: Type[_PrepareStateType], args_dict: Dict, /) -> _PrepareStateType:
        new_dict = {}
        for key, value in args_dict.items():
            found = _PreparePrePattern.search(key)
            if found is not None:
                new_dict[found.group(1)] = value

        return data_con(**new_dict)
