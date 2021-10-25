# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import abc
import re
from typing import Any, TypeVar, Pattern, Generic, Dict, Type, Final, final
from dataclasses import dataclass

from bann.b_container.constants.fr_string import StateNLib


@dataclass(init=False)
class TestStateKwargs:
    def __init__(self, **_: Any) -> None:  # type: ignore
        super().__init__()


_TestStateType = TypeVar('_TestStateType', bound=TestStateKwargs)
_TestPrePattern: Final[Pattern[str]] = re.compile(f"^{StateNLib.TE.value}(.+)$")


class TestState(Generic[_TestStateType], abc.ABC):

    @abc.abstractmethod
    def get_kwargs_repr(self) -> str:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def get_kwargs(self) -> _TestStateType:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def set_kwargs(self, args_dict: Dict, /) -> None:
        raise NotImplementedError('Interface!')

    @staticmethod
    @final
    def get_pre_arg() -> str:
        return StateNLib.TE.value

    @staticmethod
    @final
    def parse_dict(data_con: Type[_TestStateType], args_dict: Dict, /) -> _TestStateType:
        new_dict = {}
        for key, value in args_dict.items():
            found = _TestPrePattern.search(key)
            if found is not None:
                new_dict[found.group(1)] = value

        return data_con(**new_dict)
