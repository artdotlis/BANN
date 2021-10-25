# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import abc
import re
from dataclasses import dataclass
from typing import Any, TypeVar, Pattern, Generic, Dict, Type, Tuple, Callable, Final, final

from bann.b_container.constants.gen_strings import GenStateNLib


@dataclass(init=False)
class NetStateKwargs:
    def __init__(self, **_: Any) -> None:  # type: ignore
        super().__init__()


_NetStateType = TypeVar('_NetStateType', bound=NetStateKwargs)
_NetStateParse = TypeVar('_NetStateParse', bound=NetStateKwargs)
_NetPrePattern: Final[Pattern[str]] = re.compile(f"^{GenStateNLib.NET.value}(.+)$")


class NetState(Generic[_NetStateType], abc.ABC):
    @abc.abstractmethod
    def get_kwargs_repr(self) -> str:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def get_kwargs(self) -> _NetStateType:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def set_kwargs(self, args_dict: Dict, /) -> None:
        raise NotImplementedError('Interface!')

    @staticmethod
    @final
    def get_pre_arg() -> str:
        return GenStateNLib.NET.value

    @staticmethod
    @final
    def parse_dict(data_con: Type[_NetStateParse], args_dict: Dict, /) -> _NetStateParse:
        new_dict = {}
        for key, value in args_dict.items():
            found = _NetPrePattern.search(key)
            if found is not None:
                new_dict[found.group(1)] = value

        return data_con(**new_dict)


@final
@dataclass
class NetLibElemCon:
    state_types: Dict[str, Tuple[Callable[[str], object], str]]
    net_state: Type[NetState]
