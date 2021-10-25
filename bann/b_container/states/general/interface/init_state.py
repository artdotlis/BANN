# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import abc
import re
from dataclasses import dataclass
from typing import Any, TypeVar, Generic, Dict, Type, Pattern, Tuple, Callable, Final, final

from bann.b_container.constants.gen_strings import GenStateNLib
from bann.b_hyper_optim.hyper_optim_interface import HyperParamInterface


@dataclass(init=False)
class InitStateKwargs:
    def __init__(self, **_: Any) -> None:  # type: ignore
        super().__init__()


_InitStateType = TypeVar('_InitStateType', bound=InitStateKwargs)
_InitPrePattern: Final[Pattern[str]] = re.compile(f"^{GenStateNLib.INIT.value}(.+)$")


class InitState(Generic[_InitStateType], HyperParamInterface, abc.ABC):

    @abc.abstractmethod
    def get_kwargs_repr(self) -> str:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def get_kwargs(self) -> _InitStateType:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def set_kwargs(self, args_dict: Dict, /) -> None:
        raise NotImplementedError('Interface!')

    @staticmethod
    @final
    def get_pre_arg() -> str:
        return GenStateNLib.INIT.value

    @staticmethod
    @final
    def parse_dict(data_con: Type[_InitStateType], args_dict: Dict, /) -> _InitStateType:
        new_dict = {}
        for key, value in args_dict.items():
            found = _InitPrePattern.search(key)
            if found is not None:
                new_dict[found.group(1)] = value

        return data_con(**new_dict)


class NetInitGlobalInterface(abc.ABC):

    @abc.abstractmethod
    def update_init(self, data_up: InitState, /) -> None:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def update_init_type(self) -> Type[InitState]:
        raise NotImplementedError('Interface!')


@final
@dataclass
class InitLibElemCon:
    state_types: Dict[str, Tuple[Callable[[str], object], str]]
    init_state: Type[InitState]
