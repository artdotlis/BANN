# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import abc
import re
from dataclasses import dataclass
from typing import Any, TypeVar, Pattern, Generic, Dict, Type, Final, final

from bann.b_container.constants.fr_string import StateNLib


@dataclass(init=False)
class CriterionKwargs:
    def __init__(self, **_: Any) -> None:  # type: ignore
        super().__init__()

    @property
    def get_criterion_dict(self) -> Dict:
        return {}


_CriterionType = TypeVar('_CriterionType', bound=CriterionKwargs)
_CriterionPrePattern: Final[Pattern[str]] = re.compile(f"^{StateNLib.CRIT.value}(.+)$")


class CriterionState(Generic[_CriterionType], abc.ABC):

    @abc.abstractmethod
    def get_kwargs_repr(self) -> str:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def get_kwargs(self) -> _CriterionType:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def set_kwargs(self, args_dict: Dict, /) -> None:
        raise NotImplementedError('Interface!')

    @staticmethod
    @final
    def get_pre_arg() -> str:
        return StateNLib.CRIT.value

    @staticmethod
    @final
    def parse_dict(data_con: Type[_CriterionType], args_dict: Dict, /) -> _CriterionType:
        new_dict = {}
        for key, value in args_dict.items():
            found = _CriterionPrePattern.search(key)
            if found is not None:
                new_dict[found.group(1)] = value

        return data_con(**new_dict)
