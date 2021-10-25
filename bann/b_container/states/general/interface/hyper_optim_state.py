# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import re
from pathlib import Path
from typing import TypeVar, Pattern, Generic, Dict, Type, Optional, Tuple, Callable, Final, final

import abc
from dataclasses import dataclass

from bann.b_container.constants.gen_strings import GenStateNLib
from bann.b_container.errors.custom_erors import KnownHyperOptimError


@dataclass
class OptimHyperStateKwargs:
    stop_fp: Optional[Path] = None
    stop_it: int = 0
    stop_min: int = 0


_OptimHyperStateType = TypeVar('_OptimHyperStateType', bound=OptimHyperStateKwargs)
_OptimHyperPrePattern: Final[Pattern[str]] = re.compile(f"^{GenStateNLib.HYPER.value}(.+)$")


class OptimHyperState(Generic[_OptimHyperStateType], abc.ABC):
    @abc.abstractmethod
    def get_kwargs_repr(self) -> str:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def get_kwargs(self) -> _OptimHyperStateType:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def set_kwargs(self, args_dict: Dict, /) -> None:
        raise NotImplementedError('Interface!')

    @staticmethod
    @final
    def get_pre_arg() -> str:
        return GenStateNLib.HYPER.value

    @staticmethod
    @final
    def parse_dict(data_con: Type[_OptimHyperStateType],
                   args_dict: Dict, /) -> _OptimHyperStateType:
        new_dict = {}
        for key, value in args_dict.items():
            found = _OptimHyperPrePattern.search(key)
            if found is not None:
                new_dict[found.group(1)] = value

        return data_con(**new_dict)


def _check_path(path: str, /) -> Path:
    return Path(path)


_HyperStateTypes: Final[Dict[str, Tuple[Callable[[str], object], str]]] = {
    f'{OptimHyperState.get_pre_arg()}stop_fp': (_check_path, "Path"),
    f'{OptimHyperState.get_pre_arg()}stop_it': (
        lambda iterations: int(iterations) if int(iterations) >= 0 else 0,
        "int (>=0)"
    ),
    f'{OptimHyperState.get_pre_arg()}stop_min': (
        lambda time_min: int(time_min) if int(time_min) >= 0 else 0,
        "int (>=0)"
    )
}


def merged_hyper_gen_lib(to_merge: Dict[str, Tuple[Callable[[str], object], str]], /) \
        -> Dict[str, Tuple[Callable[[str], object], str]]:
    merged_dict = {**_HyperStateTypes}
    for key, value in to_merge.items():
        if key in merged_dict:
            raise KnownHyperOptimError(f"Hyper param duplicate: {key}")
        merged_dict[key] = value
    return merged_dict
