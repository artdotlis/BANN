# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import final, Optional, Dict, Final, Tuple, Callable

from bann.b_container.errors.custom_erors import KnownHyperOptimError
from bann.b_container.functions.dict_str_repr import dict_json_repr
from bann.b_container.states.general.interface.hyper_optim_state import OptimHyperStateKwargs, \
    OptimHyperState, merged_hyper_gen_lib


@final
@dataclass
class _EGAHyperState(OptimHyperStateKwargs):
    include_first_run: bool = True
    repeats: int = 10
    end_fitness: Optional[float] = None
    population: int = 50
    co_rate: float = 0.65
    m_rate: float = 0.025
    memory: int = 100
    memory_repeats: int = 1


@final
class EGAHyperState(OptimHyperState[_EGAHyperState]):
    def __init__(self) -> None:
        super().__init__()
        self.__kwargs: Optional[_EGAHyperState] = None

    def get_kwargs_repr(self) -> str:
        return dict_json_repr(self.get_kwargs().__dict__, self.get_pre_arg())

    def get_kwargs(self) -> _EGAHyperState:
        if self.__kwargs is None:
            raise KnownHyperOptimError("Kwargs not set!")
        return self.__kwargs

    def set_kwargs(self, args_dict: Dict, /) -> None:
        if self.__kwargs is not None:
            raise KnownHyperOptimError("Kwargs already set!")
        self.__kwargs = super().parse_dict(_EGAHyperState, args_dict)


_OptimHyperStateTypes: Final[Dict[str, Tuple[Callable[[str], object], str]]] = {
    f'{OptimHyperState.get_pre_arg()}memory': (
        lambda repeats: int(repeats) if int(repeats) >= 0 else 0,
        "int (>=0)"
    ),
    f'{OptimHyperState.get_pre_arg()}memory_repeats': (
        lambda repeats: int(repeats) if int(repeats) >= 1 else 1,
        "int (>=1)"
    ),
    f'{OptimHyperState.get_pre_arg()}repeats': (
        lambda repeats: int(repeats) if int(repeats) >= 1 else 10,
        "int (>=1)"
    ),
    f'{OptimHyperState.get_pre_arg()}end_fitness': (
        lambda fit: float(fit) if float(fit) > 0 else None,
        "float (>0)"
    ),
    f'{OptimHyperState.get_pre_arg()}include_first_run': (
        lambda inc: inc == 'T', "T for True else False"
    ),
    f'{OptimHyperState.get_pre_arg()}population': (
        lambda fit: int(fit) if int(fit) >= 10 else 10,
        "int (>=10)"
    ),
    f'{OptimHyperState.get_pre_arg()}co_rate': (
        lambda alpha: float(alpha) if 0 < float(alpha) <= 1 else 0.65,
        "float (0<x<=1)"
    ),
    f'{OptimHyperState.get_pre_arg()}m_rate': (
        lambda alpha: float(alpha) if 0 <= float(alpha) <= 1 else 0.05,
        "float (0<=x<=1)"
    )
}


def get_ega_hyper_state_types() -> Dict[str, Tuple[Callable[[str], object], str]]:
    return merged_hyper_gen_lib(_OptimHyperStateTypes)
