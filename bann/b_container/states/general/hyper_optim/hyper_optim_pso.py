# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Callable, Final, final

from bann.b_container.functions.dict_str_repr import dict_json_repr
from bann.b_container.errors.custom_erors import KnownHyperOptimError
from bann.b_container.states.general.interface.hyper_optim_state import OptimHyperStateKwargs, \
    OptimHyperState, merged_hyper_gen_lib


@dataclass
class _PSOHyperState(OptimHyperStateKwargs):
    repeats: int = 10
    end_fitness: Optional[float] = None
    swarm: int = 12
    c_1: float = 2
    c_2: float = 2
    weight: float = 0.8
    memory: int = 100
    memory_repeats: int = 1


@final
class PSOHyperState(OptimHyperState[_PSOHyperState]):
    def __init__(self) -> None:
        super().__init__()
        self.__kwargs: Optional[_PSOHyperState] = None

    def get_kwargs_repr(self) -> str:
        return dict_json_repr(self.get_kwargs().__dict__, self.get_pre_arg())

    def get_kwargs(self) -> _PSOHyperState:
        if self.__kwargs is None:
            raise KnownHyperOptimError("Kwargs not set!")
        return self.__kwargs

    def set_kwargs(self, args_dict: Dict, /) -> None:
        if self.__kwargs is not None:
            raise KnownHyperOptimError("Kwargs already set!")
        self.__kwargs = super().parse_dict(_PSOHyperState, args_dict)


_PSOHyperStateTypes: Final[Dict[str, Tuple[Callable[[str], object], str]]] = {
    f'{OptimHyperState.get_pre_arg()}memory': (
        lambda repeats: int(repeats) if int(repeats) >= 0 else 0,
        "int (>=0)"
    ),
    f'{OptimHyperState.get_pre_arg()}memory_repeats': (
        lambda repeats: int(repeats) if int(repeats) >= 1 else 1,
        "int (>=1)"
    ),
    f'{OptimHyperState.get_pre_arg()}repeats': (
        lambda repeats: int(repeats) if int(repeats) >= 3 else 3,
        "int (>=3)"
    ),
    f'{OptimHyperState.get_pre_arg()}end_fitness': (
        lambda fit: float(fit) if float(fit) > 0 else None,
        "float (>0)"
    ),
    f'{OptimHyperState.get_pre_arg()}c_1': (
        lambda fit: float(fit) if float(fit) > 0 else 2,
        "float (>0)"
    ),
    f'{OptimHyperState.get_pre_arg()}c_2': (
        lambda fit: float(fit) if float(fit) > 0 else 2,
        "float (>0)"
    ),
    f'{OptimHyperState.get_pre_arg()}weight': (
        lambda fit: float(fit) if float(fit) > 0 else 0.9,
        "float (>0)"
    ),
    f'{OptimHyperState.get_pre_arg()}swarm': (
        lambda fit: int(fit) if int(fit) >= 12 else 12,
        "int (>=12)"
    )
}


def get_pso_hyper_state_types() -> Dict[str, Tuple[Callable[[str], object], str]]:
    return merged_hyper_gen_lib(_PSOHyperStateTypes)


@final
@dataclass
class _DAPSOHyperState(_PSOHyperState):
    l_repeats: int = 10
    l_increase: float = 0.
    include_first_run: bool = True
    alpha: float = 0.9
    speed_prob: float = 0.2
    survival_rate: float = 0.9
    dim_prob: float = 0.0


@final
class DAPSOHyperState(OptimHyperState[_DAPSOHyperState]):
    def __init__(self) -> None:
        super().__init__()
        self.__kwargs: Optional[_DAPSOHyperState] = None

    def get_kwargs_repr(self) -> str:
        return dict_json_repr(self.get_kwargs().__dict__, self.get_pre_arg())

    def get_kwargs(self) -> _DAPSOHyperState:
        if self.__kwargs is None:
            raise KnownHyperOptimError("Kwargs not set!")
        return self.__kwargs

    def set_kwargs(self, args_dict: Dict, /) -> None:
        if self.__kwargs is not None:
            raise KnownHyperOptimError("Kwargs already set!")
        self.__kwargs = super().parse_dict(_DAPSOHyperState, args_dict)


_DAPSOHyperStateTypes: Final[Dict[str, Tuple[Callable[[str], object], str]]] = {
    **_PSOHyperStateTypes,
    f'{OptimHyperState.get_pre_arg()}l_repeats': (
        lambda repeats: int(repeats) if int(repeats) >= 4 else 4,
        "int (>=4)"
    ),
    f'{OptimHyperState.get_pre_arg()}l_increase': (
        lambda inc: float(inc) if float(inc) >= 0 else 0,
        "float (>0)"
    ),
    f'{OptimHyperState.get_pre_arg()}include_first_run': (
        lambda inc: inc == 'T', "T for True else False"
    ),
    f'{OptimHyperState.get_pre_arg()}alpha': (
        lambda alpha: float(alpha) if 0 < float(alpha) <= 1 else 0.9,
        "float (0<x<=1)"
    ),
    f'{OptimHyperState.get_pre_arg()}speed_prob': (
        lambda alpha: float(alpha) if 0 < float(alpha) <= 1 else 0.2,
        "float (0<x<=1)"
    ),
    f'{OptimHyperState.get_pre_arg()}survival_rate': (
        lambda alpha: float(alpha) if 0 < float(alpha) <= 1 else 0.9,
        "float (0<x<=1)"
    ),
    f'{OptimHyperState.get_pre_arg()}dim_prob': (
        lambda alpha: float(alpha) if 0 <= float(alpha) <= 1 else 0.0,
        "float (0<=x<=1)"
    )
}


def get_da_pso_hyper_state_types() -> Dict[str, Tuple[Callable[[str], object], str]]:
    return merged_hyper_gen_lib(_DAPSOHyperStateTypes)
