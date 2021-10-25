# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Callable, final, Final

from bann.b_container.errors.custom_erors import KnownHyperOptimError
from bann.b_container.functions.dict_str_repr import dict_json_repr
from bann.b_container.states.general.interface.hyper_optim_state import OptimHyperStateKwargs, \
    OptimHyperState, merged_hyper_gen_lib


@final
@dataclass
class _SimplyTrainStateArgs(OptimHyperStateKwargs):
    package: int = 2
    repeats: int = 10


@final
class SimplyTrainState(OptimHyperState[_SimplyTrainStateArgs]):
    def __init__(self) -> None:
        super().__init__()
        self.__kwargs: Optional[_SimplyTrainStateArgs] = None

    def get_kwargs_repr(self) -> str:
        return dict_json_repr(self.get_kwargs().__dict__, self.get_pre_arg())

    def get_kwargs(self) -> _SimplyTrainStateArgs:
        if self.__kwargs is None:
            raise KnownHyperOptimError("Kwargs not set!")
        return self.__kwargs

    def set_kwargs(self, args_dict: Dict, /) -> None:
        if self.__kwargs is not None:
            raise KnownHyperOptimError("Kwargs already set!")
        self.__kwargs = super().parse_dict(_SimplyTrainStateArgs, args_dict)


_SimplyTrainStateTypes: Final[Dict[str, Tuple[Callable[[str], object], str]]] = {
    f'{OptimHyperState.get_pre_arg()}repeats': (
        lambda repeats: int(repeats) if int(repeats) >= 1 else 10,
        "int (>=1)"
    ),
    f'{OptimHyperState.get_pre_arg()}package': (
        lambda repeats: int(repeats) if int(repeats) >= 1 else 2,
        "int (>=1)"
    )
}


def get_simply_train_state_types() -> Dict[str, Tuple[Callable[[str], object], str]]:
    return merged_hyper_gen_lib(_SimplyTrainStateTypes)
