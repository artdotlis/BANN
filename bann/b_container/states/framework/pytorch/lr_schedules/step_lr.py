# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Callable, Union, Type, final, Final

from bann.b_container.functions.dict_str_repr import dict_json_repr
from bann.b_container.errors.custom_erors import KnownLRSchedulerStateError
from bann.b_container.states.framework.interface.lr_scheduler import LRSchedulerKwargs, \
    LRSchedulerState
from bann.b_container.functions.compare_min_max import CompareNumElem, compare_min_max_int, \
    compare_min_max_float
from bann.b_container.functions.compare_min_max import AlwaysCompareNumElem


@final
@dataclass
class _StepLRStateKwargs(LRSchedulerKwargs):
    step_size: int = 30
    gamma: float = 0.1
    verbose: bool = False
    last_epoch: int = -1

    @property
    def get_scheduler_dict(self) -> Dict:
        return {
            'step_size': self.step_size,
            'gamma': self.gamma,
            'last_epoch': self.last_epoch,
            # TODO add in 1.8.1
            # 'verbose': self.verbose
        }

    # Max Min settings
    max_step_size: int = 10000


@final
class StepLRState(LRSchedulerState[_StepLRStateKwargs]):

    def __init__(self) -> None:
        super().__init__()
        self.__kwargs: Optional[_StepLRStateKwargs] = None

    def get_kwargs_repr(self, index: int, /) -> str:
        return dict_json_repr(self.get_kwargs().__dict__, f"{self.get_pre_arg()}{index}_")

    @property
    def type_values(self) -> Tuple[Union[Type[int], Type[float]], ...]:
        erg = (int, float)
        return erg

    @property
    def hyper_min_values(self) -> Tuple[AlwaysCompareNumElem, ...]:
        erg = (
            AlwaysCompareNumElem(True, 1),
            AlwaysCompareNumElem(False, 0)
        )
        return erg

    @property
    def hyper_max_values(self) -> Tuple[AlwaysCompareNumElem, ...]:
        erg = (
            AlwaysCompareNumElem(True, self.get_kwargs().max_step_size),
            AlwaysCompareNumElem(True, 1)
        )
        return erg

    @property
    def min_values(self) -> Tuple[CompareNumElem, ...]:
        erg = (
            CompareNumElem(True, 1),
            CompareNumElem(False, 0)
        )
        return erg

    @property
    def max_values(self) -> Tuple[CompareNumElem, ...]:
        return tuple(CompareNumElem(True, None) for _ in range(2))

    def get_kwargs(self) -> _StepLRStateKwargs:
        if self.__kwargs is None:
            raise KnownLRSchedulerStateError("Kwargs not set!")
        return self.__kwargs

    def set_kwargs(self, args_dict: Dict, /) -> None:
        if self.__kwargs is not None:
            raise KnownLRSchedulerStateError("Kwargs already set!")
        self.__kwargs = super().parse_dict(_StepLRStateKwargs, args_dict)
        self.set_new_hyper_param(self.get_hyper_param())

    def get_hyper_param(self) -> Tuple[float, ...]:
        return (
            float(self.get_kwargs().step_size),
            self.get_kwargs().gamma
        )

    def set_new_hyper_param(self, params: Tuple[float, ...], /) -> None:
        if len(params) != 2:
            raise KnownLRSchedulerStateError(
                f"The argument tuple has {len(params)} elements but needed 2!"
            )
        self.get_kwargs().step_size = compare_min_max_int(
            params[0], self.min_values[0], self.max_values[0]
        )
        self.get_kwargs().gamma = compare_min_max_float(
            params[1], self.min_values[1], self.max_values[1]
        )


_LRStepStateTypes: Final[Dict[str, Tuple[Callable[[str], object], str]]] = {
    f'{LRSchedulerState.get_pre_arg()}step_size': (
        lambda val: int(val) if int(val) >= 1 else 30,
        "int (>=1)"
    ),
    f'{LRSchedulerState.get_pre_arg()}gamma': (
        lambda val: float(val) if float(val) > 0 else 0.1,
        "float (>0)"
    ),
    f'{LRSchedulerState.get_pre_arg()}verbose': (
        lambda val: val == 'T',
        "True if T else False"
    ),
    f'{LRSchedulerState.get_pre_arg()}last_epoch': (
        lambda val: int(val) if int(val) > 0 else -1,
        "int (>0) else -1"
    ),
    f'{LRSchedulerState.get_pre_arg()}max_step_size': (
        lambda val: int(val) if int(val) >= 2 else 10000,
        "int (>=2)"
    )
}


def get_step_lr_state_types() -> Dict[str, Tuple[Callable[[str], object], str]]:
    return _LRStepStateTypes
