# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Dict, Callable, Optional, Tuple, Union, Type, Final, final

from bann.b_container.functions.check_arg_complex import check_arg_tuple_or_scalar_single
from bann.b_container.functions.dict_str_repr import dict_json_repr
from bann.b_container.functions.extend_lists import comp_extend_list, set_extend_list
from bann.b_container.states.framework.interface.lr_scheduler import LRSchedulerKwargs, \
    LRSchedulerState
from bann.b_container.errors.custom_erors import KnownLRSchedulerStateError
from bann.b_container.functions.compare_min_max import CompareNumElem, compare_min_max_float, \
    compare_min_max_int
from bann.b_container.functions.compare_min_max import AlwaysCompareNumElem

_MODE: Final[Tuple[str, str]] = ("min", "max")
_TH_MODE: Final[Tuple[str, str]] = ('rel', 'abs')


@final
@dataclass
class _ReduceLROnPlateauStateKwargs(LRSchedulerKwargs):
    mode: str = _MODE[0]
    threshold_mode: str = _TH_MODE[0]
    factor: float = 0.1
    patience: int = 10
    verbose: bool = False
    threshold: float = 0.0001
    cooldown: int = 0
    min_lr: Union[float, Tuple[float, ...]] = 0
    eps: float = 1e-08

    @property
    def get_scheduler_dict(self) -> Dict:
        return {
            'mode': self.mode,
            'threshold_mode': self.threshold_mode,
            'factor': self.factor,
            'patience': self.patience,
            'verbose': self.verbose,
            'threshold': self.threshold,
            'cooldown': self.cooldown,
            'min_lr': self.min_lr,
            'eps': self.eps
        }

    # Max Min settings
    max_patience: int = 100
    max_cooldown: int = 100
    max_threshold: float = 1e-3
    max_min_lr: float = 1e-4


@final
class ReduceLROnPlateauState(LRSchedulerState[_ReduceLROnPlateauStateKwargs]):

    def __init__(self) -> None:
        super().__init__()
        self.__kwargs: Optional[_ReduceLROnPlateauStateKwargs] = None

    def get_kwargs_repr(self, index: int, /) -> str:
        return dict_json_repr(self.get_kwargs().__dict__, f"{self.get_pre_arg()}{index}_")

    @property
    def type_values(self) -> Tuple[Union[Type[int], Type[float]], ...]:
        base_list = [float, int, float, int]
        comp_extend_list(self.get_kwargs().min_lr, base_list, float)
        return tuple(base_list)

    @property
    def hyper_min_values(self) -> Tuple[AlwaysCompareNumElem, ...]:
        erg = [
            AlwaysCompareNumElem(False, 0),
            AlwaysCompareNumElem(False, 0),
            AlwaysCompareNumElem(False, 0),
            AlwaysCompareNumElem(True, 0)
        ]
        comp_extend_list(self.get_kwargs().min_lr, erg, AlwaysCompareNumElem(True, 0))
        return tuple(erg)

    @property
    def hyper_max_values(self) -> Tuple[AlwaysCompareNumElem, ...]:
        erg = [
            AlwaysCompareNumElem(False, 1),
            AlwaysCompareNumElem(True, self.get_kwargs().max_patience),
            AlwaysCompareNumElem(False, self.get_kwargs().max_threshold),
            AlwaysCompareNumElem(True, self.get_kwargs().max_cooldown)
        ]
        comp_extend_list(
            self.get_kwargs().min_lr, erg,
            AlwaysCompareNumElem(True, self.get_kwargs().max_min_lr)
        )
        return tuple(erg)

    @property
    def min_values(self) -> Tuple[CompareNumElem, ...]:
        erg = [
            CompareNumElem(False, 0),
            CompareNumElem(False, 0),
            CompareNumElem(False, 0),
            CompareNumElem(True, 0)
        ]
        comp_extend_list(
            self.get_kwargs().min_lr, erg,
            CompareNumElem(True, 0)
        )
        return tuple(erg)

    @property
    def max_values(self) -> Tuple[CompareNumElem, ...]:
        erg_list = [CompareNumElem(False, 1)]
        erg_list.extend(CompareNumElem(True, None) for _ in range(3))
        comp_extend_list(
            self.get_kwargs().min_lr, erg_list,
            CompareNumElem(True, None)
        )
        return tuple(erg_list)

    def get_kwargs(self) -> _ReduceLROnPlateauStateKwargs:
        if self.__kwargs is None:
            raise KnownLRSchedulerStateError("Kwargs not set!")
        return self.__kwargs

    def set_kwargs(self, args_dict: Dict, /) -> None:
        if self.__kwargs is not None:
            raise KnownLRSchedulerStateError("Kwargs already set!")
        self.__kwargs = super().parse_dict(_ReduceLROnPlateauStateKwargs, args_dict)
        self.set_new_hyper_param(self.get_hyper_param())

    def get_hyper_param(self) -> Tuple[float, ...]:
        erg = [
            self.get_kwargs().factor,
            float(self.get_kwargs().patience),
            self.get_kwargs().threshold,
            float(self.get_kwargs().cooldown)
        ]
        set_extend_list([self.get_kwargs().min_lr], erg)
        return tuple(erg)

    def set_new_hyper_param(self, params: Tuple[float, ...], /) -> None:
        param_len = 5
        if isinstance(self.get_kwargs().min_lr, tuple):
            param_len += (len(self.get_kwargs().min_lr) - 1)
        if len(params) != param_len:
            raise KnownLRSchedulerStateError(
                f"The argument tuple has {len(params)} elements but needed {param_len}!"
            )
        self.get_kwargs().factor = compare_min_max_float(
            params[0], self.min_values[0], self.max_values[0]
        )
        self.get_kwargs().patience = compare_min_max_int(
            params[1], self.min_values[1], self.max_values[1]
        )
        self.get_kwargs().threshold = compare_min_max_float(
            params[2], self.min_values[2], self.max_values[2]
        )
        self.get_kwargs().cooldown = compare_min_max_int(
            params[3], self.min_values[3], self.max_values[3]
        )
        if isinstance(self.get_kwargs().min_lr, tuple):
            self.get_kwargs().min_lr = tuple(compare_min_max_float(
                params[i_add + 4], self.min_values[i_add + 4], self.max_values[i_add + 4]
            ) for i_add in range(len(self.get_kwargs().min_lr)))
        else:
            self.get_kwargs().min_lr = compare_min_max_float(
                params[4], self.min_values[4], self.max_values[4]
            )


_ReduceLROnPlateauStateTypes: Final[Dict[str, Tuple[Callable[[str], object], str]]] = {
    f'{LRSchedulerState.get_pre_arg()}mode': (
        lambda val: str(val) if val in _MODE else _MODE[0],
        f"str ({','.join(_MODE)})"
    ),
    f'{LRSchedulerState.get_pre_arg()}factor': (
        lambda val: float(val) if float(val) > 0 else 0.1,
        "float (>0)"
    ),
    f'{LRSchedulerState.get_pre_arg()}patience': (
        lambda val: int(val) if int(val) >= 1 else 10,
        "int (>=1)"
    ),
    f'{LRSchedulerState.get_pre_arg()}verbose': (
        lambda val: val == 'T',
        "True if T else False"
    ),
    f'{LRSchedulerState.get_pre_arg()}threshold': (
        lambda val: float(val) if float(val) > 0 or float(val) < 1 else 0.0001,
        "float (>0)"
    ),
    f'{LRSchedulerState.get_pre_arg()}threshold_mode': (
        lambda val: str(val) if val in _TH_MODE else _TH_MODE[0],
        f"str ({','.join(_TH_MODE)})"
    ),
    f'{LRSchedulerState.get_pre_arg()}cooldown': (
        lambda val: int(val) if int(val) >= 0 else 0,
        "int (>=0)"
    ),
    f'{LRSchedulerState.get_pre_arg()}min_lr': (
        lambda val: check_arg_tuple_or_scalar_single(
            val, lambda val_i: float(val_i) if float(val_i) >= 0 else 0.0
        ),
        "float (>=0) or Tuple[float (>=0), ...] as str,..."
    ),
    f'{LRSchedulerState.get_pre_arg()}eps': (
        lambda val: float(val) if float(val) > 0 else 1e-08,
        "float (>0)"
    ),
    f'{LRSchedulerState.get_pre_arg()}max_patience': (
        lambda val: int(val) if int(val) >= 2 else 100,
        "int (>=2)"
    ),
    f'{LRSchedulerState.get_pre_arg()}max_cooldown': (
        lambda val: int(val) if int(val) >= 1 else 100,
        "int (>=1)"
    ),
    f'{LRSchedulerState.get_pre_arg()}max_threshold': (
        lambda val: float(val) if float(val) > 0 else 1e-3,
        "float (>0)"
    ),
    f'{LRSchedulerState.get_pre_arg()}max_min_lr': (
        lambda val: float(val) if float(val) >= 0 else 1e-4,
        "float (>=0)"
    )
}


def get_reduce_lr_on_plateau_state_types() -> Dict[str, Tuple[Callable[[str], object], str]]:
    return _ReduceLROnPlateauStateTypes
