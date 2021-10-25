# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Dict, Callable, Tuple, Optional, Union, Type, Final, final

from bann.b_container.functions.check_arg_complex import check_arg_tuple_or_scalar_single
from bann.b_container.functions.dict_str_repr import dict_json_repr
from bann.b_container.errors.custom_erors import KnownLRSchedulerStateError
from bann.b_container.functions.compare_min_max import AlwaysCompareNumElem, CompareNumElem, \
    compare_min_max_float, compare_min_max_int
from bann.b_container.functions.extend_lists import comp_extend_list, CToAddContainer, \
    set_extend_list
from bann.b_container.states.framework.interface.lr_scheduler import LRSchedulerKwargs, \
    LRSchedulerState

_MODE: Final[Tuple[str, str, str]] = ("triangular", "triangular2", "exp_range")
_EX_MODE: Final[Tuple[str, ...]] = ("cycle", "iterations")
_EX_FN: Final[Tuple[str, ...]] = tuple()
_GAMMA_M: Final[Tuple[str, ...]] = ("exp_range",)


def _scale_1(_: float, /) -> float:
    return 1.


def _get_scale_fn(name: Optional[str], /) -> Optional[Callable[[float], float]]:
    if name is None:
        return None
    if name == "default":
        return _scale_1
    return _scale_1


@final
@dataclass
class _CyclicLRStateKwargs(LRSchedulerKwargs):
    cycle_momentum: bool = True
    mode: str = _MODE[0]
    base_lr: Union[float, Tuple[float, ...]] = 0.001
    max_lr: Union[float, Tuple[float, ...]] = 0.5
    base_momentum: Union[float, Tuple[float, ...]] = 0.8
    max_momentum: Union[float, Tuple[float, ...]] = 0.9
    step_size_up: int = 2000
    step_size_down: int = 2000
    # only in exp_range available
    gamma: float = 1.0
    # 0 <= scale_fn(x) <= 1 for all x >= 0
    scale_fn: Optional[str] = None
    scale_mode: str = _EX_MODE[0]
    last_epoch: int = -1
    verbose: bool = False

    @property
    def get_scheduler_dict(self) -> Dict:
        return {
            'cycle_momentum': self.cycle_momentum,
            'mode': self.mode,
            'scale_fn': _get_scale_fn(self.scale_fn),
            'scale_mode': self.scale_mode,
            'base_lr': self.base_lr,
            'max_lr': self.max_lr,
            'step_size_up': self.step_size_up,
            'step_size_down': self.step_size_down,
            'gamma': self.gamma,
            'base_momentum': self.base_momentum,
            'max_momentum': self.max_momentum,
            'last_epoch': self.last_epoch,
            # TODO add in 1.8.1
            # 'verbose': self.verbose
        }

    # Max Min settings
    max_h_lr: float = 10.
    max_step_size_up: int = 250000
    max_step_size_down: int = 250000


@final
class CyclicLRState(LRSchedulerState[_CyclicLRStateKwargs]):
    def __init__(self) -> None:
        super().__init__()
        self.__kwargs: Optional[_CyclicLRStateKwargs] = None

    def get_kwargs_repr(self, index: int, /) -> str:
        return dict_json_repr(self.get_kwargs().__dict__, f"{self.get_pre_arg()}{index}_")

    @property
    def type_values(self) -> Tuple[Union[Type[int], Type[float]], ...]:
        base_list = [int, int]
        if self.get_kwargs().mode in _GAMMA_M or self.get_kwargs().scale_fn in _GAMMA_M:
            base_list.append(float)
        comp_extend_list(self.get_kwargs().base_lr, base_list, float)
        comp_extend_list(self.get_kwargs().max_lr, base_list, float)
        if self.get_kwargs().cycle_momentum:
            comp_extend_list(self.get_kwargs().base_momentum, base_list, float)
            comp_extend_list(self.get_kwargs().max_momentum, base_list, float)
        return tuple(base_list)

    @property
    def hyper_min_values(self) -> Tuple[AlwaysCompareNumElem, ...]:
        erg = [
            AlwaysCompareNumElem(True, 1),
            AlwaysCompareNumElem(True, 1)
        ]
        if self.get_kwargs().mode in _GAMMA_M or self.get_kwargs().scale_fn in _GAMMA_M:
            erg.append(AlwaysCompareNumElem(False, 0))
        comp_extend_list(self.get_kwargs().base_lr, erg, AlwaysCompareNumElem(False, 0))
        comp_extend_list(
            self.get_kwargs().max_lr, erg,
            CToAddContainer(True, False, self.get_kwargs().base_lr)
        )
        if self.get_kwargs().cycle_momentum:
            comp_extend_list(self.get_kwargs().base_momentum, erg, AlwaysCompareNumElem(True, 0))
            comp_extend_list(
                self.get_kwargs().max_momentum, erg,
                CToAddContainer(True, False, self.get_kwargs().base_momentum)
            )
        return tuple(erg)

    @property
    def hyper_max_values(self) -> Tuple[AlwaysCompareNumElem, ...]:
        erg = [
            AlwaysCompareNumElem(True, self.get_kwargs().max_step_size_up),
            AlwaysCompareNumElem(True, self.get_kwargs().max_step_size_down)
        ]
        if self.get_kwargs().mode in _GAMMA_M or self.get_kwargs().scale_fn in _GAMMA_M:
            erg.append(AlwaysCompareNumElem(True, 1))
        comp_extend_list(
            self.get_kwargs().base_lr, erg,
            CToAddContainer(True, False, self.get_kwargs().max_lr)
        )
        comp_extend_list(
            self.get_kwargs().max_lr, erg,
            AlwaysCompareNumElem(True, self.get_kwargs().max_h_lr)
        )
        if self.get_kwargs().cycle_momentum:
            comp_extend_list(
                self.get_kwargs().base_momentum, erg,
                CToAddContainer(True, False, self.get_kwargs().max_momentum)
            )
            comp_extend_list(
                self.get_kwargs().max_momentum, erg, AlwaysCompareNumElem(False, 1)
            )
        return tuple(erg)

    @property
    def min_values(self) -> Tuple[CompareNumElem, ...]:
        erg = [
            CompareNumElem(True, 1),
            CompareNumElem(True, 1)
        ]
        if self.get_kwargs().mode in _GAMMA_M or self.get_kwargs().scale_fn in _GAMMA_M:
            erg.append(CompareNumElem(False, 0))
        comp_extend_list(
            self.get_kwargs().base_lr, erg, CompareNumElem(False, 0)
        )
        comp_extend_list(
            self.get_kwargs().max_lr, erg,
            CToAddContainer(False, False, self.get_kwargs().base_lr)
        )
        if self.get_kwargs().cycle_momentum:
            comp_extend_list(
                self.get_kwargs().base_momentum, erg, CompareNumElem(True, 0)
            )
            comp_extend_list(
                self.get_kwargs().max_momentum, erg,
                CToAddContainer(False, False, self.get_kwargs().base_momentum)
            )
        return tuple(erg)

    @property
    def max_values(self) -> Tuple[CompareNumElem, ...]:
        erg = [
            CompareNumElem(True, None),
            CompareNumElem(True, None)
        ]
        if self.get_kwargs().mode in _GAMMA_M or self.get_kwargs().scale_fn in _GAMMA_M:
            erg.append(CompareNumElem(True, 1))
        comp_extend_list(
            self.get_kwargs().base_lr, erg,
            CToAddContainer(False, False, self.get_kwargs().max_lr)
        )
        comp_extend_list(
            self.get_kwargs().max_lr, erg,
            CompareNumElem(True, None)
        )
        if self.get_kwargs().cycle_momentum:
            comp_extend_list(
                self.get_kwargs().base_momentum, erg,
                CToAddContainer(False, False, self.get_kwargs().max_momentum)
            )
            comp_extend_list(
                self.get_kwargs().max_momentum, erg,
                CompareNumElem(False, 1)
            )
        return tuple(erg)

    def get_kwargs(self) -> _CyclicLRStateKwargs:
        if self.__kwargs is None:
            raise KnownLRSchedulerStateError("Kwargs not set!")
        return self.__kwargs

    def set_kwargs(self, args_dict: Dict, /) -> None:
        if self.__kwargs is not None:
            raise KnownLRSchedulerStateError("Kwargs already set!")
        self.__kwargs = super().parse_dict(_CyclicLRStateKwargs, args_dict)
        self.set_new_hyper_param(self.get_hyper_param())

    def get_hyper_param(self) -> Tuple[float, ...]:
        erg = [
            float(self.get_kwargs().step_size_up),
            float(self.get_kwargs().step_size_down)
        ]
        if self.get_kwargs().mode in _GAMMA_M or self.get_kwargs().scale_fn in _GAMMA_M:
            erg.append(self.get_kwargs().gamma)
        set_extend_list([
            self.get_kwargs().base_lr,
            self.get_kwargs().max_lr
        ], erg)
        if self.get_kwargs().cycle_momentum:
            set_extend_list([
                self.get_kwargs().base_momentum,
                self.get_kwargs().max_momentum
            ], erg)
        return tuple(erg)

    def set_new_hyper_param(self, params: Tuple[float, ...], /) -> None:
        param_len = 4
        index_add = [0, 0, 0, 0]
        if isinstance(self.get_kwargs().base_lr, tuple):
            param_len += (2 * len(self.get_kwargs().base_lr) - 2)
            index_add[1] += (len(self.get_kwargs().base_lr) - 1)
            for ind_i in range(2, 4):
                index_add[ind_i] += (2 * len(self.get_kwargs().base_lr) - 2)
        if self.get_kwargs().mode in _GAMMA_M or self.get_kwargs().scale_fn in _GAMMA_M:
            param_len += 1
            for ind_i in range(4):
                index_add[ind_i] += 1
        if self.get_kwargs().cycle_momentum:
            param_len += 2
            if isinstance(self.get_kwargs().base_momentum, tuple):
                param_len += (2 * len(self.get_kwargs().base_momentum) - 2)
                index_add[3] += (len(self.get_kwargs().base_momentum) - 1)
        if len(params) != param_len:
            raise KnownLRSchedulerStateError(
                f"The argument tuple has {len(params)} elements but needed {param_len}"
            )
        self.get_kwargs().step_size_up = compare_min_max_int(
            params[0], self.min_values[0], self.max_values[0]
        )
        self.get_kwargs().step_size_down = compare_min_max_int(
            params[1], self.min_values[1], self.max_values[1]
        )
        if self.get_kwargs().mode in _GAMMA_M or self.get_kwargs().scale_fn in _GAMMA_M:
            self.get_kwargs().gamma = compare_min_max_float(
                params[2], self.min_values[2], self.max_values[2]
            )
        if isinstance(self.get_kwargs().base_lr, tuple):
            self.get_kwargs().base_lr = tuple(compare_min_max_float(
                params[i_add + 2 + index_add[0]],
                self.min_values[i_add + 2 + index_add[0]],
                self.max_values[i_add + 2 + index_add[0]]
            ) for i_add in range(len(self.get_kwargs().base_lr)))
        else:
            self.get_kwargs().base_lr = compare_min_max_float(
                params[2 + index_add[0]],
                self.min_values[2 + index_add[0]], self.max_values[2 + index_add[0]]
            )
        if isinstance(self.get_kwargs().max_lr, tuple):
            self.get_kwargs().max_lr = tuple(compare_min_max_float(
                params[i_add + 3 + index_add[1]],
                self.min_values[i_add + 3 + index_add[1]],
                self.max_values[i_add + 3 + index_add[1]]
            ) for i_add in range(len(self.get_kwargs().max_lr)))
        else:
            self.get_kwargs().max_lr = compare_min_max_float(
                params[3 + index_add[1]], self.min_values[3 + index_add[1]],
                self.max_values[3 + index_add[1]]
            )
        if self.get_kwargs().cycle_momentum:
            if isinstance(self.get_kwargs().base_momentum, tuple):
                self.get_kwargs().base_momentum = tuple(compare_min_max_float(
                    params[i_add + 4 + index_add[2]],
                    self.min_values[i_add + 4 + index_add[2]],
                    self.max_values[i_add + 4 + index_add[2]]
                ) for i_add in range(len(self.get_kwargs().base_momentum)))
            else:
                self.get_kwargs().base_momentum = compare_min_max_float(
                    params[4 + index_add[2]], self.min_values[4 + index_add[2]],
                    self.max_values[4 + index_add[2]]
                )
            if isinstance(self.get_kwargs().max_momentum, tuple):
                self.get_kwargs().max_momentum = tuple(compare_min_max_float(
                    params[i_add + 5 + index_add[3]],
                    self.min_values[i_add + 5 + index_add[3]],
                    self.max_values[i_add + 5 + index_add[3]]
                ) for i_add in range(len(self.get_kwargs().max_momentum)))
            else:
                self.get_kwargs().max_momentum = compare_min_max_float(
                    params[5 + index_add[3]], self.min_values[5 + index_add[3]],
                    self.max_values[5 + index_add[3]]
                )


_CyclicLRStateTypes: Final[Dict[str, Tuple[Callable[[str], object], str]]] = {
    f'{LRSchedulerState.get_pre_arg()}base_lr': (
        lambda val: check_arg_tuple_or_scalar_single(
            val, lambda val_i: float(val_i) if float(val_i) > 0 else 0.001
        ),
        "float (>0) or Tuple[float (>0), ...] as str,..."
    ),
    f'{LRSchedulerState.get_pre_arg()}max_lr': (
        lambda val: check_arg_tuple_or_scalar_single(
            val, lambda val_i: float(val_i) if float(val_i) > 0 else 0.5
        ),
        "float (>0) or Tuple[float (>0), ...] as str,..."
    ),
    f'{LRSchedulerState.get_pre_arg()}base_momentum': (
        lambda val: check_arg_tuple_or_scalar_single(
            val, lambda val_i: float(val_i) if float(val_i) >= 0 else 0.8
        ),
        "float (>=0) or Tuple[float (>=0), ...] as str,..."
    ),
    f'{LRSchedulerState.get_pre_arg()}max_momentum': (
        lambda val: check_arg_tuple_or_scalar_single(
            val, lambda val_i: float(val_i) if float(val_i) >= 0 else 0.9
        ),
        "float (>=0) or Tuple[float (>=0), ...] as str,..."
    ),
    f'{LRSchedulerState.get_pre_arg()}last_epoch': (
        lambda val: int(val) if int(val) > 0 else -1,
        "int (>0) else -1"
    ),
    f'{LRSchedulerState.get_pre_arg()}step_size_up': (
        lambda val: int(val) if int(val) > 0 else 1,
        "int (>0)"
    ),
    f'{LRSchedulerState.get_pre_arg()}step_size_down': (
        lambda val: int(val) if int(val) > 0 else 1,
        "int (>0)"
    ),
    f'{LRSchedulerState.get_pre_arg()}mode': (
        lambda val: str(val) if val in _MODE else _MODE[0],
        f"str ({','.join(_MODE)})"
    ),
    f'{LRSchedulerState.get_pre_arg()}scale_fn': (
        lambda val: str(val) if val in _EX_FN else None,
        f"str ({','.join(_EX_FN)})"
    ),
    f'{LRSchedulerState.get_pre_arg()}scale_mode': (
        lambda val: str(val) if val in _EX_MODE else _EX_MODE[0],
        f"str ({','.join(_EX_MODE)})"
    ),
    f'{LRSchedulerState.get_pre_arg()}gamma': (
        lambda val: float(val) if float(val) > 0 else 1,
        "float (>0)"
    ),
    f'{LRSchedulerState.get_pre_arg()}cycle_momentum': (
        lambda val: val == 'T',
        "True if T else False"
    ),
    f'{LRSchedulerState.get_pre_arg()}max_h_lr': (
        lambda val: float(val) if float(val) > 0 else 0.5,
        "float (>0)"
    ),
    f'{LRSchedulerState.get_pre_arg()}max_step_size_up': (
        lambda val: int(val) if int(val) >= 2 else 250000,
        "int (>=2)"
    ),
    f'{LRSchedulerState.get_pre_arg()}verbose': (
        lambda val: val == 'T',
        "True if T else False"
    ),
    f'{LRSchedulerState.get_pre_arg()}max_step_size_down': (
        lambda val: int(val) if int(val) >= 2 else 250000,
        "int (>=2)"
    )
}


def get_cyclic_lr_state_types() -> Dict[str, Tuple[Callable[[str], object], str]]:
    return _CyclicLRStateTypes
