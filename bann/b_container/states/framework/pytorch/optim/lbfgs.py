# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, Type, Callable, Final, final

from bann.b_container.functions.dict_str_repr import dict_json_repr
from bann.b_container.errors.custom_erors import KnownOptimStateError
from bann.b_container.functions.compare_min_max import AlwaysCompareNumElem, CompareNumElem, \
    compare_min_max_float, compare_min_max_int
from bann.b_container.states.framework.interface.optim_state import OptimStateKwargs, MainOptimSt

_MODE: Final[Tuple[str]] = ("strong_wolfe",)


@final
@dataclass
class _LBFGSStateKwargs(OptimStateKwargs):
    lr: float = 0.0001
    max_iter: int = 20
    max_eval: int = 25
    tolerance_grad: float = 1e-5
    tolerance_change: float = 1e-9
    history_size: int = 100
    line_search_fn: Optional[str] = None

    @property
    def get_optim_dict(self) -> Dict:
        return {
            'lr': self.lr,
            'max_iter': self.max_iter,
            'max_eval': self.max_eval,
            'tolerance_grad': self.tolerance_grad,
            'tolerance_change': self.tolerance_change,
            'history_size': self.history_size,
            'line_search_fn': self.line_search_fn
        }

    # Max Min settings
    max_lr: float = 10.
    max_h_iter: int = 1000
    max_h_eval: int = 1250


@final
class LBFGSState(MainOptimSt[_LBFGSStateKwargs]):

    def __init__(self) -> None:
        super().__init__()
        self.__kwargs: Optional[_LBFGSStateKwargs] = None

    def get_kwargs_repr(self, index: int, /) -> str:
        return dict_json_repr(self.get_kwargs().__dict__, f"{self.get_pre_arg()}{index}_")

    @property
    def type_values(self) -> Tuple[Union[Type[int], Type[float]], ...]:
        return float, int, int, float, float

    @property
    def hyper_min_values(self) -> Tuple[AlwaysCompareNumElem, ...]:
        erg = (
            AlwaysCompareNumElem(False, 0),
            AlwaysCompareNumElem(True, 1),
            AlwaysCompareNumElem(True, 1),
            AlwaysCompareNumElem(False, 0),
            AlwaysCompareNumElem(False, 0)
        )
        return erg

    @property
    def hyper_max_values(self) -> Tuple[AlwaysCompareNumElem, ...]:
        erg = (
            AlwaysCompareNumElem(True, self.get_kwargs().max_lr),
            AlwaysCompareNumElem(True, self.get_kwargs().max_h_iter),
            AlwaysCompareNumElem(True, self.get_kwargs().max_h_eval),
            AlwaysCompareNumElem(False, 1),
            AlwaysCompareNumElem(False, 1)
        )
        return erg

    @property
    def min_values(self) -> Tuple[CompareNumElem, ...]:
        erg = (
            CompareNumElem(False, 0),
            CompareNumElem(True, 1),
            CompareNumElem(True, 1),
            CompareNumElem(False, 0),
            CompareNumElem(False, 0)
        )
        return erg

    @property
    def max_values(self) -> Tuple[CompareNumElem, ...]:
        list_erg = [CompareNumElem(True, None) for _ in range(3)]
        list_erg.extend([
            CompareNumElem(False, 1),
            CompareNumElem(False, 1)
        ])
        return tuple(list_erg)

    def get_kwargs(self) -> _LBFGSStateKwargs:
        if self.__kwargs is None:
            raise KnownOptimStateError("Kwargs not set!")
        return self.__kwargs

    def set_kwargs(self, args_dict: Dict, /) -> None:
        if self.__kwargs is not None:
            raise KnownOptimStateError("Kwargs already set!")

        self.__kwargs = super().parse_dict(_LBFGSStateKwargs, args_dict)
        self.set_new_hyper_param(self.get_hyper_param())

    def get_hyper_param(self) -> Tuple[float, ...]:
        return (
            self.get_kwargs().lr,
            float(self.get_kwargs().max_iter),
            float(self.get_kwargs().max_eval),
            self.get_kwargs().tolerance_grad,
            self.get_kwargs().tolerance_change
        )

    def set_new_hyper_param(self, params: Tuple[float, ...], /) -> None:
        if len(params) != 5:
            raise KnownOptimStateError(
                f"The argument tuple has {len(params)} elements but needed 5!"
            )
        self.get_kwargs().lr = compare_min_max_float(
            params[0], self.min_values[0], self.max_values[0]
        )
        self.get_kwargs().max_iter = compare_min_max_int(
            params[1], self.min_values[1], self.max_values[1]
        )
        self.get_kwargs().max_eval = compare_min_max_int(
            params[2], self.min_values[2], self.max_values[2]
        )
        self.get_kwargs().tolerance_grad = compare_min_max_float(
            params[3], self.min_values[3], self.max_values[3]
        )
        self.get_kwargs().tolerance_change = compare_min_max_float(
            params[4], self.min_values[4], self.max_values[4]
        )


_LBFGSStateTypes: Final[Dict[str, Tuple[Callable[[str], object], str]]] = {
    f'{MainOptimSt.get_pre_arg()}lr': (
        lambda val: float(val) if float(val) > 0 else 0.0001,
        "float (>0)"
    ),
    f'{MainOptimSt.get_pre_arg()}max_iter': (
        lambda val: int(val) if int(val) >= 1 else 20,
        "int (>=1)"
    ),
    f'{MainOptimSt.get_pre_arg()}max_eval': (
        lambda val: int(val) if int(val) >= 1 else 25,
        "int (>=1)"
    ),
    f'{MainOptimSt.get_pre_arg()}tolerance_grad': (
        lambda val: float(val) if float(val) > 0 else 1e-5,
        "float (>0)"
    ),
    f'{MainOptimSt.get_pre_arg()}tolerance_change': (
        lambda val: float(val) if float(val) > 0 else 1e-9,
        "float (>0)"
    ),
    f'{MainOptimSt.get_pre_arg()}history_size': (
        lambda val: int(val) if int(val) >= 10 else 100,
        "int (>=10)"
    ),
    f'{MainOptimSt.get_pre_arg()}line_search_fn': (
        lambda val: str(val) if str(val) in _MODE else None,
        f"{_MODE} or None"
    ),
    f'{MainOptimSt.get_pre_arg()}max_lr': (
        lambda val: float(val) if float(val) > 0 else 0.0001,
        "float (>0)"
    ),
    f'{MainOptimSt.get_pre_arg()}max_h_iter': (
        lambda val: int(val) if int(val) >= 2 else 1000,
        "int (>=2)"
    ),
    f'{MainOptimSt.get_pre_arg()}max_h_eval': (
        lambda val: int(val) if int(val) >= 3 else 1250,
        "int (>=3)"
    )
}


def get_lbfgs_state_types() -> Dict[str, Tuple[Callable[[str], object], str]]:
    return _LBFGSStateTypes
