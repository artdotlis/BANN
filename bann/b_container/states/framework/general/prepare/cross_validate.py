# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import final, Optional, Dict, Final, Tuple, Callable

from bann.b_container.errors.custom_erors import KnownPrepareError
from bann.b_container.functions.dict_str_repr import dict_json_repr
from bann.b_container.states.framework.interface.prepare_state import PrepareStateKwargs, \
    PrepareState


@final
@dataclass
class _CrossCon(PrepareStateKwargs):
    k_folds: int = 10
    n_repeats: int = 1
    bias_var_f: float = 0.5
    cross_p: int = 1
    eval_on: bool = True


@final
class CrossValTState(PrepareState[_CrossCon]):
    def __init__(self) -> None:
        super().__init__()
        self.__kwargs: Optional[_CrossCon] = None

    def get_kwargs_repr(self) -> str:
        return dict_json_repr(self.get_kwargs().__dict__, self.get_pre_arg())

    def get_kwargs(self) -> _CrossCon:
        if self.__kwargs is None:
            raise KnownPrepareError("Kwargs not set!")
        return self.__kwargs

    def set_kwargs(self, args_dict: Dict, /) -> None:
        if self.__kwargs is not None:
            raise KnownPrepareError("Kwargs already set!")
        self.__kwargs = super().parse_dict(_CrossCon, args_dict)


_PrepareStateTypes: Final[Dict[str, Tuple[Callable[[str], object], str]]] = {
    f'{PrepareState.get_pre_arg()}k_folds': (
        lambda val: int(val) if int(val) >= 2 else 10,
        "int (>=2)"
    ),
    f'{PrepareState.get_pre_arg()}n_repeats': (
        lambda val: int(val) if int(val) >= 1 else 1,
        "int (>=1)"
    ),
    f'{PrepareState.get_pre_arg()}bias_var_f': (
        lambda val: float(val) if 0 <= float(val) <= 1 else 0.5,
        "float (0<=x<=1)"
    ),
    f'{PrepareState.get_pre_arg()}cross_p': (
        lambda val: int(val) if int(val) >= 1 else 1,
        "int (>=1)"
    ),
    f'{PrepareState.get_pre_arg()}eval_on': (
        lambda val: val == 'T',
        "True if T else False"
    )
}


def get_prepare_cross_state_types() -> Dict[str, Tuple[Callable[[str], object], str]]:
    return _PrepareStateTypes
