# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Optional, Dict, Callable, Tuple, Final, final

import torch

from bann.b_container.functions.check_arg_complex import get_comma_two_tuple
from bann.b_frameworks.pytorch.p_criterion.p_reduction_std_enum import \
    get_loss_st_reduction_lib
from bann.b_container.functions.dict_str_repr import dict_json_repr
from bann.b_container.errors.custom_erors import KnownCriterionStateError
from bann.b_container.states.framework.interface.criterion_state import CriterionKwargs, \
    CriterionState


_LIB_REDUCTION: Final[Tuple[str, ...]] = get_loss_st_reduction_lib()


@final
@dataclass
class _CrossEntropyLossStateKwargs(CriterionKwargs):
    reduction: str = _LIB_REDUCTION[1]
    weight: Tuple[float, ...] = tuple()
    ignore_index: Optional[int] = None

    @property
    def get_criterion_dict(self) -> Dict:
        return {
            'reduction': self.reduction,
            'weight': torch.tensor(self.weight) if self.weight else None,
            'ignore_index': self.ignore_index
        }


@final
class CrossEntropyLossState(CriterionState[_CrossEntropyLossStateKwargs]):

    def __init__(self) -> None:
        super().__init__()
        self.__kwargs: Optional[_CrossEntropyLossStateKwargs] = None

    def get_kwargs_repr(self) -> str:
        return dict_json_repr(self.get_kwargs().__dict__, self.get_pre_arg())

    def get_kwargs(self) -> _CrossEntropyLossStateKwargs:
        if self.__kwargs is None:
            raise KnownCriterionStateError("Kwargs not set!")
        return self.__kwargs

    def set_kwargs(self, args_dict: Dict, /) -> None:
        if self.__kwargs is not None:
            raise KnownCriterionStateError("Kwargs already set!")
        self.__kwargs = super().parse_dict(_CrossEntropyLossStateKwargs, args_dict)


_CrossEntropyLStateTypes: Final[Dict[str, Tuple[Callable[[str], object], str]]] = {
    f'{CriterionState.get_pre_arg()}reduction': (
        lambda val: str(val) if val in _LIB_REDUCTION else _LIB_REDUCTION[1],
        f"str ({','.join(_LIB_REDUCTION)})"
    ),
    f'{CriterionState.get_pre_arg()}ignore_index': (
        lambda val: int(val) if int(val) >= 0 else None,
        "int (>=0) else None"
    ),
    f'{CriterionState.get_pre_arg()}weight': (
        lambda a_v: get_comma_two_tuple(
            a_v, lambda v_1: float(v_1) if float(v_1) > 0 else 1,
            lambda v_1: float(v_1) if 0 < float(v_1) <= 1 else 1
        ),
        f"float,float (>0)"
    )
}


def get_cross_entropy_state_types() -> Dict[str, Tuple[Callable[[str], object], str]]:
    return _CrossEntropyLStateTypes
