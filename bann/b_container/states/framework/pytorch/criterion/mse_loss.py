# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Callable, Final, final

from bann.b_frameworks.pytorch.p_criterion.p_reduction_std_enum import \
    get_loss_st_reduction_lib
from bann.b_container.errors.custom_erors import KnownCriterionStateError
from bann.b_container.functions.dict_str_repr import dict_json_repr
from bann.b_container.states.framework.interface.criterion_state import CriterionKwargs, \
    CriterionState


_LIB_REDUCTION: Final[Tuple[str, ...]] = get_loss_st_reduction_lib()


@final
@dataclass
class _MSELossStateKwargs(CriterionKwargs):
    reduction: str = _LIB_REDUCTION[1]

    @property
    def get_criterion_dict(self) -> Dict:
        return {'reduction': self.reduction}


@final
class MSELossState(CriterionState[_MSELossStateKwargs]):

    def __init__(self) -> None:
        super().__init__()
        self.__kwargs: Optional[_MSELossStateKwargs] = None

    def get_kwargs_repr(self) -> str:
        return dict_json_repr(self.get_kwargs().__dict__, self.get_pre_arg())

    def get_kwargs(self) -> _MSELossStateKwargs:
        if self.__kwargs is None:
            raise KnownCriterionStateError("Kwargs not set!")
        return self.__kwargs

    def set_kwargs(self, args_dict: Dict, /) -> None:
        if self.__kwargs is not None:
            raise KnownCriterionStateError("Kwargs already set!")
        self.__kwargs = super().parse_dict(_MSELossStateKwargs, args_dict)


_MSELossStateTypes: Final[Dict[str, Tuple[Callable[[str], object], str]]] = {
    f'{CriterionState.get_pre_arg()}reduction': (
        lambda val: str(val) if val in _LIB_REDUCTION else _LIB_REDUCTION[1],
        f"str ({','.join(_LIB_REDUCTION)})"
    )
}


def get_mse_loss_state_types() -> Dict[str, Tuple[Callable[[str], object], str]]:
    return _MSELossStateTypes
