# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import Dict, Tuple, Callable, final, Optional, Union, Type

from bann.b_container.errors.custom_erors import KnownTrainStateError
from bann.b_container.functions.compare_min_max import AlwaysCompareNumElem, CompareNumElem
from bann.b_container.functions.dict_str_repr import dict_json_repr
from bann.b_container.states.framework.interface.train_state import TrainState, TrainStateKwargs


@final
class LocalTState(TrainState[TrainStateKwargs]):

    def __init__(self) -> None:
        super().__init__()
        self.__kwargs: Optional[TrainStateKwargs] = None

    def get_kwargs_repr(self) -> str:
        return dict_json_repr(self.get_kwargs().__dict__, self.get_pre_arg())

    @property
    def type_values(self) -> Tuple[Union[Type[int], Type[float]], ...]:
        return ()

    @property
    def hyper_min_values(self) -> Tuple[AlwaysCompareNumElem, ...]:
        return ()

    @property
    def hyper_max_values(self) -> Tuple[AlwaysCompareNumElem, ...]:
        return ()

    @property
    def min_values(self) -> Tuple[CompareNumElem, ...]:
        return ()

    @property
    def max_values(self) -> Tuple[CompareNumElem, ...]:
        return ()

    def get_kwargs(self) -> TrainStateKwargs:
        if self.__kwargs is None:
            raise KnownTrainStateError("Kwargs not set!")
        return self.__kwargs

    def set_kwargs(self, args_dict: Dict, /) -> None:
        if self.__kwargs is not None:
            raise KnownTrainStateError("Kwargs already set!")
        self.__kwargs = super().parse_dict(TrainStateKwargs, args_dict)
        self.set_new_hyper_param(self.get_hyper_param())

    def get_hyper_param(self) -> Tuple[float, ...]:
        return ()

    def set_new_hyper_param(self, params: Tuple[float, ...], /) -> None:
        if params:
            raise KnownTrainStateError(f"Expected zero args got len {len(params)}")


def get_train_local_state_types() -> Dict[str, Tuple[Callable[[str], object], str]]:
    return {}
