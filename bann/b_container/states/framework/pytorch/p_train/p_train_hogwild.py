# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Union, Type, Callable, final, Final

from bann.b_container.functions.dict_str_repr import dict_json_repr
from bann.b_container.states.framework.pytorch.p_train.p_train_constants import TrainGenCon, \
    get_train_general_state_types
from bann.b_container.errors.custom_erors import KnownTrainStateError
from bann.b_container.functions.compare_min_max import CompareNumElem, compare_min_max_int
from bann.b_container.states.framework.interface.train_state import TrainState
from bann.b_container.functions.compare_min_max import AlwaysCompareNumElem


@final
@dataclass
class _HogWildCon(TrainGenCon):
    tr_worker: int = 1
    split_data: bool = False


@final
class HogwildTState(TrainState[_HogWildCon]):

    def __init__(self) -> None:
        super().__init__()
        self.__kwargs: Optional[_HogWildCon] = None

    def get_kwargs_repr(self) -> str:
        return dict_json_repr(self.get_kwargs().__dict__, self.get_pre_arg())

    @property
    def type_values(self) -> Tuple[Union[Type[int], Type[float]], ...]:
        erg = (int, *[int for _ in self.get_kwargs().batch_size])
        return erg

    @property
    def hyper_min_values(self) -> Tuple[AlwaysCompareNumElem, ...]:
        min_batch = self.get_kwargs().min_batch_size
        if min_batch > self.get_kwargs().max_batch_size:
            min_batch = self.get_kwargs().max_batch_size
        min_epoch = self.get_kwargs().min_epoch_size
        if min_epoch > self.get_kwargs().max_epoch_size:
            min_epoch = self.get_kwargs().max_epoch_size
        erg = (
            AlwaysCompareNumElem(True, min_epoch),
            *[AlwaysCompareNumElem(True, min_batch) for _ in self.get_kwargs().batch_size]
        )
        return erg

    @property
    def hyper_max_values(self) -> Tuple[AlwaysCompareNumElem, ...]:
        erg = (
            AlwaysCompareNumElem(True, self.get_kwargs().max_epoch_size),
            *[
                AlwaysCompareNumElem(True, self.get_kwargs().max_batch_size)
                for _ in self.get_kwargs().batch_size
            ]
        )
        return erg

    @property
    def min_values(self) -> Tuple[CompareNumElem, ...]:
        min_epoch = self.get_kwargs().min_epoch_size
        if min_epoch > self.get_kwargs().max_epoch_size:
            min_epoch = self.get_kwargs().max_epoch_size
        min_batch = self.get_kwargs().min_batch_size
        if min_batch > self.get_kwargs().max_batch_size:
            min_batch = self.get_kwargs().max_batch_size
        erg = (
            CompareNumElem(True, 1 if min_epoch < 2 else min_epoch),
            *[CompareNumElem(True, 1 if min_batch < 2 else min_batch)
              for _ in self.get_kwargs().batch_size]
        )
        return erg

    @property
    def max_values(self) -> Tuple[CompareNumElem, ...]:
        max_batch = self.get_kwargs().max_batch_size
        if max_batch < self.get_kwargs().min_batch_size:
            max_batch = self.get_kwargs().min_batch_size
        return (
            CompareNumElem(True, None),
            *[CompareNumElem(True, max_batch if max_batch > 2 else 2)
              for _ in self.get_kwargs().batch_size]
        )

    def get_kwargs(self) -> _HogWildCon:
        if self.__kwargs is None:
            raise KnownTrainStateError("Kwargs not set!")
        return self.__kwargs

    def set_kwargs(self, args_dict: Dict, /) -> None:
        if self.__kwargs is not None:
            raise KnownTrainStateError("Kwargs already set!")
        self.__kwargs = super().parse_dict(_HogWildCon, args_dict)
        if not len(self.get_kwargs().batch_size):
            self.__kwargs.batch_size = (100,)
        self.set_new_hyper_param(self.get_hyper_param())

    def get_hyper_param(self) -> Tuple[float, ...]:
        return (
            float(self.get_kwargs().epoch_size),
            *[float(batch_e) for batch_e in self.get_kwargs().batch_size]
        )

    def set_new_hyper_param(self, params: Tuple[float, ...], /) -> None:
        if len(params) != 1 + len(self.get_kwargs().batch_size):
            raise KnownTrainStateError(
                f"Expected len {1 + len(self.get_kwargs().batch_size)} got len {len(params)}"
            )
        self.get_kwargs().epoch_size = compare_min_max_int(
            params[0], self.min_values[0], self.max_values[0]
        )
        self.get_kwargs().batch_size = tuple(
            compare_min_max_int(
                params[p_i], self.min_values[p_i], self.max_values[p_i]
            ) for p_i in range(1, len(params))
        )


_TrainStateTypes: Final[Dict[str, Tuple[Callable[[str], object], str]]] = {
    f'{TrainState.get_pre_arg()}tr_worker': (
        lambda val: int(val) if int(val) >= 1 else 1,
        "int (>=1)"
    ),
    f'{TrainState.get_pre_arg()}split_data': (
        lambda val: val == 'T',
        "True if T else False"
    )
}


def get_train_hogwild_state_types() -> Dict[str, Tuple[Callable[[str], object], str]]:
    bann_gen_trainer = get_train_general_state_types()
    merged_dict = {**_TrainStateTypes}
    for key, value in bann_gen_trainer.items():
        if key in merged_dict:
            raise KnownTrainStateError(f"Duplicated key {key} in {HogwildTState.__name__}")
        merged_dict[key] = value
    return merged_dict
