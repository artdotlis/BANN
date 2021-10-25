# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import final, Dict, Tuple, Callable, Final, Optional, Union, Type

from bann.b_container.errors.custom_erors import KnownOptimStateError
from bann.b_container.functions.check_arg_complex import check_arg_tuple_single
from bann.b_container.functions.compare_min_max import AlwaysCompareNumElem, CompareNumElem, \
    compare_min_max_int, sort_tuple_int
from bann.b_container.functions.dict_str_repr import dict_json_repr
from bann.b_container.states.framework.interface.optim_state import OptimStateKwargs, \
    MainOptimSt


@final
@dataclass
class _SwitcherStateKwargs(OptimStateKwargs):
    sw_epoch: Tuple[int, ...] = tuple()
    # min max settings
    sw_epoch_min: Tuple[int, ...] = tuple()
    sw_epoch_max: Tuple[int, ...] = tuple()
    # on off settings
    sw_epoch_on: Tuple[bool, ...] = tuple()


@final
class SWState(MainOptimSt[_SwitcherStateKwargs]):

    def __init__(self) -> None:
        super().__init__()
        self.__kwargs: Optional[_SwitcherStateKwargs] = None

    def get_kwargs_repr(self, _: int, /) -> str:
        return dict_json_repr(self.get_kwargs().__dict__, self.get_pre_arg())

    @property
    def type_values(self) -> Tuple[Union[Type[int], Type[float]], ...]:
        if self.get_kwargs().sw_epoch_on:
            epoch_l = [int for n_val in self.get_kwargs().sw_epoch_on if n_val]
        else:
            epoch_l = [int for _ in self.get_kwargs().sw_epoch]
        return tuple(epoch_l)

    @property
    def hyper_min_values(self) -> Tuple[AlwaysCompareNumElem, ...]:
        return tuple(AlwaysCompareNumElem(True, min_v) for min_v in self.get_kwargs().sw_epoch_min)

    @property
    def hyper_max_values(self) -> Tuple[AlwaysCompareNumElem, ...]:
        return tuple(
            AlwaysCompareNumElem(True, max_d) for max_d in self.get_kwargs().sw_epoch_max
        )

    @property
    def min_values(self) -> Tuple[CompareNumElem, ...]:
        return tuple(
            CompareNumElem(True, 1.0) for _ in self.get_kwargs().sw_epoch_min
        )

    @property
    def max_values(self) -> Tuple[CompareNumElem, ...]:
        return tuple(
            CompareNumElem(True, None) for _ in self.get_kwargs().sw_epoch_max
        )

    def get_kwargs(self) -> _SwitcherStateKwargs:
        if self.__kwargs is None:
            raise KnownOptimStateError("Kwargs not set!")
        return self.__kwargs

    def set_kwargs(self, args_dict: Dict, /) -> None:
        if self.__kwargs is not None:
            raise KnownOptimStateError("Kwargs already set!")
        self.__kwargs = super().parse_dict(_SwitcherStateKwargs, args_dict)
        self.set_new_hyper_param(self.get_hyper_param())
        if self.__kwargs.sw_epoch_on:
            if len(self.__kwargs.sw_epoch) != len(self.__kwargs.sw_epoch_on):
                raise KnownOptimStateError(
                    f"Mismatching length for net sizes sw_epoch {len(self.__kwargs.sw_epoch)}"
                    + f" and sw_epoch_on {len(self.__kwargs.sw_epoch_on)}"
                )
            if sum(self.__kwargs.sw_epoch_on) != len(self.__kwargs.sw_epoch_max) or \
                    sum(self.__kwargs.sw_epoch_on) != len(self.__kwargs.sw_epoch_min):
                raise KnownOptimStateError(
                    f"Mismatching length for sw_epoch to be optimised: "
                    + f"expected {sum(self.__kwargs.sw_epoch_on)} got "
                    + f"min;max {len(self.__kwargs.sw_epoch_min)};"
                    + f"{len(self.__kwargs.sw_epoch_max)}"
                )
        elif len(self.__kwargs.sw_epoch) != len(self.__kwargs.sw_epoch_max) \
                or len(self.__kwargs.sw_epoch) != len(self.__kwargs.sw_epoch_min):
            raise KnownOptimStateError(
                f"Mismatching length for sw_epoch {len(self.__kwargs.sw_epoch)}"
                + f" min;max {len(self.__kwargs.sw_epoch_min)};{len(self.__kwargs.sw_epoch_max)}"
            )
        self.__kwargs.sw_epoch_min = sort_tuple_int(self.__kwargs.sw_epoch_min)
        self.__kwargs.sw_epoch_max = sort_tuple_int(self.__kwargs.sw_epoch_max)
        self.__kwargs.sw_epoch_max = tuple(
            int(sw_min) if sw_min >= self.__kwargs.sw_epoch_max[sw_index]
            else int(self.__kwargs.sw_epoch_max[sw_index])
            for sw_index, sw_min in enumerate(self.__kwargs.sw_epoch_min)
        )

    def get_hyper_param(self) -> Tuple[float, ...]:
        if self.get_kwargs().sw_epoch_on:
            sw_e_l = [
                float(sw_ep) for sw_index, sw_ep in enumerate(self.get_kwargs().sw_epoch)
                if self.get_kwargs().sw_epoch_on[sw_index]
            ]
        else:
            sw_e_l = [float(sw_ep) for sw_ep in self.get_kwargs().sw_epoch]
        return tuple(sw_e_l)

    def set_new_hyper_param(self, params: Tuple[float, ...], /) -> None:
        p_len = sum(self.get_kwargs().sw_epoch_on) if self.get_kwargs().sw_epoch_on \
            else len(self.get_kwargs().sw_epoch)
        if p_len != len(params):
            raise KnownOptimStateError(f"Expected len {p_len} got len {len(params)}")
        if len(self.min_values) != len(self.max_values) or p_len != len(self.min_values):
            raise KnownOptimStateError(
                f"Mismatch in min:max {len(self.min_values)}:{len(self.max_values)} ({p_len})"
            )
        if self.get_kwargs().sw_epoch_on:
            buf_list_f = []
            last_opt = 0
            for sw_index, sw_on in enumerate(self.get_kwargs().sw_epoch_on):
                if sw_on:
                    buf_list_f.append(compare_min_max_int(
                        params[last_opt], self.min_values[last_opt], self.max_values[last_opt]
                    ))
                    last_opt += 1
                else:
                    buf_list_f.append(self.get_kwargs().sw_epoch[sw_index])
            self.get_kwargs().sw_epoch = tuple(buf_list_f)
        else:
            self.get_kwargs().sw_epoch = tuple(
                compare_min_max_int(
                    params[x_index], self.min_values[x_index], self.max_values[x_index]
                )
                for x_index in range(len(self.get_kwargs().sw_epoch))
            )
        self.get_kwargs().sw_epoch = sort_tuple_int(tuple(
            int(sw_e) if sw_e >= 1 else 1
            for sw_e in self.get_kwargs().sw_epoch
        ))


_SwitcherStateTypes: Final[Dict[str, Tuple[Callable[[str], object], str]]] = {
    f'{MainOptimSt.get_pre_arg()}sw_epoch': (lambda val: check_arg_tuple_single(
        val, lambda val1: int(val1) if int(val1) >= 1 else 1
    ), "Tuple[int, ...] as str,..."),
    f'{MainOptimSt.get_pre_arg()}sw_epoch_max': (lambda val: check_arg_tuple_single(
        val, lambda val1: int(val1) if int(val1) >= 2 else 2
    ), "Tuple[int, ...] as str,..."),
    f'{MainOptimSt.get_pre_arg()}sw_epoch_min': (lambda val: check_arg_tuple_single(
        val, lambda val1: int(val1) if int(val1) >= 1 else 1
    ), "Tuple[int, ...] as str,..."),
    f'{MainOptimSt.get_pre_arg()}sw_epoch_on': (lambda val: check_arg_tuple_single(
        val, lambda swi: swi == 'T'
    ), "Tuple[bool, ...] as str(T or F),...")
}


def get_switcher_state_types() -> Dict[str, Tuple[Callable[[str], object], str]]:
    return _SwitcherStateTypes
