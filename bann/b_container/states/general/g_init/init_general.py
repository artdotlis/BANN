# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import abc
from pathlib import Path
from typing import Optional, Tuple, Callable, Dict, Union, Type, TypeVar, final, Final
from dataclasses import dataclass

from bann.b_data_functions.pytorch.subset_dataset import check_subset_size
from bann.b_container.functions.dict_str_repr import dict_json_repr
from bann.b_container.functions.check_arg_complex import check_arg_tuple_single
from bann.b_container.errors.custom_erors import KnownInitStateError
from bann.b_container.functions.compare_min_max import CompareNumElem, compare_min_max_float, \
    compare_min_max_int
from bann.b_container.states.general.interface.init_state import InitState, InitStateKwargs
from bann.b_container.functions.compare_min_max import AlwaysCompareNumElem


@dataclass
class GenInitStateCon(InitStateKwargs):
    subset_size: Optional[int] = None
    input_train: Optional[Path] = None
    input_eval: Optional[Path] = None
    input_test: Optional[Path] = None
    input_net: Optional[Path] = None
    drop_rate: Tuple[float, ...] = tuple()
    net_sizes: Tuple[int, ...] = tuple()
    # min max settings
    drop_rate_max: Tuple[float, ...] = tuple()
    net_sizes_min: Tuple[int, ...] = tuple()
    net_sizes_max: Tuple[int, ...] = tuple()
    # optim on off
    net_sizes_on: Tuple[bool, ...] = tuple()
    # drop rate on off
    drop_rate_on: Tuple[bool, ...] = tuple()


_InitType = TypeVar('_InitType', bound=GenInitStateCon)


class InitGeneralStateInterface(InitState[_InitType], abc.ABC):

    def __init__(self) -> None:
        super().__init__()
        self.__kwargs: Optional[_InitType] = None

    @final
    def disable_subset(self) -> None:
        self.get_kwargs().subset_size = None

    @final
    @property
    def type_values(self) -> Tuple[Union[Type[int], Type[float]], ...]:
        if self.get_kwargs().drop_rate_on:
            dr_l = [float for n_val in self.get_kwargs().drop_rate_on if n_val]
        else:
            dr_l = [float for _ in self.get_kwargs().drop_rate]
        if self.get_kwargs().net_sizes_on:
            nets_l = [int for n_val in self.get_kwargs().net_sizes_on if n_val]
        else:
            nets_l = [int for _ in self.get_kwargs().net_sizes]
        return *dr_l, *nets_l

    @final
    @property
    def hyper_min_values(self) -> Tuple[AlwaysCompareNumElem, ...]:
        dr_l = [AlwaysCompareNumElem(True, 0.0) for _ in self.get_kwargs().drop_rate_max]
        min_n = [AlwaysCompareNumElem(True, min_s) for min_s in self.get_kwargs().net_sizes_min]
        return *dr_l, *min_n

    @final
    @property
    def hyper_max_values(self) -> Tuple[AlwaysCompareNumElem, ...]:
        dr_l = [AlwaysCompareNumElem(True, max_d) for max_d in self.get_kwargs().drop_rate_max]
        max_n = [AlwaysCompareNumElem(True, max_s) for max_s in self.get_kwargs().net_sizes_max]
        return *dr_l, *max_n

    @final
    @property
    def min_values(self) -> Tuple[CompareNumElem, ...]:
        dr_l = [CompareNumElem(True, 0.0) for _ in self.get_kwargs().drop_rate_max]
        min_n = [CompareNumElem(True, min_s) for min_s in self.get_kwargs().net_sizes_min]
        return *dr_l, *min_n

    @final
    @property
    def max_values(self) -> Tuple[CompareNumElem, ...]:
        dr_l = [CompareNumElem(True, max_d) for max_d in self.get_kwargs().drop_rate_max]
        max_n = [CompareNumElem(True, max_s) for max_s in self.get_kwargs().net_sizes_max]
        return *dr_l, *max_n

    @final
    def get_kwargs(self) -> _InitType:
        if self.__kwargs is None:
            raise KnownInitStateError("Kwargs not set!")
        return self.__kwargs

    @final
    def get_hyper_param(self) -> Tuple[float, ...]:
        if self.get_kwargs().drop_rate_on:
            dr_l = tuple(
                dr_r for dr_i, dr_r in enumerate(self.get_kwargs().drop_rate)
                if self.get_kwargs().drop_rate_on[dr_i]
            )
        else:
            dr_l = self.get_kwargs().drop_rate
        if self.get_kwargs().net_sizes_on:
            net_s = [
                float(net) for net_index, net in enumerate(self.get_kwargs().net_sizes)
                if self.get_kwargs().net_sizes_on[net_index]
            ]
        else:
            net_s = [float(net) for net in self.get_kwargs().net_sizes]
        return *dr_l, *net_s

    @final
    def set_new_hyper_param(self, params: Tuple[float, ...], /) -> None:
        p_len = sum(self.get_kwargs().drop_rate_on) if self.get_kwargs().drop_rate_on \
            else len(self.get_kwargs().drop_rate)
        p_len += sum(self.get_kwargs().net_sizes_on) if self.get_kwargs().net_sizes_on \
            else len(self.get_kwargs().net_sizes)
        if p_len != len(params):
            raise KnownInitStateError(f"Expected len {p_len} got len {len(params)}")
        if len(self.min_values) != len(self.max_values) or p_len != len(self.min_values):
            raise KnownInitStateError(
                f"Mismatch in min:max {len(self.min_values)}:{len(self.max_values)} ({p_len})"
            )
        if self.get_kwargs().drop_rate_on:
            buf_list_f = []
            last_opt = 0
            for dr_index, dr_on in enumerate(self.get_kwargs().drop_rate_on):
                if dr_on:
                    buf_list_f.append(compare_min_max_float(
                        params[last_opt], self.min_values[last_opt], self.max_values[last_opt]
                    ))
                    last_opt += 1
                else:
                    buf_list_f.append(self.get_kwargs().drop_rate[dr_index])
            self.get_kwargs().drop_rate = tuple(buf_list_f)
        else:
            self.get_kwargs().drop_rate = tuple(
                compare_min_max_float(
                    params[x_index], self.min_values[x_index], self.max_values[x_index]
                )
                for x_index in range(len(self.get_kwargs().drop_rate))
            )
        if self.get_kwargs().net_sizes_on:
            buf_list = []
            last_opt = len(self.get_kwargs().drop_rate)
            for n_index, n_on in enumerate(self.get_kwargs().net_sizes_on):
                if n_on:
                    buf_list.append(compare_min_max_int(
                        params[last_opt], self.min_values[last_opt], self.max_values[last_opt]
                    ))
                    last_opt += 1
                else:
                    buf_list.append(self.get_kwargs().net_sizes[n_index])
            self.get_kwargs().net_sizes = tuple(buf_list)
        else:
            self.get_kwargs().net_sizes = tuple(
                compare_min_max_int(
                    params[x_index], self.min_values[x_index], self.max_values[x_index]
                )
                for x_index in range(len(self.get_kwargs().drop_rate), p_len)
            )

    @final
    def set_kwargs_interface(self, args_dict: Dict, cont_type: Type[_InitType], /) -> None:
        if self.__kwargs is not None:
            raise KnownInitStateError("Kwargs already set!")
        self.__kwargs = super().parse_dict(cont_type, args_dict)
        self.set_new_hyper_param(self.get_hyper_param())
        self.__kwargs.net_sizes = tuple(
            int(net_s) if net_s >= 1 else 1
            for net_s in self.__kwargs.net_sizes
        )
        self.__kwargs.drop_rate = tuple(
            float(dr_r) if 0 <= dr_r <= 1 else 0.0
            for dr_r in self.__kwargs.drop_rate
        )
        if self.__kwargs.drop_rate_on:
            if len(self.__kwargs.drop_rate) != len(self.__kwargs.drop_rate_on):
                raise KnownInitStateError(
                    f"Mismatching length for drop rates {len(self.__kwargs.drop_rate)}"
                    + f" and drop_rate_on {len(self.__kwargs.drop_rate_on)}"
                )
            if sum(self.__kwargs.drop_rate_on) != len(self.__kwargs.drop_rate_max):
                raise KnownInitStateError(
                    f"Mismatching length for drop rates max: {len(self.__kwargs.drop_rate_max)}"
                    + f" and drop_rate_on: {sum(self.__kwargs.drop_rate_max)}"
                )
        elif len(self.__kwargs.drop_rate) != len(self.__kwargs.drop_rate_max):
            raise KnownInitStateError(
                f"Mismatching length for drop rates {len(self.__kwargs.drop_rate)}"
                + f" drop rates max {len(self.__kwargs.drop_rate_max)}"
            )
        if self.__kwargs.net_sizes_on:
            if len(self.__kwargs.net_sizes) != len(self.__kwargs.net_sizes_on):
                raise KnownInitStateError(
                    f"Mismatching length for net sizes n_size {len(self.__kwargs.net_sizes)}"
                    + f" and net_sizes_on {len(self.__kwargs.net_sizes_on)}"
                )
            if sum(self.__kwargs.net_sizes_on) != len(self.__kwargs.net_sizes_max) or \
                    sum(self.__kwargs.net_sizes_on) != len(self.__kwargs.net_sizes_min):
                raise KnownInitStateError(
                    f"Mismatching length for net sizes to be optimised: "
                    + f"expected {sum(self.__kwargs.net_sizes_on)} got "
                    + f"min;max {len(self.__kwargs.net_sizes_min)};"
                    + f"{len(self.__kwargs.net_sizes_max)}"
                )
        elif len(self.__kwargs.net_sizes) != len(self.__kwargs.net_sizes_max) \
                or len(self.__kwargs.net_sizes) != len(self.__kwargs.net_sizes_min):
            raise KnownInitStateError(
                f"Mismatching length for net sizes n_size {len(self.__kwargs.net_sizes)}"
                + f" min;max {len(self.__kwargs.net_sizes_min)};{len(self.__kwargs.net_sizes_max)}"
            )

        self.__kwargs.net_sizes_max = tuple(
            int(net_min) if net_min >= self.__kwargs.net_sizes_max[net_index]
            else int(self.__kwargs.net_sizes_max[net_index])
            for net_index, net_min in enumerate(self.__kwargs.net_sizes_min)
        )


@final
class InitGeneralState(InitGeneralStateInterface[GenInitStateCon]):
    def get_kwargs_repr(self) -> str:
        return dict_json_repr(self.get_kwargs().__dict__, self.get_pre_arg())

    def set_kwargs(self, args_dict: Dict) -> None:
        super().set_kwargs_interface(args_dict, GenInitStateCon)


_InitGeneralStateTypes: Final[Dict[str, Tuple[Callable[[str], object], str]]] = {
    f'{InitState.get_pre_arg()}subset_size': (check_subset_size, "int (>=1)"),
    f'{InitState.get_pre_arg()}input_train': (Path, "Path"),
    f'{InitState.get_pre_arg()}input_eval': (Path, "Path"),
    f'{InitState.get_pre_arg()}input_test': (Path, "Path"),
    f'{InitState.get_pre_arg()}input_net': (Path, "Path"),
    f'{InitState.get_pre_arg()}drop_rate': (lambda val: check_arg_tuple_single(
        val, lambda val1: float(val1) if 0.0 <= float(val1) <= 1.0 else 0.2
    ), "Tuple[float, ...] as str,..."),
    f'{InitState.get_pre_arg()}net_sizes': (lambda val: check_arg_tuple_single(
        val, lambda val1: int(val1) if int(val1) >= 1 else 1
    ), "Tuple[int, ...] as str,..."),
    f'{InitState.get_pre_arg()}net_sizes_min': (lambda val: check_arg_tuple_single(
        val, lambda val1: int(val1) if int(val1) >= 1 else 1
    ), "Tuple[int, ...] as str,..."),
    f'{InitState.get_pre_arg()}net_sizes_max': (lambda val: check_arg_tuple_single(
        val, lambda val1: int(val1) if int(val1) >= 2 else 2
    ), "Tuple[int, ...] as str,..."),
    f'{InitState.get_pre_arg()}drop_rate_max': (lambda val: check_arg_tuple_single(
        val, lambda val1: float(val1) if 0 <= float(val1) <= 1 else 0.5
    ), "Tuple[int, ...] as str,..."),
    f'{InitState.get_pre_arg()}net_sizes_on': (lambda val: check_arg_tuple_single(
        val, lambda save: save == 'T'
    ), "Tuple[bool, ...] as str(T or F),..."),
    f'{InitState.get_pre_arg()}drop_rate_on': (lambda val: check_arg_tuple_single(
        val, lambda save: save == 'T'
    ), "Tuple[bool, ...] as str(T or F),...")
}


def get_init_general_state_types() -> Dict[str, Tuple[Callable[[str], object], str]]:
    return _InitGeneralStateTypes
