# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import Dict, Tuple, Optional, Union, Type, List

from bann.b_hyper_optim.errors.custom_erors import KnownHyperError
from bann.b_container.functions.compare_min_max import AlwaysCompareNumElem
from bann.b_hyper_optim.optimiser.fun_const.pso_const import HyperExtras
from bann.b_hyper_optim.hyper_optim_interface import HyperOptimReturnElem
from bann.b_container.functions.dict_str_repr import dict_string_repr
from bann.b_container.states.general.interface.hyper_optim_state import OptimHyperState
from bann.b_hyper_optim.fun_const_wr.hyper_const import FlatTupleT
from rewowr.public.functions.syncout_dep_functions import logger_print_to_console
from rewowr.public.interfaces.logger_interface import SyncStdoutInterface


def h_create_flat_params(hyper_args: Dict[str, Tuple[float, ...]], /) -> FlatTupleT:
    list_str = []
    list_int = []
    list_str_flat = []
    list_str_counter: List[int] = []
    sum_len = 0
    for hyp_key, hyp_val in hyper_args.items():
        list_str.append(hyp_key)
        list_int.append(len(hyp_val))
        list_str_flat.extend(hyp_key for _ in range(len(hyp_val)))
        list_str_counter.extend(range(len(hyp_val)))
        sum_len += len(hyp_val)
    return FlatTupleT(
        list_str=tuple(list_str),
        list_int=tuple(list_int),
        flat=tuple(list_str_flat),
        flat_cnt=tuple(list_str_counter),
        sum_el=sum_len
    )


def h_print_to_logger(hyper_name: str, sync_out: SyncStdoutInterface,
                      hyper_stats: OptimHyperState, /) -> None:
    output_string = f"The arguments given to {hyper_name}:\n"
    output_string += f"\t{dict_string_repr(hyper_stats.get_kwargs().__dict__)}"
    logger_print_to_console(sync_out, output_string)


def h_extra_val(type_val: Union[Type[float], Type[int]], dif: float, /) -> float:
    if type_val == int:
        return 1
    if type_val == float:
        return dif if 0 < dif < 1e-8 else 4.5e-8
    raise KnownHyperError("Wrong type should never happen!")


def h_check_end_fitness(fitness_current: float, fitness_max: Optional[float], /) -> float:
    if fitness_max is None:
        return True
    if fitness_current > fitness_max:
        return True
    return False


def h_map_tuple_to_dict(state_types: Dict[str, str], tuple_to_unfold: Tuple[float, ...],
                        flat_param: FlatTupleT, /) -> Dict[str, HyperOptimReturnElem]:
    new_params: Dict[str, HyperOptimReturnElem] = {}
    running_sum = 0
    for dict_index, dict_name in enumerate(flat_param.list_str):
        new_params[dict_name] = HyperOptimReturnElem(
            param=tuple_to_unfold[running_sum: running_sum + flat_param.list_int[dict_index]],
            state_type=state_types[dict_name]
        )
        running_sum += flat_param.list_int[dict_index]
    return new_params


def h_map_dict_to_tuple(dict_to_p: Dict[str, HyperOptimReturnElem],
                        flat_param: FlatTupleT, /) -> Tuple[float, ...]:
    return tuple(
        value
        for dict_key in flat_param.list_str
        for value in dict_to_p[dict_key].param
    )


def h_map_f_dict_to_tuple(dict_to_p: Dict[str, Tuple[float, ...]],
                          flat_param: FlatTupleT, /) -> Tuple[float, ...]:
    return tuple(
        value
        for dict_key in flat_param.list_str
        for value in dict_to_p[dict_key]
    )


def h_create_hyper_space(max_space: Dict[str, Tuple[AlwaysCompareNumElem, ...]],
                         min_space: Dict[str, Tuple[AlwaysCompareNumElem, ...]],
                         space_type: Dict[str, Tuple[Union[Type[float], Type[int]], ...]],
                         flat_param: FlatTupleT, /) -> HyperExtras:
    max_space_l: List[float] = []
    min_space_l: List[float] = []
    space_types_l: List[Union[Type[float], Type[int]]] = []
    for dict_key in flat_param.list_str:
        dif_tuple = tuple(
            max_space[dict_key][min_index].value - min_v.value
            for min_index, min_v in enumerate(min_space[dict_key])
        )
        min_vals = tuple(
            min_v.value
            if min_v.equal
            else min_v.value + h_extra_val(space_type[dict_key][min_index], dif_tuple[min_index])
            for min_index, min_v in enumerate(min_space[dict_key])
        )
        max_vals = tuple(
            max_v.value
            if max_v.equal
            else max_v.value - h_extra_val(space_type[dict_key][max_index], dif_tuple[max_index])
            for max_index, max_v in enumerate(max_space[dict_key])
        )
        space_types_l.extend(space_type[dict_key])
        max_space_l.extend(max_vals)
        min_space_l.extend(min_vals)
    return HyperExtras(
        search_space_type=tuple(space_types_l),
        search_space_max=tuple(max_space_l),
        search_space_min=tuple(min_space_l)
    )
