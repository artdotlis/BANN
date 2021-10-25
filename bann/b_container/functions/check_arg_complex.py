# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import re

from typing import Pattern, Tuple, Callable, Final, TypeVar, Dict, Optional, Union, List, Match

from bann.b_container.errors.custom_erors import KnownCheckComError
from rewowr.public.functions.check_re_wr_wo_arguments import check_parse_type

_COMMA: Final[Pattern[str]] = re.compile(r',')
_TypeRValue = TypeVar('_TypeRValue')
_TypeRValue2 = TypeVar('_TypeRValue2')


def check_arg_tuple(arg: str, type_tuple: Tuple[Callable[[str], object], ...], /) -> Tuple:
    split_ergs = check_parse_type(arg, _COMMA.split)
    if len(type_tuple) != len(split_ergs):
        raise KnownCheckComError(
            f"Expected {len(type_tuple)} arguments got {len(split_ergs)}!"
        )
    return tuple(
        check_parse_type(param, type_tuple[param_index])
        for param_index, param in enumerate(split_ergs)
    )


def check_arg_tuple_single(arg: str, type_fun: Callable[[str], _TypeRValue], /) \
        -> Tuple[_TypeRValue, ...]:
    split_ergs = check_parse_type(arg, _COMMA.split)
    return tuple(
        check_parse_type(param, type_fun) for param in split_ergs
    )


def _check_comma(value: str, /) -> bool:
    res = _COMMA.search(value)
    if res is None:
        return False
    return True


def check_arg_tuple_or_scalar_single(value: str, type_fun: Callable[[str], _TypeRValue], /) \
        -> Union[float, Tuple[_TypeRValue, ...]]:
    if _check_comma(value):
        return check_arg_tuple_single(value, type_fun)
    return check_parse_type(value, type_fun)


def create_comma_split_fun(to_check: Tuple[str, ...], /) -> Callable[[str], Tuple[str, ...]]:
    def create_acts_tuple(args: str, /) -> Tuple[str, ...]:
        split_ergs = check_parse_type(args, _COMMA.split)
        return tuple(act for act in split_ergs if act in to_check)
    return create_acts_tuple


_COLON: Final[Pattern[str]] = re.compile(r':')
_SEMI_COLON: Final[Pattern[str]] = re.compile(r';')


def get_layer_wise_args(type_fun: Callable[[str], _TypeRValue], arg: str, /) \
        -> Tuple[Tuple[_TypeRValue, ...], Dict[str, Tuple[_TypeRValue, ...]]]:
    split_ergs = check_parse_type(arg, _SEMI_COLON.split)
    default_v = check_arg_tuple_single(split_ergs[0], type_fun)
    if len(split_ergs) == 1:
        return default_v, {}
    layer_w_values = [check_parse_type(argument, _COLON.split) for argument in split_ergs[1:]]
    for layer_e in layer_w_values:
        if len(layer_e) < 2:
            raise KnownCheckComError(
                f"Expected layer wise argument of at least two elements, got {len(layer_e)}!"
            )
    return default_v, {
        layer_e[0]: check_arg_tuple_single(layer_e[1], type_fun)
        for layer_e in layer_w_values
    }


def get_comma_two_tuple(arg: str, t_fun_1: Callable[[str], _TypeRValue],
                        t_fun_2: Callable[[str], _TypeRValue2], /) \
        -> Tuple[_TypeRValue, _TypeRValue2]:
    split_ergs = check_parse_type(arg, _COMMA.split)
    if len(split_ergs) != 2:
        raise KnownCheckComError(f"Expected a tuple got {arg}")
    first = check_arg_tuple_single(split_ergs[0], t_fun_1)[0]
    second = check_arg_tuple_single(split_ergs[1], t_fun_2)[0]
    return first, second


def get_comma_one_two_tuple(arg: str, t_fun_1: Callable[[str], _TypeRValue],
                            t_fun_2: Callable[[str], _TypeRValue2], /) \
        -> Tuple[_TypeRValue, Optional[_TypeRValue2]]:
    split_ergs = check_parse_type(arg, _COMMA.split)
    if len(split_ergs) > 2:
        raise KnownCheckComError(f"Expected a tuple got {arg}")
    first = check_arg_tuple_single(split_ergs[0], t_fun_1)[0]
    if len(split_ergs) == 2:
        second: Optional[_TypeRValue2] = check_arg_tuple_single(split_ergs[1], t_fun_2)[0]
    else:
        second = None
    return first, second


def get_comma_set(arg: str, t_fun: Callable[[str], _TypeRValue], /) -> Tuple[_TypeRValue, ...]:
    split_ergs = check_arg_tuple_single(arg, t_fun)
    if not split_ergs:
        raise KnownCheckComError(f"Expected a tuple with at least one element, got {arg}")
    return tuple(set(split_ergs))


def _check_pattern_and_return(val: str, pattern_list: List[Pattern[str]]) \
        -> Optional[Tuple[Pattern[str], Match[str], int]]:
    for pat_i, pat_el in enumerate(pattern_list):
        res = pat_el.search(val)
        if res is not None:
            return pat_el, res, pat_i
    return None


def check_pattern(val: str, pattern_list: List[Pattern[str]]) -> bool:
    if _check_pattern_and_return(val, pattern_list) is not None:
        return True
    return False


def get_pattern_match(val: str, pattern_list: List[Pattern[str]]) \
        -> Optional[Tuple[List[str], int]]:
    res = _check_pattern_and_return(val, pattern_list)
    if res is not None:
        return list(res[1].groups()), res[2]
    return None
