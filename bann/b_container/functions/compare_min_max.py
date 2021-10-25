# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Optional, final, Tuple

from bann.b_container.errors.custom_erors import KnownMinMaxError


@final
@dataclass
class CompareNumElem:
    equal: bool
    value: Optional[float]


@final
@dataclass
class AlwaysCompareNumElem:
    equal: bool
    value: float


def _compare_min_float_eq(com_value: float, min_val: float, /) -> float:
    if com_value < min_val:
        return min_val

    return com_value


def _compare_min_float(com_value: float, min_val: float, to_add: float, /) -> float:
    if com_value <= min_val:
        return min_val + to_add

    return com_value


def _compare_max_float_eq(com_value: float, max_val: float, /) -> float:
    if com_value > max_val:
        return max_val

    return com_value


def _compare_max_float(com_value: float, max_val: float, to_sub: float, /) -> float:
    if com_value >= max_val:
        return max_val - to_sub

    return com_value


def compare_min_max_float(com_value: float, min_val: CompareNumElem, max_val: CompareNumElem, /) \
        -> float:
    erg = com_value
    if not (max_val.value is None or min_val.value is None) and (
            min_val.value > max_val.value
            or (not (min_val.equal or max_val.equal) and min_val.value >= max_val.value)
    ):
        raise KnownMinMaxError(f"Min: ({min_val.value}) >= max: ({max_val.value})")
    if min_val.value is not None:
        if min_val.equal:
            erg = _compare_min_float_eq(erg, min_val.value)
        else:
            to_add = 4.5e-8
            if max_val.value is not None:
                dif = (max_val.value - min_val.value) / 10000
                to_add = dif if 0 < dif < to_add else to_add
            erg = _compare_min_float(erg, min_val.value, to_add)

    if max_val.value is not None:
        if max_val.equal:
            erg = _compare_max_float_eq(erg, max_val.value)
        else:
            to_add = 4.5e-8
            if min_val.value is not None:
                dif = (max_val.value - min_val.value) / 10000
                to_add = dif if 0 < dif < to_add else to_add
            erg = _compare_max_float(erg, max_val.value, to_add)

    return erg


def compare_min_max_int(com_value: float, min_val: CompareNumElem, max_val: CompareNumElem, /) \
        -> int:
    erg = float(round(com_value))
    if not (max_val.value is None or min_val.value is None):
        if (
                max_val.value - min_val.value < 0
                or (not (min_val.equal or max_val.equal) and max_val.value - min_val.value < 2)
                or ((min_val.equal != max_val.equal) and max_val.value - min_val.value < 1)
        ):
            raise KnownMinMaxError(f"Min: ({min_val.value}) >= max: ({max_val.value})")
    if min_val.value is not None:
        if min_val.equal:
            erg = _compare_min_float_eq(erg, round(min_val.value))
        else:
            erg = _compare_min_float(erg, round(min_val.value), 1.0)

    if max_val.value is not None:
        if max_val.equal:
            erg = _compare_max_float_eq(erg, round(max_val.value))
        else:
            erg = _compare_max_float(erg, round(max_val.value), 1.0)
    return round(erg)


def sort_tuple_int(to_sort: Tuple[int, ...], /) -> Tuple[int, ...]:
    if len(to_sort) < 2:
        return to_sort
    sorted_list = sorted(to_sort)
    for s_i, fix_el in enumerate(sorted_list[1:], 1):
        if sorted_list[s_i - 1] >= fix_el:
            sorted_list[s_i] = sorted_list[s_i - 1] + 1
    return tuple(sorted_list)
