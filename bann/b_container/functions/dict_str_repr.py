# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Dict, Any, List, final, TypeVar, Tuple

from bann.b_container.errors.custom_erors import KnownPrintPError


@final
@dataclass
class LayerWiseArgsCon:
    default: Any
    layer_wise: Dict[str, Any]


def dict_string_repr(dictionary: Dict, /) -> str:
    state_string = '\n\t'.join(
        f"{str(key)}:\t{str(val)}" for key, val in dictionary.items() if val is not None
    )
    return state_string


_ValueType = TypeVar('_ValueType')


def _create_str_value(value: _ValueType, tested_once: bool, /) -> str:
    if tested_once and isinstance(value, LayerWiseArgsCon):
        first_default = f"{_create_str_value(value.default, False)}"
        first_sec = ';'.join(
            f"{d_k}:{_create_str_value(d_v, False)}" for d_k, d_v in value.layer_wise.items()
        )
        if first_sec:
            first_default += f";{first_sec}"
        return first_default
    if isinstance(value, LayerWiseArgsCon):
        raise KnownPrintPError("Should never!")
    if isinstance(value, (list, set, tuple)):
        return ','.join(
            _create_str_value(to_p, False) for to_p in value if to_p is not None
        )
    if isinstance(value, bool):
        if value:
            return "T"
        return "F"
    return str(value)


def dict_json_repr(dictionary: Dict, pre_str: str, /) -> str:
    res_first: List[Tuple[str, str]] = [
        (key, _create_str_value(val, True))
        for key, val in dictionary.items() if val is not None
    ]
    res: List[str] = [
        f"\"{pre_str}{str(res_values[0])}\":\t\"{res_values[1]}\","
        for res_values in res_first if res_values[1]
    ]
    state_string = '\n\t'.join(res_e for res_e in res if res_e).rstrip(',')
    return state_string
