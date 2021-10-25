# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import final, Union, Tuple, TypeVar, List

from bann.b_container.errors.custom_erors import KnownExtendError
from bann.b_container.functions.compare_min_max import AlwaysCompareNumElem, CompareNumElem


@final
@dataclass
class CToAddContainer:
    always: bool
    equal: bool
    data: Union[Tuple[float, ...], float]


def _create_comp_num_elem(always: bool, equal: bool, data: float, /) \
        -> Union[AlwaysCompareNumElem, CompareNumElem]:
    if always:
        return AlwaysCompareNumElem(equal, data)
    return CompareNumElem(equal, data)


_TCheck = TypeVar('_TCheck')
_TAdd = TypeVar('_TAdd')


def comp_extend_list(val_2_check: Union[_TCheck, Tuple[_TCheck, ...]],
                     list_to_extend: List, value_to_add: _TAdd, /) -> None:
    if isinstance(val_2_check, tuple):
        if isinstance(value_to_add, CToAddContainer):
            list_to_extend.extend(
                _create_comp_num_elem(
                    value_to_add.always, value_to_add.equal,
                    value_to_add.data[id_v] if isinstance(value_to_add.data, tuple)
                    else value_to_add.data
                )
                for id_v, _ in enumerate(val_2_check)
            )
        else:
            list_to_extend.extend(value_to_add for _ in val_2_check)
    else:
        if isinstance(value_to_add, CToAddContainer):
            list_to_extend.append(_create_comp_num_elem(
                value_to_add.always, value_to_add.equal,
                value_to_add.data[0] if isinstance(value_to_add.data, tuple)
                else value_to_add.data
            ))
        else:
            list_to_extend.append(value_to_add)


def set_extend_list(val_list: List[Union[_TCheck, Tuple[_TCheck, ...]]],
                    list_to_extend: List, /) -> None:
    check_tuple = False
    run_check = 0
    if not val_list:
        raise KnownExtendError("Empty list received!")
    for l_el in val_list:
        if isinstance(l_el, tuple):
            if not (run_check or check_tuple):
                raise KnownExtendError("Mismatched types found!")
            check_tuple = True
        run_check += 1
    if check_tuple:
        check_length = None
        for l_el in val_list:
            if check_length is None:
                check_length = len(l_el)
            if check_length != len(l_el):
                raise KnownExtendError("Mismatched length found!")
        list_to_extend.extend(l_el for l_tuple in val_list for l_el in l_tuple)
    else:
        list_to_extend.extend(l_el for l_el in val_list)
