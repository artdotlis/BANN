# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Tuple, final


@final
@dataclass
class FlatTupleT:
    list_str: Tuple[str, ...]
    list_int: Tuple[int, ...]
    flat: Tuple[str, ...]
    flat_cnt: Tuple[int, ...]
    sum_el: int
