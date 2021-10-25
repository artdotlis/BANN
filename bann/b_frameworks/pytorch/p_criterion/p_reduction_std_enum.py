# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from enum import Enum
from typing import Tuple, final


@final
class LossReduction(Enum):
    MEAN = 'mean'
    NONE = 'none'
    SUM = 'sum'


def get_loss_st_reduction_lib() -> Tuple[str, ...]:
    return tuple(elem.value for elem in LossReduction.__members__.values())
