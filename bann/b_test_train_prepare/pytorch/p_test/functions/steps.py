# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import Final

C_STEP_SIZE: Final[int] = 10000
C_STEP_SIZE_F: Final[float] = C_STEP_SIZE * 1.0


def calc_step_f(step_cnt: int, /) -> int:
    step_f: int = int(C_STEP_SIZE / step_cnt)
    if step_f <= 0:
        step_f = 1
    return step_f
