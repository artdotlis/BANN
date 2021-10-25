# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import final, Optional, Dict, List


@final
@dataclass
class RtTfF:
    r_tp: int = 0
    r_tn: int = 0
    r_fp: int = 0
    r_fn: int = 0


def merge_ttff_fun(container: RtTfF, target: RtTfF, /) -> int:
    sum_added: int = 0
    container.r_fn += target.r_fn
    sum_added += target.r_fn
    container.r_fp += target.r_fp
    sum_added += target.r_fp
    container.r_tn += target.r_tn
    sum_added += target.r_tn
    container.r_tp += target.r_tp
    sum_added += target.r_tp
    return sum_added


def check_ttff_merged(merged_sum: List[Dict[int, int]], /) -> None:
    check_sum: Optional[int] = None
    for end_cl in merged_sum:
        for end_sum in end_cl.values():
            if check_sum is None:
                check_sum = end_sum
            if check_sum != end_sum:
                print(f"\nWARNING!! Inconsistent TTFF-data {check_sum} != {end_sum}\n\n")
