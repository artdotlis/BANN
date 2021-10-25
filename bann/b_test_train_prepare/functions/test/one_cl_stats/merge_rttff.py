# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import Tuple, List, Dict

from bann.b_test_train_prepare.container.test.rttff_c import merge_ttff_fun, RtTfF, \
    check_ttff_merged
from bann.b_test_train_prepare.pytorch.p_test.functions.steps import C_STEP_SIZE, calc_step_f


def calc_one_class_stats(class_num: int, step_cnt: int,
                         data: List[Tuple[Dict[int, RtTfF], ...]], /) \
        -> Tuple[Tuple[List[int], ...], Tuple[Dict[int, RtTfF], ...]]:
    step_f = calc_step_f(step_cnt)
    classes_list = tuple(
        {num * step_f: RtTfF() for num in range(int(C_STEP_SIZE / step_f) + 1)}
        for _ in range(class_num)
    )
    rev_list = tuple(
        [step_f * num for num in range(int(C_STEP_SIZE / step_f) + 1)]
        for _ in range(class_num)
    )
    check_sum: List[Dict[int, int]] = [{} for _ in range(class_num)]
    for data_el in data:
        for index in range(class_num):
            for key, value in data_el[index].items():
                check_sum[index][key] = check_sum[index].get(key, 0) + \
                                        merge_ttff_fun(classes_list[index][key], value)
    check_ttff_merged(check_sum)
    return rev_list, classes_list
