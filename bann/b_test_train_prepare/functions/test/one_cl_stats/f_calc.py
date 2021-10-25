# -*- coding: utf-8 -*-
"""
.. moduleauthor:: Artur Lissin
"""
from typing import Tuple, List, Dict

from bann.b_test_train_prepare.container.test.rttff_c import RtTfF
from bann.b_test_train_prepare.functions.test.one_cl_stats.merge_rttff import calc_one_class_stats
from bann.b_test_train_prepare.functions.test.one_cl_stats.precision_calc import precision_calc
from bann.b_test_train_prepare.functions.test.one_cl_stats.recall_calc import recall_calc


def _f_calc(tp_c: int, fp_c: int, fn_c: int, beta: float, /) -> float:
    recall = recall_calc(tp_c, fn_c)
    precision = precision_calc(fp_c, tp_c)
    try:
        res = (1 + beta**2) * ((precision * recall) / ((beta**2 * precision) + recall))
    except ZeroDivisionError as ex_z:
        print(ex_z)
        res = -1
    return res


def _get_regex(extra_args: List[str], /) -> float:
    if not len(extra_args):
        return 1.0
    try:
        beta = 1.0
    except ValueError as ex_v:
        print(ex_v)
        beta = 1.0
    return beta


def merge_one_class_f_score(data: List[Tuple[Dict[int, RtTfF], ...]], class_num: int,
                            step_cnt: int, extra_args: List[str], /) -> Tuple[str, ...]:
    rev_list, classes_list = calc_one_class_stats(class_num, step_cnt, data)
    beta = _get_regex(extra_args)
    res = [
        (str(cl_id), str(_f_calc(
            cl_el[rev_list[cl_id][1]].r_tp,
            cl_el[rev_list[cl_id][1]].r_fp,
            cl_el[rev_list[cl_id][1]].r_fn,
            beta
        )))
        for cl_id, cl_el in enumerate(classes_list)
    ]
    cross_tab = [
        f"\"OneClass_F({extra_args})\": " + "{",
        "\"ClassID\": [" + ','.join(re_t[0] for re_t in res) + "],",
        f"\"F({extra_args})\": [" + ','.join(re_t[1] for re_t in res) + "]"
        "}"
    ]
    return tuple(cross_tab)
