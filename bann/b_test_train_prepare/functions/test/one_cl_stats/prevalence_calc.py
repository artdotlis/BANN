# -*- coding: utf-8 -*-
"""
.. moduleauthor:: Artur Lissin
.. moduleauthor:: Fabian Tann
"""
from typing import Tuple, List, Dict

from bann.b_test_train_prepare.container.test.rttff_c import RtTfF
from bann.b_test_train_prepare.functions.test.one_cl_stats.merge_rttff import calc_one_class_stats


def _prevalence_calc(tp_c: int, tn_c: int, fp_c: int, fn_c: int, /) -> float:
    try:
        res = (tp_c + fn_c) / (tp_c + tn_c + fp_c + fn_c)
    except ZeroDivisionError as ex_z:
        print(ex_z)
        res = -1
    return res


def merge_one_class_prevalence(data: List[Tuple[Dict[int, RtTfF], ...]], class_num: int,
                               step_cnt: int, /) -> Tuple[str, ...]:
    rev_list, classes_list = calc_one_class_stats(class_num, step_cnt, data)
    res = [
        (str(cl_id), str(_prevalence_calc(
            cl_el[rev_list[cl_id][1]].r_tp,
            cl_el[rev_list[cl_id][1]].r_tn,
            cl_el[rev_list[cl_id][1]].r_fp,
            cl_el[rev_list[cl_id][1]].r_fn
        )))
        for cl_id, cl_el in enumerate(classes_list)
    ]
    cross_tab = [
        "\"OneClass_Prevalence\": {",
        "\"ClassID\": [" + ','.join(re_t[0] for re_t in res) + "],",
        "\"Prevalence\": [" + ','.join(re_t[1] for re_t in res) + "]"
        "}"
    ]
    return tuple(cross_tab)
