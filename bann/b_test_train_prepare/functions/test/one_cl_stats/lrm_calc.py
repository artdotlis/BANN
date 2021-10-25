# -*- coding: utf-8 -*-
"""
.. moduleauthor:: Artur Lissin
.. moduleauthor:: Fabian Tann
"""
from typing import Tuple, List, Dict

from bann.b_test_train_prepare.functions.test.one_cl_stats.fnr_calc import fnr_calc
from bann.b_test_train_prepare.container.test.rttff_c import RtTfF
from bann.b_test_train_prepare.functions.test.one_cl_stats.merge_rttff import calc_one_class_stats
from bann.b_test_train_prepare.functions.test.one_cl_stats.tnr_calc import tnr_calc


def lrm_calc(fn_c: int, tp_c: int, fp_c: int, tn_c: int, /) -> float:
    try:
        res = fnr_calc(fn_c, tp_c) / tnr_calc(fp_c, tn_c)
    except ZeroDivisionError as ex_z:
        print(ex_z)
        res = -1
    return res


def merge_one_class_lrm(data: List[Tuple[Dict[int, RtTfF], ...]], class_num: int,
                        step_cnt: int, /) -> Tuple[str, ...]:
    rev_list, classes_list = calc_one_class_stats(class_num, step_cnt, data)
    res = [
        (str(cl_id), str(lrm_calc(
            cl_el[rev_list[cl_id][1]].r_fn,
            cl_el[rev_list[cl_id][1]].r_tp,
            cl_el[rev_list[cl_id][1]].r_fp,
            cl_el[rev_list[cl_id][1]].r_tn,
        )))
        for cl_id, cl_el in enumerate(classes_list)
    ]
    cross_tab = [
        "\"OneClass_LRM\": {",
        "\"ClassID\": [" + ','.join(re_t[0] for re_t in res) + "],",
        "\"LRM\": [" + ','.join(re_t[1] for re_t in res) + "]"
        "}"
    ]
    return tuple(cross_tab)
