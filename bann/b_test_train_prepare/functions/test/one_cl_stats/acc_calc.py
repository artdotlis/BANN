# -*- coding: utf-8 -*-
"""
.. moduleauthor:: Artur Lissin
.. moduleauthor:: Fabian Tann
"""
from typing import Tuple, List, Dict


from bann.b_test_train_prepare.container.test.rttff_c import RtTfF
from bann.b_test_train_prepare.functions.test.one_cl_stats.merge_rttff import calc_one_class_stats


def _acc_calc(fp_c: int, tn_c: int, tp_c: int, fn_c: int, /) -> float:
    try:
        res = (tp_c + tn_c) / (tp_c + tn_c + fp_c + fn_c)
    except ZeroDivisionError as ex_z:
        print(ex_z)
        res = -1
    return res


def merge_one_class_acc(data: List[Tuple[Dict[int, RtTfF], ...]], class_num: int,
                        step_cnt: int, /) -> Tuple[str, ...]:
    rev_list, classes_list = calc_one_class_stats(class_num, step_cnt, data)
    res = [
        (str(cl_id), str(_acc_calc(
            cl_el[rev_list[cl_id][1]].r_fp,
            cl_el[rev_list[cl_id][1]].r_tn,
            cl_el[rev_list[cl_id][1]].r_tp,
            cl_el[rev_list[cl_id][1]].r_fn
        )))
        for cl_id, cl_el in enumerate(classes_list)
    ]
    cross_tab = [
        "\"OneClass_ACC\": {",
        "\"ClassID\": [" + ','.join(re_t[0] for re_t in res) + "],",
        "\"ACC\": [" + ','.join(re_t[1] for re_t in res) + "]"
        "}"
    ]
    return tuple(cross_tab)
