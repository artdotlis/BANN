# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import List, Tuple, Dict

from bann.b_test_train_prepare.container.test.rttff_c import merge_ttff_fun, RtTfF, \
    check_ttff_merged
from bann.b_test_train_prepare.pytorch.p_test.functions.steps import C_STEP_SIZE, calc_step_f


def _one_class_cross_tab_str_erg(data: Dict[int, RtTfF], class_id: int, /) \
        -> List[Tuple[str, str, str, str, str, str]]:
    return [
        (
            str(class_id), str(tf_v.r_tp), str(tf_v.r_fp),
            str(tf_v.r_fn), str(tf_v.r_tn), str(th_v / 100.)
        )
        for th_v, tf_v in data.items()
    ]


def merge_one_class_cross_tab(data: List[Tuple[Dict[int, RtTfF], ...]], class_num: int,
                              step_cnt: int, /) -> Tuple[str, ...]:
    step_f = calc_step_f(step_cnt)
    classes_list = tuple(
        {num * step_f: RtTfF() for num in range(int(C_STEP_SIZE / step_f) + 1)}
        for _ in range(class_num)
    )
    check_sum: List[Dict[int, int]] = [{} for _ in range(class_num)]
    for data_el in data:
        for index in range(class_num):
            for key, value in data_el[index].items():
                check_sum[index][key] = check_sum[index].get(key, 0) + \
                                        merge_ttff_fun(classes_list[index][key], value)
    check_ttff_merged(check_sum)
    res = [
        _one_class_cross_tab_str_erg(cl_el, cl_id) for cl_id, cl_el in enumerate(classes_list)
    ]
    cross_tab = [
        "\"OneClass_CrossTab\": {",
        "\"ClassID\": [" + ','.join(cid[0] for re_t in res for cid in re_t) + "],",
        "\"TP\": [" + ','.join(cid[1] for re_t in res for cid in re_t) + "],",
        "\"FP\": [" + ','.join(cid[2] for re_t in res for cid in re_t) + "],",
        "\"FN\": [" + ','.join(cid[3] for re_t in res for cid in re_t) + "],",
        "\"TN\": [" + ','.join(cid[4] for re_t in res for cid in re_t) + "],",
        "\"Threshold_in_%\": [" + ','.join(cid[5] for re_t in res for cid in re_t) + "]",
        "}"
    ]
    return tuple(cross_tab)
