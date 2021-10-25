# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import List, Tuple, Dict

from bann.b_test_train_prepare.pytorch.p_test.functions.auc_calc import calc_auc
from bann.b_test_train_prepare.container.test.rttff_c import merge_ttff_fun, RtTfF, \
    check_ttff_merged
from bann.b_test_train_prepare.pytorch.p_test.functions.steps import C_STEP_SIZE, calc_step_f


def _create_fill_output_line(to_fill: Dict[str, List[float]],
                             th_v: int, tf_v: RtTfF,
                             class_id: int, /) -> Tuple[str, str, str, str]:
    prec_div = (tf_v.r_tp + tf_v.r_fp)
    if prec_div <= 0:
        prec = 0.0
    else:
        prec = (1.0 * tf_v.r_tp) / prec_div
    recall_div = (tf_v.r_tp + tf_v.r_fn)
    if recall_div <= 0:
        recall = 0.0
    else:
        recall = (1.0 * tf_v.r_tp) / recall_div
    to_fill.setdefault(f"{recall}", []).append(prec)
    return str(class_id), str(prec), str(recall), str(th_v / 100.)


def _one_class_pr_str_erg(data: Dict[int, RtTfF], cl_id: int, /) \
        -> Tuple[List[Tuple[str, str, str, str]], Tuple[str, str, str]]:
    dict_tpr: Dict[str, List[float]] = {}
    pr_res = [
        _create_fill_output_line(dict_tpr, th_v, tf_v, cl_id)
        for th_v, tf_v in data.items()
    ]
    trapz_auc, step_auc = calc_auc(dict_tpr, True)
    return pr_res, (str(cl_id), str(trapz_auc), str(step_auc))


def merge_one_class_pr(data: List[Tuple[Dict[int, RtTfF], ...]], class_num: int,
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
    res = tuple(_one_class_pr_str_erg(cl_el, cl_id) for cl_id, cl_el in enumerate(classes_list))
    erg_list = [
        "\"OneClass_PR\": {",
        "\"ClassID\": [" + ','.join(cid[0] for re_t in res for cid in re_t[0]) + "],",
        "\"Precision\": [" + ','.join(cid[1] for re_t in res for cid in re_t[0]) + "],",
        "\"Recall\": [" + ','.join(cid[2] for re_t in res for cid in re_t[0]) + "],",
        "\"Threshold_in_%\": [" + ','.join(cid[3] for re_t in res for cid in re_t[0]) + "]",
        "},"
        "\"OneClass_PR_AUC\": {",
        "\"ClassID\": [" + ','.join(res_t[1][0] for res_t in res) + "],",
        "\"Trapz_AUC\": [" + ','.join(res_t[1][1] for res_t in res) + "],",
        "\"AP_AUC\": [" + ','.join(res_t[1][2] for res_t in res) + "]",
        "}"
    ]

    return tuple(erg_list)
