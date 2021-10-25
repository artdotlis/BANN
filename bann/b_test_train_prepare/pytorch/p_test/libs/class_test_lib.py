# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import re
from enum import Enum
from typing import final, List, Final, Dict, Callable, Tuple, Pattern

import torch

from bann.b_container.functions.check_arg_complex import get_pattern_match, check_pattern
from bann.b_test_train_prepare.errors.custom_errors import KnownTesterError
from bann.b_test_train_prepare.functions.test.one_cl_stats.dor_calc import merge_one_class_dor
from bann.b_test_train_prepare.functions.test.one_cl_stats.f_calc import merge_one_class_f_score
from bann.b_test_train_prepare.functions.test.one_cl_stats.fdr_calc import merge_one_class_fdr
from bann.b_test_train_prepare.functions.test.one_cl_stats.fpr_calc import merge_one_class_fpr
from bann.b_test_train_prepare.functions.test.one_cl_stats.tnr_calc import merge_one_class_tnr
from bann.b_test_train_prepare.pytorch.p_test.functions.crosstab import merge_one_class_cross_tab
from bann.b_test_train_prepare.functions.test.one_cl_stats.acc_calc import \
    merge_one_class_acc
from bann.b_test_train_prepare.functions.test.one_cl_stats.fallout_calc import \
    merge_one_class_fallout
from bann.b_test_train_prepare.functions.test.one_cl_stats.fnr_calc import \
    merge_one_class_fnr
from bann.b_test_train_prepare.functions.test.one_cl_stats.fomr_calc import \
    merge_one_class_fomr
from bann.b_test_train_prepare.functions.test.one_cl_stats.lrm_calc import \
    merge_one_class_lrm
from bann.b_test_train_prepare.functions.test.one_cl_stats.lrp_calc import \
    merge_one_class_lrp
from bann.b_test_train_prepare.functions.test.one_cl_stats.mcc_calc import \
    merge_one_class_mcc
from bann.b_test_train_prepare.functions.test.one_cl_stats.npv_calc import \
    merge_one_class_npv
from bann.b_test_train_prepare.functions.test.one_cl_stats.ppv_calc import \
    merge_one_class_ppv
from bann.b_test_train_prepare.functions.test.one_cl_stats.precision_calc import \
    merge_one_class_precision
from bann.b_test_train_prepare.functions.test.one_cl_stats.prevalence_calc import \
    merge_one_class_prevalence
from bann.b_test_train_prepare.functions.test.one_cl_stats.recall_calc import \
    merge_one_class_recall
from bann.b_test_train_prepare.functions.test.one_cl_stats.sensitivity_calc import \
    merge_one_class_sensitivity
from bann.b_test_train_prepare.pytorch.p_test.functions.one_class import one_class_prediction
from bann.b_test_train_prepare.pytorch.p_test.functions.pr_fun import merge_one_class_pr
from bann.b_test_train_prepare.pytorch.p_test.functions.roc_fun import merge_one_class_roc
from bann.b_test_train_prepare.pytorch.p_test.functions.srcc import merge_srcc, TargetOutput
from bann.b_test_train_prepare.pytorch.p_test.functions.tp_fp_tn_fn import one_class_ttff
from bann.b_test_train_prepare.pytorch.p_test.libs.container import OneClassOutTar


@final
class ClassTestsLib(Enum):
    ROC = "ROC"
    PR = "PR"
    CROSS = "CROSSTAB"
    SRCC = "SRCC"
    MCC = "MCC"
    PPREV = "Prevalence"
    SENS = "Sensitivity"
    FALL = "Fallout"
    FNR = "FNR"
    FPR = "FPR"
    TNR = "TNR"
    ACC = "ACC"
    PPV = "PPV"
    FDR = "FDR"
    FOMR = "FOMR"
    NPV = "NPV"
    LRP = "LRP"
    LRM = "LRM"
    DOR = "DOR"
    PREC = "Precision"
    REC = "Recall"


@final
class ClassTestsREGEXLib(Enum):
    FSCORE = (re.compile(r'F\((.+)\)'), "F-Score", "F(.+)")


def get_class_test_list() -> List[str]:
    return [c_t.value for c_t in ClassTestsLib.__members__.values()]


def get_class_test_pattern_list() -> List[Pattern[str]]:
    return [c_t.value[0] for c_t in ClassTestsREGEXLib.__members__.values()]


def get_class_test_list_comb() -> List[str]:
    return [c_t.value[2] for c_t in ClassTestsREGEXLib.__members__.values()]


_TEST_ONE_CLASS_REGEX_DICT: Final[Dict[
    str, Callable[[str, int, int, Dict[str, List[Tuple]], List[str]], Tuple[str, ...]]
]] = {
    ClassTestsREGEXLib.FSCORE.value[1]:
        lambda test_n, class_num, step_cnt, results, extra_args:
        merge_one_class_f_score(results[test_n], class_num, 2, extra_args)
}
_TEST_ONE_CLASS_DICT: Final[Dict[
    str, Callable[[str, int, int, Dict[str, List[Tuple]]], Tuple[str, ...]]
]] = {
    ClassTestsLib.ROC.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_roc(results[test_n], class_num, step_cnt),
    ClassTestsLib.PR.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_pr(results[test_n], class_num, step_cnt),
    ClassTestsLib.CROSS.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_cross_tab(results[test_n], class_num, step_cnt),
    ClassTestsLib.SRCC.value:
        lambda test_n, class_num, step_cnt, results:
            merge_srcc(results[test_n]),
    ClassTestsLib.MCC.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_mcc(results[test_n], class_num, 2),
    ClassTestsLib.PPREV.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_prevalence(results[test_n], class_num, 2),
    ClassTestsLib.SENS.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_sensitivity(results[test_n], class_num, 2),
    ClassTestsLib.FALL.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_fallout(results[test_n], class_num, 2),
    ClassTestsLib.FNR.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_fnr(results[test_n], class_num, 2),
    ClassTestsLib.FPR.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_fpr(results[test_n], class_num, 2),
    ClassTestsLib.TNR.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_tnr(results[test_n], class_num, 2),
    ClassTestsLib.ACC.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_acc(results[test_n], class_num, 2),
    ClassTestsLib.PPV.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_ppv(results[test_n], class_num, 2),
    ClassTestsLib.FDR.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_fdr(results[test_n], class_num, 2),
    ClassTestsLib.FOMR.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_fomr(results[test_n], class_num, 2),
    ClassTestsLib.NPV.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_npv(results[test_n], class_num, 2),
    ClassTestsLib.LRP.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_lrp(results[test_n], class_num, 2),
    ClassTestsLib.LRM.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_lrm(results[test_n], class_num, 2),
    ClassTestsLib.DOR.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_dor(results[test_n], class_num, 2),
    ClassTestsLib.PREC.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_precision(results[test_n], class_num, 2),
    ClassTestsLib.REC.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_recall(results[test_n], class_num, 2)
}


def _get_one_class_stats_50_50() -> Tuple[str, ...]:
    return (
        ClassTestsLib.MCC.value, ClassTestsLib.PPREV.value, ClassTestsLib.SENS.value,
        ClassTestsLib.FALL.value, ClassTestsLib.FNR.value, ClassTestsLib.FPR.value,
        ClassTestsLib.TNR.value, ClassTestsLib.ACC.value, ClassTestsLib.PPV.value,
        ClassTestsLib.FDR.value, ClassTestsLib.FOMR.value, ClassTestsLib.NPV.value,
        ClassTestsLib.LRP.value, ClassTestsLib.LRM.value, ClassTestsLib.DOR.value,
        ClassTestsLib.PREC.value, ClassTestsLib.REC.value
    )


def _get_class_stats_50_50_regex() -> Tuple[Tuple[Pattern[str], str], ...]:
    return (
        (ClassTestsREGEXLib.FSCORE.value[0], ClassTestsREGEXLib.FSCORE.value[1])
    )


def one_class_run_test(test_n: str, class_num: int, step_cnt: int,
                       results: Dict[str, List[Tuple]], /) -> Tuple[str, ...]:
    res = _TEST_ONE_CLASS_DICT.get(test_n, None)
    if res is None:
        res_buf = _get_class_stats_50_50_regex()
        res_extra = get_pattern_match(test_n, [ele[0] for ele in res_buf])
        if res_extra is not None:
            res = _TEST_ONE_CLASS_REGEX_DICT.get(res_buf[res_extra[1]], None)
        if res is None:
            raise KnownTesterError(f"Could not find test with the name {test_n}!")
        return res(test_n, class_num, step_cnt, results, res_extra[0])
    return res(test_n, class_num, step_cnt, results)


def one_class_calc_stats(test_n: str, class_num: int, step_cnt: int, device: torch.device,
                         out_tar: OneClassOutTar, /) -> Tuple:
    if test_n in (ClassTestsLib.ROC.value, ClassTestsLib.PR.value, ClassTestsLib.CROSS.value):
        return one_class_ttff(out_tar.output, out_tar.target, device, class_num, step_cnt)
    elif test_n == ClassTestsLib.SRCC.value:
        return TargetOutput(
            output=one_class_prediction(out_tar.output_act, device),
            target=out_tar.target
        ),
    elif test_n in _get_one_class_stats_50_50() or \
            check_pattern(test_n, [ele[0] for ele in _get_class_stats_50_50_regex()]):
        return one_class_ttff(out_tar.output, out_tar.target, device, class_num, 2)
    else:
        raise KnownTesterError(f"Could not find test with the name {test_n}!")
