# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import re
from enum import Enum
from typing import final, List, Dict, Tuple, Callable, Final, Pattern

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
from bann.b_test_train_prepare.pytorch.p_test.functions.pr_fun import merge_one_class_pr
from bann.b_test_train_prepare.pytorch.p_test.functions.roc_fun import merge_one_class_roc
from bann.b_test_train_prepare.pytorch.p_test.functions.srcc import merge_srcc, TargetOutput
from bann.b_test_train_prepare.pytorch.p_test.functions.tp_fp_tn_fn import regression_ttff
from bann.b_test_train_prepare.pytorch.p_test.libs.container import RegOutTarDevC


@final
class RegTestsLib(Enum):
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
class RegTestsREGEXLib(Enum):
    FSCORE = (re.compile(r'F\((.+)\)'), "F-Score", "F(.+)")


def get_reg_test_list() -> List[str]:
    return [c_t.value for c_t in RegTestsLib.__members__.values()]


def get_reg_test_pattern_list() -> List[Pattern[str]]:
    return [c_t.value[0] for c_t in RegTestsREGEXLib.__members__.values()]


def get_reg_test_list_comb() -> List[str]:
    return [c_t.value[2] for c_t in RegTestsREGEXLib.__members__.values()]


_TEST_REG_REGEX_DICT: Final[Dict[
    str, Callable[[str, int, int, Dict[str, List[Tuple]], List[str]], Tuple[str, ...]]
]] = {
    RegTestsREGEXLib.FSCORE.value[1]:
        lambda test_n, class_num, step_cnt, results, extra_args:
        merge_one_class_f_score(results[test_n], class_num, 2, extra_args)
}
_TEST_REG_DICT: Final[Dict[
    str, Callable[[str, int, int, Dict[str, List[Tuple]]], Tuple[str, ...]]
]] = {
    RegTestsLib.ROC.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_roc(results[test_n], class_num, step_cnt),
    RegTestsLib.PR.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_pr(results[test_n], class_num, step_cnt),
    RegTestsLib.CROSS.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_cross_tab(results[test_n], class_num, step_cnt),
    RegTestsLib.SRCC.value:
        lambda test_n, class_num, step_cnt, results:
            merge_srcc(results[test_n]),
    RegTestsLib.MCC.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_mcc(results[test_n], class_num, 2),
    RegTestsLib.PPREV.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_prevalence(results[test_n], class_num, 2),
    RegTestsLib.SENS.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_sensitivity(results[test_n], class_num, 2),
    RegTestsLib.FALL.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_fallout(results[test_n], class_num, 2),
    RegTestsLib.FNR.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_fnr(results[test_n], class_num, 2),
    RegTestsLib.FPR.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_fpr(results[test_n], class_num, 2),
    RegTestsLib.TNR.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_tnr(results[test_n], class_num, 2),
    RegTestsLib.ACC.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_acc(results[test_n], class_num, 2),
    RegTestsLib.PPV.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_ppv(results[test_n], class_num, 2),
    RegTestsLib.FDR.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_fdr(results[test_n], class_num, 2),
    RegTestsLib.FOMR.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_fomr(results[test_n], class_num, 2),
    RegTestsLib.NPV.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_npv(results[test_n], class_num, 2),
    RegTestsLib.LRP.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_lrp(results[test_n], class_num, 2),
    RegTestsLib.LRM.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_lrm(results[test_n], class_num, 2),
    RegTestsLib.DOR.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_dor(results[test_n], class_num, 2),
    RegTestsLib.PREC.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_precision(results[test_n], class_num, 2),
    RegTestsLib.REC.value:
        lambda test_n, class_num, step_cnt, results:
            merge_one_class_recall(results[test_n], class_num, 2)
}


def _get_reg_stats_50_50() -> Tuple[str, ...]:
    return (
        RegTestsLib.MCC.value, RegTestsLib.PPREV.value, RegTestsLib.SENS.value,
        RegTestsLib.FALL.value, RegTestsLib.FNR.value, RegTestsLib.FPR.value,
        RegTestsLib.TNR.value, RegTestsLib.ACC.value, RegTestsLib.PPV.value,
        RegTestsLib.FDR.value, RegTestsLib.FOMR.value, RegTestsLib.NPV.value,
        RegTestsLib.LRP.value, RegTestsLib.LRM.value, RegTestsLib.DOR.value,
        RegTestsLib.PREC.value, RegTestsLib.REC.value
    )


def _get_reg_stats_50_50_regex() -> Tuple[Tuple[Pattern[str], str], ...]:
    return (
        (RegTestsREGEXLib.FSCORE.value[0], RegTestsREGEXLib.FSCORE.value[1])
    )


def reg_run_test(test_n: str, class_num: int, step_cnt: int,
                 results: Dict[str, List[Tuple]], /) -> Tuple[str, ...]:
    res = _TEST_REG_DICT.get(test_n, None)
    if res is None:
        res_buf = _get_reg_stats_50_50_regex()
        res_extra = get_pattern_match(test_n, [ele[0] for ele in res_buf])
        if res_extra is not None:
            res = _TEST_REG_REGEX_DICT.get(res_buf[res_extra[1]], None)
        if res is None:
            raise KnownTesterError(f"Could not find test with the name {test_n}!")
        return res(test_n, class_num, step_cnt, results, res_extra[0])
    return res(test_n, class_num, step_cnt, results)


def reg_calc_stats(test_n: str, class_num: int, step_cnt: int, bin_cut: Tuple[float, ...],
                   out_tar: RegOutTarDevC, /) -> Tuple:
    if bin_cut and test_n in (
            RegTestsLib.ROC.value, RegTestsLib.PR.value, RegTestsLib.CROSS.value
    ):
        return regression_ttff(out_tar, class_num, step_cnt, bin_cut)
    elif test_n == RegTestsLib.SRCC.value:
        return TargetOutput(output=out_tar.output, target=out_tar.target),
    elif test_n in _get_reg_stats_50_50() or \
            check_pattern(test_n, [ele[0] for ele in _get_reg_stats_50_50_regex()]):
        return regression_ttff(out_tar, class_num, 2, bin_cut)
    else:
        raise KnownTesterError(f"Could not find test with the name {test_n}!")
