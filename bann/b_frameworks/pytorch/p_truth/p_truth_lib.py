# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import Dict, Final

from bann.b_frameworks.pytorch.p_truth.implementation.matrix_distance import TruthMSimClass
from bann.b_frameworks.pytorch.p_truth.implementation.one_class import TruthOneClass
from bann.b_frameworks.pytorch.interfaces.truth_interface import TruthClassInterface
from bann.b_container.errors.custom_erors import KnownTruthFunError

from bann_ex_con.pytorch.p_truth.p_truth_lib import ex_p_check_truth_fun_id, ex_p_init_truth_fun

_TruthFunLib: Final[Dict[str, TruthClassInterface]] = {
    TruthOneClass.truth_name(): TruthOneClass(),
    TruthMSimClass.truth_name(): TruthMSimClass()
}


def p_init_truth_fun(truth_fun_id: str, /) -> TruthClassInterface:
    truth_fun = _TruthFunLib.get(truth_fun_id, None)
    if truth_fun is None:
        truth_fun = ex_p_init_truth_fun(truth_fun_id)
    return truth_fun


def p_check_truth_fun_id(truth_fun_id: str, /) -> bool:
    local_t = _TruthFunLib.get(truth_fun_id, None)
    ex_t = ex_p_check_truth_fun_id(truth_fun_id)
    local_results = local_t is not None
    if ex_t and local_results:
        raise KnownTruthFunError(f"Found duplicate {truth_fun_id}")
    return local_results or ex_t
