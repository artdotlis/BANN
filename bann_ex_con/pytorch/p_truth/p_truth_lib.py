# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import Dict, Final

from bann.b_frameworks.pytorch.interfaces.truth_interface import TruthClassInterface
from bann.b_container.errors.custom_erors import KnownTruthFunError


_TruthFunLib: Final[Dict[str, TruthClassInterface]] = {
    # TODO: fill
}


def ex_p_init_truth_fun(truth_fun_id: str, /) -> TruthClassInterface:
    truth_fun = _TruthFunLib.get(truth_fun_id, None)
    if truth_fun is None:
        raise KnownTruthFunError(
            f"Could not find the truth function with the id {truth_fun_id}!"
        )
    return truth_fun


def ex_p_check_truth_fun_id(truth_fun_id: str, /) -> bool:
    if _TruthFunLib.get(truth_fun_id, None) is None:
        return False
    return True
