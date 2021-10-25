# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Dict, final, Final, Callable

from bann.b_frameworks.pytorch.p_truth.p_truth_lib import p_check_truth_fun_id, p_init_truth_fun
from bann.b_frameworks.pytorch.interfaces.truth_interface import TruthClassInterface
from bann.b_container.errors.custom_erors import KnownTruthFunError
from bann.b_pan_integration.framwork_key_lib import FrameworkKeyLib


_TruthFunT: Final = Callable[[str], TruthClassInterface]
_TruthFunCheckT: Final = Callable[[str], bool]

# --------------------------------------------------------------------------------------------------


@final
@dataclass
class _FrameWorkTruthContainer:
    truth_fun: _TruthFunT
    truth_fun_check: _TruthFunCheckT


_FrameWorkTruthLib: Final[Dict[str, _FrameWorkTruthContainer]] = {
    FrameworkKeyLib.PYTORCH.value: _FrameWorkTruthContainer(
        truth_fun=p_init_truth_fun,
        truth_fun_check=p_check_truth_fun_id
    )
}


def get_framework_truth_lib(framework: str, /) -> _FrameWorkTruthContainer:
    framework_fun = _FrameWorkTruthLib.get(framework, None)
    if framework_fun is None:
        raise KnownTruthFunError(f"The framework {framework} is not defined!")
    return framework_fun
