# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Dict, Tuple, final, Final, Callable

from bann.b_frameworks.pytorch.p_activation_fun.p_activation_fun_lib import pytorch_act_names, \
    p_init_act_fun
from bann.b_frameworks.pytorch.interfaces.activation_function_interface import AFClassInterface
from bann.b_container.errors.custom_erors import KnownActivationFunctionError
from bann.b_pan_integration.framwork_key_lib import FrameworkKeyLib


_ActBFun: Final = Callable[[str], AFClassInterface]
_ActBFunNames: Final = Callable[[], Tuple[str, ...]]

# --------------------------------------------------------------------------------------------------


@final
@dataclass
class FrameWorkLLContainer:
    act_b: _ActBFun
    act_names_b: _ActBFunNames


_FrameWorkLLLib: Final[Dict[str, FrameWorkLLContainer]] = {
    FrameworkKeyLib.PYTORCH.value: FrameWorkLLContainer(
        act_b=p_init_act_fun,
        act_names_b=pytorch_act_names
    )
}


def get_framework_act_lib(framework: str, /) -> FrameWorkLLContainer:
    framework_fun = _FrameWorkLLLib.get(framework, None)
    if framework_fun is None:
        raise KnownActivationFunctionError(f"The framework {framework} is not defined!")
    return framework_fun
