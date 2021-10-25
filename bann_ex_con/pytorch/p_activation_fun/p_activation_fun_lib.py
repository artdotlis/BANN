# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import Tuple, Dict, Final

from bann.b_frameworks.pytorch.interfaces.activation_function_interface import AFClassInterface
from bann.b_container.errors.custom_erors import KnownActivationFunctionError
from bann_ex_con.pytorch.p_activation_fun.p_activation_fun_enum import ExPActId

_ActLib: Final[Dict[str, AFClassInterface]] = {
    # TODO fill
}


def ex_p_init_act_fun(act_fun_id: str, /) -> AFClassInterface:
    act_fun = _ActLib.get(act_fun_id, None)
    if act_fun is None:
        raise KnownActivationFunctionError(
            f"Could not find the activation function with the id {act_fun_id}!"
        )
    return act_fun


def ex_pytorch_act_names() -> Tuple[str, ...]:
    return tuple(elem.value for elem in ExPActId.__members__.values())
