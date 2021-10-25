# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import Tuple, Dict, Final

from bann.b_frameworks.pytorch.p_activation_fun.p_activation_fun_enum import PActId
from bann.b_frameworks.pytorch.p_activation_fun.implementation.elu import ELu
from bann.b_frameworks.pytorch.p_activation_fun.implementation.log_soft_max import LogSoftMax
from bann.b_frameworks.pytorch.p_activation_fun.implementation.lrelu import LReLu
from bann.b_frameworks.pytorch.p_activation_fun.implementation.no_layer import NoLayer
from bann.b_frameworks.pytorch.p_activation_fun.implementation.relu import ReLu
from bann.b_frameworks.pytorch.p_activation_fun.implementation.soft_max import SoftMax
from bann.b_frameworks.pytorch.interfaces.activation_function_interface import AFClassInterface
from bann.b_frameworks.pytorch.p_activation_fun.implementation.sigmoid import Sigmoid
from bann.b_frameworks.pytorch.p_activation_fun.implementation.tanh import Tanh
from bann.b_frameworks.pytorch.p_activation_fun.implementation.silu import SiLu
from bann.b_container.errors.custom_erors import KnownActivationFunctionError

from bann_ex_con.pytorch.p_activation_fun.p_activation_fun_lib import ex_pytorch_act_names, \
    ex_p_init_act_fun

_ActLib: Final[Dict[str, AFClassInterface]] = {
    NoLayer.activation_fun_name(): NoLayer(),
    SoftMax.activation_fun_name(): SoftMax(),
    LogSoftMax.activation_fun_name(): LogSoftMax(),
    ELu.activation_fun_name(): ELu(),
    ReLu.activation_fun_name(): ReLu(),
    LReLu.activation_fun_name(): LReLu(),
    Sigmoid.activation_fun_name(): Sigmoid(),
    Tanh.activation_fun_name(): Tanh(),
    SiLu.activation_fun_name(): SiLu()
}


def p_init_act_fun(act_fun_id: str, /) -> AFClassInterface:
    act_fun = _ActLib.get(act_fun_id, None)
    if act_fun is None:
        act_fun = ex_p_init_act_fun(act_fun_id)
    return act_fun


def pytorch_act_names() -> Tuple[str, ...]:
    local_names = tuple(elem.value for elem in PActId.__members__.values())
    ex_names = ex_pytorch_act_names()
    for elem_e in ex_names:
        if elem_e in local_names:
            raise KnownActivationFunctionError(f"Duplicate found: {elem_e}")
    return *local_names, *ex_names
