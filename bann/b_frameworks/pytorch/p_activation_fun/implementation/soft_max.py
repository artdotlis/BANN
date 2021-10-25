# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import final

import torch
import torch.nn.functional as nn_fun

from bann.b_frameworks.pytorch.p_activation_fun.p_activation_fun_enum import PActId
from bann.b_frameworks.pytorch.interfaces.activation_function_interface import AFClassInterface, \
    AfcDataTC
from bann.b_container.errors.custom_erors import KnownActivationFunctionError


@final
class SoftMax(AFClassInterface):
    @staticmethod
    def act(input_val: AfcDataTC, /) -> torch.Tensor:
        if input_val.dim is None:
            raise KnownActivationFunctionError("Dim was not set!")
        return nn_fun.softmax(input_val.data, dim=input_val.dim)

    @staticmethod
    def activation_fun_name() -> str:
        return PActId.SOFTMAX.value
