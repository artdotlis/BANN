# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import final

import torch
import torch.nn.functional as nn_fun

from bann.b_frameworks.pytorch.p_activation_fun.p_activation_fun_enum import PActId
from bann.b_frameworks.pytorch.interfaces.activation_function_interface import AFClassInterface, \
    AfcDataTC


@final
class LReLu(AFClassInterface):
    @staticmethod
    def act(input_val: AfcDataTC, /) -> torch.Tensor:
        return nn_fun.leaky_relu(input_val.data)

    @staticmethod
    def activation_fun_name() -> str:
        return PActId.LRELU.value
