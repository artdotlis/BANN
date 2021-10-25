# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import final

import torch

from bann.b_frameworks.pytorch.p_activation_fun.p_activation_fun_enum import PActId
from bann.b_frameworks.pytorch.interfaces.activation_function_interface import AFClassInterface, \
    AfcDataTC


@final
class Sigmoid(AFClassInterface):
    @staticmethod
    def act(input_val: AfcDataTC, /) -> torch.Tensor:
        return torch.sigmoid(input_val.data)

    @staticmethod
    def activation_fun_name() -> str:
        return PActId.SIGMOID.value
