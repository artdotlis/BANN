# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from enum import Enum
from typing import final


@final
class PActId(Enum):
    LOGSOFTMAX = 'LogSoftMax'
    NONE = 'NoLayer'
    SOFTMAX = 'SoftMax'
    RELU = 'ReLu'
    ELU = 'ELu'
    LRELU = 'LReLu'
    SILU = 'SiLu'
    TANH = 'Tanh'
    SIGMOID = 'Sigmoid'
