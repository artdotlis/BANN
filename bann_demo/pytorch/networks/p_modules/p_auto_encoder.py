# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import math
from typing import Tuple, Any, final

import torch
from torch import nn
from torch.nn import functional as nn_fun

from bann.b_frameworks.pytorch.interfaces.ae_interface import AutoEncoderInterface

from bann_demo.pytorch.networks.errors.custom_erors import KnownAutoEncoderError


@final
class PAutoEncoderHG(AutoEncoderInterface):

    def __init__(self, input_cnt: int, layers_cnt: int) -> None:
        super().__init__()
        end_layer = 1
        out_num = input_cnt
        for _ in range(layers_cnt - 1):
            out_num = int(out_num / 2)
            if out_num < 2:
                break
            end_layer += 1
        self.__fc1: Tuple[nn.Linear, ...] = tuple(
            nn.Linear(
                math.ceil(input_cnt / math.pow(2, ind)),
                math.ceil(input_cnt / math.pow(2, ind + 1))
            )
            for ind in range(end_layer)
        )
        self.fc1: nn.ModuleList = nn.ModuleList(self.__fc1)
        self.__fc2: Tuple[nn.Linear, ...] = tuple(
            nn.Linear(
                math.ceil(input_cnt / math.pow(2, ind + 1)),
                math.ceil(input_cnt / math.pow(2, ind))
            )
            for ind in reversed(range(end_layer))
        )
        self.fc2 = nn.ModuleList(self.__fc2)

    def forward(self, *input_args: Any) -> torch.Tensor:
        if not (input_args and isinstance(input_args[0], torch.Tensor)):
            raise KnownAutoEncoderError(
                "Input data should not be empty and only contain tensor types"
            )
        input_data = input_args[0]
        fc_1_l: nn.ModuleList = self.fc1
        for l_encode in fc_1_l.children():
            input_data = nn_fun.relu(l_encode(input_data))
        for l_decode_ind in range(len(self.__fc2) - 1):
            input_data = nn_fun.relu(self.__fc2[l_decode_ind](input_data))
        input_data = self.__fc2[-1](input_data)
        return input_data

    def encode(self, *input_args: torch.Tensor) -> torch.Tensor:
        if not input_args:
            raise KnownAutoEncoderError("Input data should not be empty")
        input_data = input_args[0]
        with torch.no_grad():
            fc_1_l: nn.ModuleList = self.fc1
            for l_encode in fc_1_l.children():
                input_data = nn_fun.relu(l_encode(input_data))
        return input_data

    def decode(self, *input_args: torch.Tensor) -> torch.Tensor:
        if not input_args:
            raise KnownAutoEncoderError("Input data should not be empty")
        input_data = input_args[0]
        with torch.no_grad():
            for l_decode_ind in range(len(self.__fc2) - 1):
                input_data = nn_fun.relu(self.__fc2[l_decode_ind](input_data))
            input_data = self.__fc2[-1](input_data)
        return input_data

    @property
    def encoder(self) -> Tuple[Tuple[nn.Linear, ...], Tuple[nn.Linear, ...]]:
        return self.__fc1, ()

    @property
    def decoder(self) -> Tuple[Tuple[nn.Linear, ...], Tuple[nn.Linear, ...]]:
        return self.__fc2, ()
