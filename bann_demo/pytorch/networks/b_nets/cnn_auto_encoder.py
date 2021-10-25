# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Any, final

import torch
from torch import nn
from torch.nn import functional as nn_fun

from bann.b_frameworks.pytorch.net_model_interface import CurrentNetData, InitContainer
from bann.b_frameworks.pytorch.pytorch_lego_const import LegoContInit

from bann_demo.pytorch.networks.libs.ae_lib import CNNAEParams, get_ae_from_lib, AELib
from bann_demo.pytorch.networks.p_modules.p_conv_auto_encoder import PConvAutoEncoderHG
from bann_demo.pytorch.networks.errors.custom_erors import KnownAutoEncoderError


@final
@dataclass
class CNNAutoEncoderLCI(InitContainer, LegoContInit, CNNAEParams):
    pass


@final
class BCNNAutoEncoderNetCon(nn.Module):
    def __init__(self, cnn_ae: PConvAutoEncoderHG, /) -> None:
        super().__init__()
        encoder = cnn_ae.encoder
        self.fc1: nn.ModuleList = nn.ModuleList(encoder[0])
        self.fc2: nn.ModuleList = nn.ModuleList(encoder[1])

    def forward(self, *input_args: Any) -> torch.Tensor:
        if not (input_args and isinstance(input_args[0], torch.Tensor)):
            raise KnownAutoEncoderError("Input data should not be empty or not a tensor")
        input_data = input_args[0]
        fc_1_l: nn.ModuleList = self.fc1
        fc_2_l: nn.ModuleList = self.fc2
        for l_index, l_encode in enumerate(fc_1_l.children()):
            input_data = nn_fun.relu(l_encode(input_data))
            mod_loc: nn.Module = fc_2_l[l_index]
            input_data = mod_loc(input_data)
        return input_data


def create_cnn_auto_encoder_net(args_con: LegoContInit, /) -> CurrentNetData:
    if not isinstance(args_con, CNNAutoEncoderLCI):
        raise KnownAutoEncoderError(
            f"Expected {CNNAutoEncoderLCI.__name__} got {type(args_con).__name__}"
        )
    p_auto_encoder = get_ae_from_lib(AELib.CNNAE.value, args_con)
    if not isinstance(p_auto_encoder, PConvAutoEncoderHG):
        raise KnownAutoEncoderError(
            f"Expected {PConvAutoEncoderHG.__name__} got {type(p_auto_encoder).__name__}"
        )
    return CurrentNetData(
        fitness=float('inf'),
        com=p_auto_encoder,
        lego=BCNNAutoEncoderNetCon(p_auto_encoder)
    )
