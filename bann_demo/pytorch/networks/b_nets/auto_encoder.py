# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Any, final

import torch
from torch import nn
from torch.nn import functional as nn_fun

from bann.b_frameworks.pytorch.net_model_interface import CurrentNetData, InitContainer
from bann.b_frameworks.pytorch.pytorch_lego_const import LegoContInit

from bann_demo.pytorch.networks.p_modules.p_auto_encoder import PAutoEncoderHG
from bann_demo.pytorch.networks.libs.ae_lib import get_ae_from_lib, AELib, AEParams
from bann_demo.pytorch.networks.errors.custom_erors import KnownAutoEncoderError


@final
class BAutoEncoderNetCon(nn.Module):

    def __init__(self, ae_model: PAutoEncoderHG, /) -> None:
        super().__init__()
        self.fc1: nn.ModuleList = nn.ModuleList(ae_model.encoder[0])

    def forward(self, *input_args: Any) -> torch.Tensor:
        if not (input_args and isinstance(input_args[0], torch.Tensor)):
            raise KnownAutoEncoderError("Input data should not be empty or not a tensor")
        input_data = input_args[0]
        fc_l: nn.ModuleList = self.fc1
        enc_list = list(fc_l.children())
        for encode_l in enc_list[0:-1]:
            input_data = nn_fun.relu(encode_l(input_data))
        return enc_list[-1](input_data)


@final
@dataclass
class AutoEncoderLCI(InitContainer, LegoContInit, AEParams):
    pass


def create_auto_encoder_net(args_con: LegoContInit, /) -> CurrentNetData:
    if not isinstance(args_con, AutoEncoderLCI):
        raise KnownAutoEncoderError(
            f"Expected {AutoEncoderLCI.__name__} got {type(args_con).__name__}"
        )
    p_auto_encoder = get_ae_from_lib(AELib.AE.value, args_con)
    if not isinstance(p_auto_encoder, PAutoEncoderHG):
        raise KnownAutoEncoderError(
            f"Expected {PAutoEncoderHG.__name__} got {type(p_auto_encoder).__name__}"
        )
    return CurrentNetData(
        fitness=float('inf'),
        com=p_auto_encoder,
        lego=BAutoEncoderNetCon(p_auto_encoder)
    )
