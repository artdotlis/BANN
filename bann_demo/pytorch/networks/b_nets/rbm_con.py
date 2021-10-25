# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Any, final

import torch
from torch import nn

from bann.b_frameworks.pytorch.interfaces.rbm_interface import RBMInterface
from bann.b_frameworks.pytorch.net_model_interface import CurrentNetData, InitContainer
from bann.b_frameworks.pytorch.pytorch_lego_const import LegoContInit

from bann_demo.pytorch.networks.errors.custom_erors import KnownRBMError
from bann_demo.pytorch.networks.libs.rbm_lib import get_rbm_from_lib, RBMParams


@final
class BRBMNetCon(nn.Module):

    def __init__(self, rbm_model: RBMInterface, /) -> None:
        super().__init__()
        self.fc1 = rbm_model.rbm()

    def forward(self, *input_args: Any) -> torch.Tensor:
        if not (input_args and isinstance(input_args[0], torch.Tensor)):
            raise KnownRBMError("Input data should not be empty and only contain tensor types")
        return self.fc1(input_args[0])


@final
@dataclass
class RBMLCI(InitContainer, LegoContInit):
    in_features: int
    out_features: int
    gibbs_sampling: int
    rbm_type: str


def create_rbm_net(args_con: LegoContInit, /) -> CurrentNetData:
    if not isinstance(args_con, RBMLCI):
        raise KnownRBMError(f"Expected {RBMLCI.__name__} got {type(args_con).__name__}")
    rbm_module = get_rbm_from_lib(args_con.rbm_type, RBMParams(
        in_features=args_con.in_features,
        out_features=args_con.out_features,
        gibbs_sampling=args_con.gibbs_sampling
    ))
    return CurrentNetData(
        fitness=float('inf'),
        com=rbm_module,
        lego=BRBMNetCon(rbm_module)
    )
