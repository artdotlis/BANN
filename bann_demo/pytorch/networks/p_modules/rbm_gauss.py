# -*- coding: utf-8 -*-
"""Restricted Boltzmann Machine with Gauss sampling

Inspired by:
    https://dx.plos.org/10.1371/journal.pone.0171015

    title:
        Gaussian-binary restricted Boltzmann machines for modeling natural image statistics
    author:
        Melchior, Jan and Wang, Nan and Wiskott, Laurenz

Energy function mentioned in:
     https://stats.stackexchange.com/questions/114844/
     how-to-compute-the-free-energy-of-a-rbm-given-its-energy

Input-data needs to have a standard deviation of 1

.. moduleauthor:: Artur Lissin
"""
import math
from typing import Any, final

import torch
from torch import nn
import torch.nn.functional as nn_fun
from torch.distributions import Normal

from bann.b_frameworks.pytorch.interfaces.rbm_interface import RBMInterface
from bann.b_frameworks.pytorch.interfaces.criterion_less import CriterionLess

from bann_demo.pytorch.networks.errors.custom_erors import KnownRBMError


@final
class RBMGauss(CriterionLess[torch.Tensor, torch.Tensor], RBMInterface):
    def __init__(self, in_features: int, out_features: int, gibbs_sampling: int) -> None:
        super().__init__()
        self.weight: nn.Parameter = nn.Parameter(
            torch.zeros(out_features, in_features), requires_grad=True
        )
        self.v_bias: nn.Parameter = nn.Parameter(torch.zeros(in_features), requires_grad=True)
        self.h_bias: nn.Parameter = nn.Parameter(torch.zeros(out_features), requires_grad=True)
        self.g_s_cnt = gibbs_sampling
        self.in_features = in_features
        self.out_features = out_features
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def extra_repr(self) -> str:
        erg = f"Gibbs sampling k {self.g_s_cnt}, visible nodes {self.in_features}," \
              + f" hidden nodes {self.out_features}"
        return erg

    def forward(self, *input_args: Any) -> torch.Tensor:
        if not (input_args and isinstance(input_args[0], torch.Tensor)):
            raise KnownRBMError(
                "Input data should not be empty and only contain tensor types"
            )
        if input_args[0].dim() != 2:
            raise KnownRBMError(f"Input data should have 2 dim got {input_args[0].dim()}")
        return self._gibbs_sampling(input_args[0])

    def _h_to_v(self, hidden: torch.Tensor, /) -> torch.Tensor:
        return Normal(nn_fun.linear(hidden, self.weight.t(), self.v_bias), 1.0).sample()

    def _v_to_h(self, visible: torch.Tensor, /) -> torch.Tensor:
        return torch.sigmoid(torch.relu(
            nn_fun.linear(visible, self.weight, self.h_bias)
        )).bernoulli()

    def _gibbs_sampling(self, input_args: torch.Tensor, /) -> torch.Tensor:
        with torch.no_grad():
            visible_g = input_args
            for _ in range(self.g_s_cnt):
                hidden_g = self._v_to_h(visible_g)
                visible_g = self._h_to_v(hidden_g)
        return visible_g

    def _free_energy(self, input_d: torch.Tensor, /) -> torch.Tensor:
        return (
            (input_d - self.v_bias).pow(2).sum(1) * 0.5
            - (nn_fun.linear(input_d, self.weight, self.h_bias).exp() + 1).log().sum(1)
        ).mean()

    def rbm(self) -> nn.Linear:
        puf = nn.Linear(self.in_features, self.out_features)
        puf.weight = self.weight
        puf.bias = self.h_bias
        return puf

    def criterion(self, output_d: torch.Tensor, input_d: torch.Tensor, /) -> torch.Tensor:
        return self._free_energy(input_d) - self._free_energy(output_d)

    @staticmethod
    def criterion_str() -> str:
        return f"{RBMGauss.__name__}: Energy based loss"
