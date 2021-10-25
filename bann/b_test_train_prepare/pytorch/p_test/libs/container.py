# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import final

import torch


@final
@dataclass
class OneClassOutTar:
    output_act: torch.Tensor
    output: torch.Tensor
    target: torch.Tensor


@final
@dataclass
class RegOutTarDevC:
    output: torch.Tensor
    target: torch.Tensor
    device: torch.device
