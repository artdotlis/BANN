# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import final

import torch


@final
@dataclass
class TruthFunArgCon:
    output: torch.Tensor
    target: torch.Tensor
    device: torch.device


@final
@dataclass
class TruthFunNoDevCon:
    output: torch.Tensor
    target: torch.Tensor
