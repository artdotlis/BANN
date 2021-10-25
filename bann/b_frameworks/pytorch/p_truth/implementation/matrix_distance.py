# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import math
from typing import final

import torch
from torch import nn

from bann.b_frameworks.pytorch.interfaces.truth_interface import TruthClassInterface
from bann.b_frameworks.pytorch.p_truth.p_truth_fun_arg_types import TruthFunArgCon
from bann.b_frameworks.pytorch.p_truth.p_truth_enum import PTruthId


def calc_matrix_sim(input_val: TruthFunArgCon, distance: nn.PairwiseDistance, /) -> float:
    target_l = input_val.target
    if target_l.device != input_val.device:
        target_l.to(input_val.device)
    output_l = input_val.output
    if output_l.device != input_val.device:
        output_l.to(input_val.device)
    size_batch = input_val.output.size(0)
    res = float(distance(
        output_l.view(size_batch, -1), target_l.view(size_batch, -1)
    ).sum())
    if not (math.isinf(res) or math.isnan(res)) and res > 0:
        return res
    return -3.0


@final
class TruthMSimClass(TruthClassInterface[TruthFunArgCon, float]):
    def __init__(self) -> None:
        super().__init__()
        self.__pairwise = nn.PairwiseDistance()

    def cr_truth_container(self, output: torch.Tensor, target: torch.Tensor,
                           device: torch.device, /) -> TruthFunArgCon:
        return TruthFunArgCon(output=output, target=target, device=device)

    def calc_truth(self, input_val: TruthFunArgCon, /) -> float:
        return calc_matrix_sim(input_val, self.__pairwise)

    @staticmethod
    def truth_name() -> str:
        return PTruthId.MSIM.value
