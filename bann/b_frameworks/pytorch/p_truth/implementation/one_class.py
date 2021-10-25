# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import final

import torch

from bann.b_frameworks.pytorch.interfaces.truth_interface import TruthClassInterface
from bann.b_frameworks.pytorch.p_truth.p_truth_enum import PTruthId
from bann.b_frameworks.pytorch.p_truth.p_truth_fun_arg_types import TruthFunArgCon


def calc_one_class_sim(input_val: TruthFunArgCon, /) -> int:
    target_l = input_val.target
    if target_l.device != input_val.device:
        target_l.to(input_val.device)
    output_l = input_val.output
    if output_l.device != input_val.device:
        output_l.to(input_val.device)
    prediction = output_l.max(1)[1]
    return int(prediction.eq(target_l).sum().item())


@final
class TruthOneClass(TruthClassInterface[TruthFunArgCon, int]):
    def cr_truth_container(self, output: torch.Tensor, target: torch.Tensor,
                           device: torch.device, /) -> TruthFunArgCon:
        return TruthFunArgCon(output=output, target=target, device=device)

    def calc_truth(self, input_val: TruthFunArgCon, /) -> int:
        return calc_one_class_sim(input_val)

    @staticmethod
    def truth_name() -> str:
        return PTruthId.ONECLASS.value
