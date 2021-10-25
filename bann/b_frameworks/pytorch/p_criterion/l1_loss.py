# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import final, Final, Tuple, Any

import torch
from torch.nn.functional import l1_loss

from bann.b_frameworks.pytorch.p_criterion.p_reduction_std_enum import LossReduction
from bann.b_frameworks.errors.custom_erors import KnownCriterionError
from bann.b_frameworks.pytorch.interfaces.custom_criterion_interface import PCustomLoss


@final
class L1Loss(PCustomLoss[torch.Tensor]):
    weight: Final[Tuple[float, float]]

    def __init__(self, reduction: str, weight: Tuple[float, float]) -> None:
        super().__init__(reduction)
        self.weight = weight

    def forward(self, *input_args: Any) -> torch.Tensor:
        if len(input_args) != 2:
            raise KnownCriterionError(f"Expected two arguments got {len(input_args)}")
        results, target = input_args
        if not(isinstance(results, torch.Tensor) and isinstance(target, torch.Tensor)):
            raise KnownCriterionError(
                f"{L1Loss.__name__} can only work with tensor types as input"
            )
        if target.size() != results.size():
            raise KnownCriterionError(
                f"({L1Loss.__name__}) Mismatch sizes t: {target.size()}; i: {results.size()}"
            )
        # l1 - loss implemented in PyTorch
        ret = l1_loss(results, target, reduction='none')
        ################################################
        if self.weight:
            with torch.no_grad():
                results_checked = results.ge(self.weight[0])
                target_checked = target.ge(self.weight[0])
                ones = torch.ones(target.size(), device=results.device)
                ones_w = torch.ones(target.size(), device=results.device) * self.weight[1]
                mul = torch.where(results_checked | target_checked, ones_w, ones)
            ret *= mul
        if self.reduction == LossReduction.MEAN.value:
            return torch.mean(ret)
        if self.reduction == LossReduction.SUM.value:
            return torch.sum(ret)
        return ret
