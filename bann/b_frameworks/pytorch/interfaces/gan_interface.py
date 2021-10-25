# -*- coding: utf-8 -*-
"""inspired by:
    - https://github.com/pytorch/examples/blob/master/dcgan/main.py

LICENSE (BSD 3-Clause License): see extra_licenses/LICENSE_P_EXAMPLES

.. moduleauthor:: Artur Lissin
"""
import abc
from dataclasses import dataclass
from typing import Tuple, final, Generic, TypeVar

import torch
from torch import nn

from bann.b_frameworks.pytorch.interfaces.truth_interface import TruthClassInterface
from bann.b_container.states.framework.pytorch.criterion_param import CriterionAlias
from bann.b_container.states.framework.interface.pytorch.optim_per_parameter import \
    PerParameterNetInterface


@final
@dataclass
class GanInTarget:
    input: Tuple[torch.Tensor, ...]
    target: torch.Tensor


_GIn = TypeVar('_GIn')
_GOut = TypeVar('_GOut')


class GanInterface(Generic[_GIn, _GOut], PerParameterNetInterface, abc.ABC):
    @abc.abstractmethod
    def create_input_target(self, input_t: Tuple[torch.Tensor, ...], device: torch.device,
                            generator: bool, /) -> GanInTarget:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def fix_target_d(self, targets: torch.Tensor, device: torch.device, /) -> torch.Tensor:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def forward_gan(self, input_t: Tuple[torch.Tensor, ...],
                    device: torch.device, /) -> torch.Tensor:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def criterion(self, output_d: _GOut, target_d: _GIn,
                  device: torch.device, criterion: CriterionAlias, /) -> torch.Tensor:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def truth(self, output_d: _GOut, target_d: _GIn,
              device: torch.device, truth_fun: TruthClassInterface, /) -> float:
        raise NotImplementedError('Interface!')

    @property
    @abc.abstractmethod
    def generator(self) -> nn.Module:
        raise NotImplementedError('Interface!')

    @property
    @abc.abstractmethod
    def discriminator(self) -> nn.Module:
        raise NotImplementedError('Interface!')
