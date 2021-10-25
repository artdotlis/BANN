# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import abc

import torch
from torch import nn


class RBMInterface(nn.Module, abc.ABC):

    @abc.abstractmethod
    def _h_to_v(self, hidden: torch.Tensor, /) -> torch.Tensor:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def _v_to_h(self, visible: torch.Tensor, /) -> torch.Tensor:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def _gibbs_sampling(self, input_args: torch.Tensor, /) -> torch.Tensor:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def _free_energy(self, input_d: torch.Tensor, /) -> torch.Tensor:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def rbm(self) -> nn.Module:
        raise NotImplementedError('Interface!')
