# -*- coding: utf-8 -*-
"""Spatial Pyramid Pooling

Explained in paper:
    http://link.springer.com/10.1007/978-3-319-10578-9_23

    title:
        Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
    author:
        He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian

.. moduleauthor:: Artur Lissin
"""
from typing import Tuple, final

import torch
from torch.nn import functional as nn_fun

from bann.b_frameworks.pytorch.interfaces.spp_interface import SPPInterface


@final
class SPP1D(SPPInterface):
    def __init__(self, f_maps: int, bin_sizes: Tuple[int, ...], /) -> None:
        if not bin_sizes:
            self.bin_sizes = [1]
        else:
            self.bin_sizes = list(bin_sizes)
        self.f_maps = f_maps

    def get_nodes_cnt(self) -> int:
        return sum(bin_s * self.f_maps for bin_s in self.bin_sizes)

    def get_fm_cnt(self) -> int:
        return self.f_maps

    def get_bin_size_c(self) -> Tuple[int, ...]:
        return sum(bin_s for bin_s in self.bin_sizes),

    def spp(self, in_data: torch.Tensor, /) -> torch.Tensor:
        pool = [
            nn_fun.adaptive_max_pool1d(in_data, bin_size)
            for bin_size in self.bin_sizes
        ]
        return torch.cat(pool, 2)


@final
class SPP2D(SPPInterface):
    def __init__(self, f_maps: int, bin_sizes: Tuple[Tuple[int, int], ...], /) -> None:
        if not bin_sizes:
            self.bin_sizes: Tuple[Tuple[int, int], ...] = ((1, 1),)
        else:
            self.bin_sizes = bin_sizes
        self.f_maps = f_maps

    def get_nodes_cnt(self) -> int:
        return sum(bin_s[0] * bin_s[1] * self.f_maps for bin_s in self.bin_sizes)

    def get_fm_cnt(self) -> int:
        return self.f_maps

    def get_bin_size_c(self) -> Tuple[int, ...]:
        return (
            sum(bin_s[0] for bin_s in self.bin_sizes),
            sum(bin_s[1] for bin_s in self.bin_sizes)
        )

    def spp(self, in_data: torch.Tensor, /) -> torch.Tensor:
        pool = [
            nn_fun.adaptive_max_pool2d(in_data, bin_size).view(
                in_data.size(0), in_data.size(1), -1
            )
            for bin_size in self.bin_sizes
        ]
        return torch.cat(pool, 2)
