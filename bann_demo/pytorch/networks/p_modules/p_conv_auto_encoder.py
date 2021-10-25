# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from enum import Enum
from typing import Tuple, List, Any, final, Final

import torch
from torch import nn
from torch.nn import functional as nn_fun

from bann.b_frameworks.pytorch.interfaces.ae_interface import AutoEncoderInterface

from bann_demo.pytorch.networks.errors.custom_erors import KnownAutoEncoderError


@final
class PConvCl(Enum):
    D1 = "Conv1D"
    D2 = "Conv2D"
    D3 = "Conv3D"


def get_ae_conv_types() -> List[str]:
    return [conv_t.value for conv_t in PConvCl.__members__.values()]


def check_ae_conv_type(conv_str: str, /) -> None:
    lib_list = get_ae_conv_types()
    if conv_str not in lib_list:
        raise KnownAutoEncoderError(f"{conv_str} could not be found in {lib_list}")


_CONV: Final = {
    PConvCl.D1.value: lambda in_n, out_n, kern: nn.Conv1d(in_n, out_n, kern, padding=int(kern / 2)),
    PConvCl.D2.value: lambda in_n, out_n, kern: nn.Conv2d(in_n, out_n, kern, padding=int(kern / 2)),
    PConvCl.D3.value: lambda in_n, out_n, kern: nn.Conv3d(in_n, out_n, kern, padding=int(kern / 2))
}
_POOL: Final = {
    PConvCl.D1.value: lambda: nn.MaxPool1d(2),
    PConvCl.D2.value: lambda: nn.MaxPool2d(2),
    PConvCl.D3.value: lambda: nn.MaxPool3d(2)
}


def _calc_dim(dim_ten: torch.Tensor, layer_cnt: int, /) -> List[torch.Tensor]:
    out_num = [dim_ten.float().floor()]
    puf_ten = out_num[-1]
    end_layer = 1
    for _ in range(layer_cnt - 1):
        puf_ten = (puf_ten / 2).floor()
        if puf_ten.le(2).sum() >= 1:
            break
        end_layer += 1
        out_num.append(puf_ten)
    return [new_el.int() for new_el in out_num]


@final
class PConvAutoEncoderHG(AutoEncoderInterface):
    def __init__(self, net_width: Tuple[int, int, torch.Tensor], layers_cnt: int,
                 kernel_cnt: int, cnn_dim: str) -> None:
        check_ae_conv_type(cnn_dim)
        super().__init__()
        if net_width[2].dim() > 1:
            raise KnownAutoEncoderError(f"Expected only one dimension got {net_width[2].dim()}")
        k_size = kernel_cnt
        if kernel_cnt / 2.0 == int(kernel_cnt / 2.0):
            k_size += 1
        channels = net_width[0]
        start_input = net_width[1]
        self.__layer_dim = _calc_dim(net_width[2], layers_cnt)
        end_layer = len(self.__layer_dim)
        puf_list: List[nn.Module] = [_CONV[cnn_dim](channels, start_input * 2, kernel_cnt)]
        puf_list.extend(
            _CONV[cnn_dim](
                index_num * start_input * 2, (1 + index_num) * start_input * 2, kernel_cnt
            )
            for index_num in range(1, end_layer)
        )
        self.encoder_conv: nn.ModuleList = nn.ModuleList(puf_list)
        self.encoder_pool: nn.Module = _POOL[cnn_dim]()
        puf_list = [
            _CONV[cnn_dim](
                (1 + index_num) * start_input * 2, index_num * start_input * 2, kernel_cnt
            )
            for index_num in range(end_layer - 1, 0, -1)
        ]
        puf_list.append(_CONV[cnn_dim](start_input * 2, channels, kernel_cnt))
        self.decoder_conv: nn.ModuleList = nn.ModuleList(puf_list)
        self.upsample: nn.ModuleList = nn.ModuleList([
            nn.Upsample(size=tuple(self.__layer_dim[index_num].tolist()))
            for index_num in range(end_layer - 1, -1, -1)
        ])

    def forward(self, *input_args: Any) -> torch.Tensor:
        if not (input_args and isinstance(input_args[0], torch.Tensor)):
            raise KnownAutoEncoderError(
                "Input data should not be empty and only contain tensor types"
            )
        input_data = input_args[0]
        encoder_l_conf: nn.ModuleList = self.encoder_conv
        for l_encode in encoder_l_conf.__iter__():
            input_data = nn_fun.relu(l_encode(input_data))
            input_data = self.encoder_pool(input_data)
        decoder_l_conv: nn.ModuleList = self.decoder_conv
        upsample_l: nn.ModuleList = self.upsample
        l_decode: nn.Module
        for d_i, l_decode in enumerate(decoder_l_conv.__iter__()):
            up_s_mod: nn.Module = upsample_l[d_i]
            input_data = nn_fun.relu(l_decode(input_data))
            input_data = up_s_mod(input_data)
        return input_data

    def encode(self, *input_args: torch.Tensor) -> torch.Tensor:
        if not input_args:
            raise KnownAutoEncoderError("Input data should not be empty")
        input_data = input_args[0]
        with torch.no_grad():
            encoder_l_conf: nn.ModuleList = self.encoder_conv
            for l_encode in encoder_l_conf.__iter__():
                input_data = nn_fun.relu(l_encode(input_data))
                input_data = self.encoder_pool(input_data)
        return input_data

    def decode(self, *input_args: torch.Tensor) -> torch.Tensor:
        if not input_args:
            raise KnownAutoEncoderError("Input data should not be empty")
        input_data = input_args[0]
        with torch.no_grad():
            decoder_l_conv: nn.ModuleList = self.decoder_conv
            upsample_l: nn.ModuleList = self.upsample
            l_decode: nn.Module
            for d_i, l_decode in enumerate(decoder_l_conv.__iter__()):
                up_s_mod: nn.Module = upsample_l[d_i]
                input_data = nn_fun.relu(l_decode(input_data))
                input_data = up_s_mod(input_data)
        return input_data

    @property
    def encoder(self) -> Tuple[Tuple[nn.Module, ...], Tuple[nn.Module, ...]]:
        encoder_conv_l: nn.ModuleList = self.encoder_conv
        return tuple(encoder_conv_l.children()), tuple(
            self.encoder_pool for _ in range(len(encoder_conv_l))
        )

    @property
    def decoder(self) -> Tuple[Tuple[nn.Module, ...], Tuple[nn.Module, ...]]:
        decoder_l_conv: nn.ModuleList = self.decoder_conv
        upsample_l: nn.ModuleList = self.upsample
        return tuple(decoder_l_conv.children()), tuple(upsample_l.children())
