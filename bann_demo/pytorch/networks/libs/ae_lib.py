# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Callable, Any, Type, Tuple, final, Final

import torch

from bann.b_frameworks.pytorch.interfaces.ae_interface import AutoEncoderInterface

from bann_demo.pytorch.networks.p_modules.p_auto_encoder import PAutoEncoderHG
from bann_demo.pytorch.networks.p_modules.p_conv_auto_encoder import PConvAutoEncoderHG
from bann_demo.pytorch.networks.errors.custom_erors import KnownAutoEncoderError


@final
class AELib(Enum):
    AE = "Autoencoder"
    CNNAE = "Conv autoencoder"


def get_ae_list() -> List[str]:
    return [ae_t.value for ae_t in AELib.__members__.values()]


@dataclass(init=False)
class AEParamInterface:
    def __init__(self, **_: Any) -> None:  # type: ignore
        super().__init__()


@dataclass
class AEParams(AEParamInterface):
    input_cnt: int
    layers_cnt: int


@dataclass
class CNNAEParams(AEParamInterface):
    net_width: Tuple[int, int, torch.Tensor]
    layers_cnt: int
    kernel_cnt: int
    cnn_dim: str


_AELib: Final[Dict[str, Callable[[AEParamInterface], AutoEncoderInterface]]] = {
    AELib.AE.value: lambda values: PAutoEncoderHG(**values.__dict__),
    AELib.CNNAE.value: lambda values: PConvAutoEncoderHG(**values.__dict__)
}
_AEParamType: Final[Dict[str, Type[AEParamInterface]]] = {
    AELib.AE.value: AEParams,
    AELib.CNNAE.value: CNNAEParams
}


def get_ae_from_lib(ae_name: str, params: AEParamInterface, /) -> AutoEncoderInterface:
    auto_encoder = _AELib.get(ae_name, None)
    p_type = _AEParamType.get(ae_name, None)
    if auto_encoder is None or p_type is None:
        raise KnownAutoEncoderError(f"Could not find the autoencoder: {ae_name}!")
    if not isinstance(params, p_type):
        raise KnownAutoEncoderError(f"Expected {p_type.__name__} got {type(params).__name__}")
    return auto_encoder(params)
