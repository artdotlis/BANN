# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Callable, final, Final

from bann.b_frameworks.pytorch.interfaces.rbm_interface import RBMInterface

from bann_demo.pytorch.networks.p_modules.rbm_bern import RBMBern
from bann_demo.pytorch.networks.p_modules.rbm_gauss import RBMGauss
from bann_demo.pytorch.networks.errors.custom_erors import KnownRBMError


@final
class RBMLib(Enum):
    BBRBM = "BinaryBinaryRBM"
    GBRBM = "GaussianBinaryRBM"


def get_rbm_list() -> List[str]:
    return [rbm_t.value for rbm_t in RBMLib.__members__.values()]


@final
@dataclass
class RBMParams:
    in_features: int
    out_features: int
    gibbs_sampling: int


_RBMLib: Final[Dict[str, Callable[[RBMParams], RBMInterface]]] = {
    RBMLib.BBRBM.value: lambda values: RBMBern(**values.__dict__),
    RBMLib.GBRBM.value: lambda values: RBMGauss(**values.__dict__)
}


def get_rbm_from_lib(rbm_name: str, params: RBMParams, /) -> RBMInterface:
    rbm = _RBMLib.get(rbm_name, None)
    if rbm is None:
        raise KnownRBMError(f"Could not find the rbm: {rbm_name}!")
    return rbm(params)
