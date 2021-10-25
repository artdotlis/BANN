# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import Dict, Callable, Final

from bann.b_frameworks.pytorch.pytorch_lego_const import LegoContInit
from bann.b_frameworks.pytorch.net_model_interface import CurrentNetData

from bann_ex_con.pytorch.external_enum import ENetInterfaceNames

from bann_demo.pytorch.networks.b_nets.simple_demo import create_current_demo_net, \
    create_pre_training_demo_net

_LocalNetCreatorLib: Final[Dict[str, Callable[[LegoContInit], CurrentNetData]]] = {
    # TODO fill with a function from a list
    ENetInterfaceNames.SIMPLEDEMONET.value: create_current_demo_net,
    ENetInterfaceNames.SIMPLEDEMOPRENET.value: create_pre_training_demo_net
}


def get_pytorch_net_creators() -> Dict[str, Callable[[LegoContInit], CurrentNetData]]:
    return _LocalNetCreatorLib
