# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import Dict

from bann.b_frameworks.pytorch.pytorch_library import get_pytorch_connections
from bann.b_pan_integration.errors.custom_erors import NetConnectionError
from pan.public.interfaces.net_connection import NetConnectionDict


def net_connection_config() -> Dict[str, NetConnectionDict]:
    erg_dict: Dict[str, NetConnectionDict] = {}

    pytorch = get_pytorch_connections()
    if pytorch[0] in erg_dict:
        raise NetConnectionError(
            f"The framework {pytorch[0]} was already registered!"
        )
    erg_dict[pytorch[0]] = pytorch[1]

    return erg_dict
