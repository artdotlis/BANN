# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import Dict

from bann.b_frameworks.pytorch.pytorch_library import get_pytorch_net_interfaces
from bann.b_pan_integration.errors.custom_erors import NetInterfaceError
from pan.public.interfaces.config_constants import NetDictLibraryType


def net_interface_config() -> Dict[str, NetDictLibraryType]:
    erg_dict: Dict[str, NetDictLibraryType] = {}

    pytorch = get_pytorch_net_interfaces()
    if pytorch[0] in erg_dict:
        raise NetInterfaceError(
            f"The framework {pytorch[0]} was already registered!"
        )
    erg_dict[pytorch[0]] = pytorch[1]

    return erg_dict
