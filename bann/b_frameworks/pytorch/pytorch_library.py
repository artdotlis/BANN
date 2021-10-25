# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from copy import deepcopy
from typing import Tuple, Dict, Final

from bann.b_frameworks.errors.custom_erors import KnownLibError
from bann.b_pan_integration.framwork_key_lib import FrameworkKeyLib

from bann_ex_con.pytorch.external_library import get_e_pytorch_connections, \
    get_e_pytorch_net_interfaces

from pan.public.interfaces.config_constants import NetDictLibraryType
from pan.public.interfaces.net_connection import NetConnectionDict

_FRAMEWORK: Final[str] = FrameworkKeyLib.PYTORCH.value
_LocalConnectionLib: Final[NetConnectionDict] = NetConnectionDict(
    framework=_FRAMEWORK,
    con_dict={}
)
_LocalNetInterfaceLib: Final[NetDictLibraryType] = NetDictLibraryType(
    framework=_FRAMEWORK,
    net_dict={}
)


def _merge_dict(cont: Dict, to_merge_dict: Dict, /) -> None:
    for d_key, d_value in to_merge_dict.items():
        if d_key in cont:
            raise KnownLibError(f"Key {d_key} already defined!")
        cont[d_key] = d_value


def get_pytorch_connections() -> Tuple[str, NetConnectionDict]:
    external_lib = get_e_pytorch_connections()
    if external_lib[0] != _FRAMEWORK:
        raise KnownLibError(f"Expected {_FRAMEWORK} got {external_lib[0]}")
    if not isinstance(external_lib[1], NetConnectionDict):
        raise KnownLibError(
            f"Expected type {NetConnectionDict.__name__} got {type(external_lib[1]).__name__}"
        )
    new_dict = deepcopy(_LocalConnectionLib)
    _merge_dict(new_dict.con_dict, external_lib[1].con_dict)
    erg = (_FRAMEWORK, new_dict)
    return erg


def get_pytorch_net_interfaces() -> Tuple[str, NetDictLibraryType]:
    external_lib = get_e_pytorch_net_interfaces()
    if external_lib[0] != _FRAMEWORK:
        raise KnownLibError(f"Expected {_FRAMEWORK} got {external_lib[0]}")
    if not isinstance(external_lib[1], NetDictLibraryType):
        raise KnownLibError(
            f"Expected type {NetDictLibraryType.__name__} got {type(external_lib[1]).__name__}"
        )
    new_dict = deepcopy(_LocalNetInterfaceLib)
    _merge_dict(new_dict.net_dict, external_lib[1].net_dict)
    erg = (_FRAMEWORK, new_dict)
    return erg
