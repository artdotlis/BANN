# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import Final

from bann.b_pan_integration.framwork_key_lib import FrameworkKeyLib

from bann_demo.pytorch.networks.errors.custom_erors import KnownSimpleDemoError

from bann_ex_con.pytorch.external_enum import EPConnectionNames

from pan.public.interfaces.net_connection import NetConnectionWr

_FRAMEWORK: Final[str] = FrameworkKeyLib.PYTORCH.value
_CON_NAME: Final[str] = EPConnectionNames.DEMO.value


def get_con_demo_name() -> str:
    return _CON_NAME


def get_demo_connection(framework: str, /) -> NetConnectionWr:
    if framework != _FRAMEWORK:
        raise KnownSimpleDemoError(f"Connection expected framework {_FRAMEWORK} got {framework}")
    return NetConnectionWr(_CON_NAME, _FRAMEWORK)
