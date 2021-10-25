# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from enum import Enum
from typing import final


@final
class GenStateNLib(Enum):
    HYPER = 'hyper_'
    INIT = 'init_'
    NET = 'net_'


@final
class GenStPaName(Enum):
    HYPER = 'hyper'
    INIT = 'init'
    NET = 'net'
