# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from enum import Enum
from typing import final


@final
class ENetLibName(Enum):
    # demo
    COMPLEX = 'ComplexNet'
    RBM = 'RBMNet'
    AE = 'AutoEncoderNet'
    # e_networks


@final
class EInitLibName(Enum):
    # e_networks
    pass
