# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from enum import Enum
from typing import final


@final
class PTruthId(Enum):
    ONECLASS = 'OneClass'
    MSIM = 'MatrixDistance'
