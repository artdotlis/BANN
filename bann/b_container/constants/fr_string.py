# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from enum import Enum
from typing import final


@final
class StateNLib(Enum):
    LRSCH = 'lr_sch_'
    OPTIM = 'optim_'
    CRIT = 'criterion_'
    TR = 'trainer_'
    TE = 'tester_'
    PR = 'prepare_'


@final
class FrStPName(Enum):
    LRSCH = 'lr_sch'
    OPTIM = 'optim'
    CRIT = 'criterion'
    TR = 'trainer'
    TE = 'tester'
    PR = 'prepare'
