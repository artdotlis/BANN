# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from enum import Enum
from typing import final


@final
class TestSubStrSuf(Enum):
    STRUCT = "struct"
    RESULT = "results"
    OUTPUT = "output"


@final
class TrainSubStrSuf(Enum):
    FITNESS = "fitness"
    TRUTH = "truth"
