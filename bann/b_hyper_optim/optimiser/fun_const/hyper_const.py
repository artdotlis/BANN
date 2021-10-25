# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Optional, final


@dataclass
class HyperOArgs:
    end_fitness: Optional[float]
    repeats: int


@final
@dataclass
class RandomOArgs(HyperOArgs):
    package: int


@final
@dataclass
class SimplyTrainOArgs:
    repeats: int
    package: int
