# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Optional, Tuple, List, final

from bann.b_hyper_optim.optimiser.fun_const.hyper_const import HyperOArgs


@final
@dataclass
class EGAOptimArgs(HyperOArgs):
    population: int
    first_run: Optional[Tuple[float, ...]]
    cross_over_r: float
    mut_r: float
    memory_repeats: int
    memory: int


@final
@dataclass
class GAPopulation:
    fittest_id: List[int]
    individual_fitness: List[float]
    individual: List[Tuple[float, ...]]
    v_max: Tuple[float, ...]
    pop_size: int
    genome: int
    global_fitness: float
