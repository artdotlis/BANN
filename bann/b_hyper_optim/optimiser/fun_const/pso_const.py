# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Type, List, final

from bann.b_hyper_optim.optimiser.fun_const.hyper_const import HyperOArgs


@final
@dataclass
class PSOHyperSt:
    c_1: float
    c_2: float
    weight: float


@dataclass
class PsoOptimArgs(HyperOArgs):
    swarm_size: int
    memory_repeats: int
    memory: int


@final
@dataclass
class DaPsoOptimArgs(PsoOptimArgs):
    first_run: Optional[Tuple[float, ...]]
    l_repeats: int
    l_increase: float
    alpha: float
    s_prob: float
    survival_rate: float
    dim_prob: float


@final
@dataclass
class HyperExtras:
    search_space_min: Tuple[float, ...]
    search_space_max: Tuple[float, ...]
    search_space_type: Tuple[Union[Type[float], Type[int]], ...]


@dataclass
class SwarmPopulation:
    global_fitness: float
    global_best_pos: Tuple[float, ...]
    individual_fitness: List[float]
    individual_pos: List[Tuple[float, ...]]
    individual_best_pos: List[Tuple[float, ...]]
    individual_vel: List[Tuple[float, ...]]
    v_max: Tuple[float, ...]
    swarm_size: int
