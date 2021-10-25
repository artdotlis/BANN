# -*- coding: utf-8 -*-
"""
    Inspired by:
        http://link.springer.com/10.1007/s11721-007-0002-0

        title:
            Particle swarm optimization: An overview
        authors:
            Poli, Riccardo and Kennedy, James and Blackwell, Tim

.. moduleauthor:: Artur Lissin
"""
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Generator, List, Tuple, Union, Type, final

from bann.b_hyper_optim.dynamic_programming.dp_matrix import DPMatrix
from bann.b_hyper_optim.optimiser.fun_const.hyper_fun import h_check_end_fitness, \
    h_create_random_params, h_create_v_max
from bann.b_hyper_optim.optimiser.fun_const.pso_const import PsoOptimArgs, HyperExtras, \
    PSOHyperSt, SwarmPopulation
from bann.b_hyper_optim.optimiser.fun_const.pso_fun import pso_calc_new_vel, pso_calc_new_pos


@final
@dataclass
class _CapValMaxArgs:
    v_index: int
    vel_pos: int
    v_max: float
    type_val: Union[Type[float], Type[int]]


def _cap_val_max(cont: _CapValMaxArgs, particle_swarm: SwarmPopulation,
                 hyper_state: PSOHyperSt, /) -> float:
    new_vel = pso_calc_new_vel(cont.v_index, cont.vel_pos, particle_swarm, hyper_state)
    if cont.v_max < abs(new_vel):
        new_vel = cont.v_max if new_vel >= 0 else -1 * cont.v_max
    if cont.type_val == int:
        new_vel = int(new_vel)
    return new_vel


def _pso_update_swarm(particle_swarm: SwarmPopulation, hyper_state: PSOHyperSt,
                      search_space_type: Tuple[Union[Type[int], Type[float]], ...], /) -> None:
    list_new_vel = [
        tuple(
            _cap_val_max(
                _CapValMaxArgs(
                    v_index=v_index,
                    vel_pos=vel_pos,
                    v_max=particle_swarm.v_max[vel_pos],
                    type_val=search_space_type[vel_pos],
                ), particle_swarm, hyper_state
            )
            for vel_pos, vel in enumerate(vel_tuple)
        )
        for v_index, vel_tuple in enumerate(particle_swarm.individual_vel)
    ]
    particle_swarm.individual_vel = list_new_vel
    random.seed()
    for list_index, vel_tuple in enumerate(list_new_vel):
        if not sum(abs(val) for val in vel_tuple):
            particle_swarm.individual_vel[list_index] = tuple(
                int(round(
                    (-1 if random.random() < 0.5 else 1)
                    * particle_swarm.v_max[pos] * random.random()
                ))
                if search_space_type[pos] == int else
                (-1 if random.random() < 0.5 else 1) * particle_swarm.v_max[pos] * random.random()
                for pos in range(len(vel_tuple))
            )
    list_new_pos = [
        tuple(
            pso_calc_new_pos(v_index, vel_pos, particle_swarm)
            for vel_pos, vel in enumerate(vel_tuple)
        )
        for v_index, vel_tuple in enumerate(particle_swarm.individual_vel)
    ]
    particle_swarm.individual_pos = list_new_pos


def _pso_init_swarm(swarm_size: int, swarm_extras: HyperExtras, /) \
        -> SwarmPopulation:
    param_sum = len(swarm_extras.search_space_type)
    randomised_pos = [h_create_random_params(swarm_extras) for _ in range(swarm_size)]
    res = SwarmPopulation(
        global_fitness=float('inf'),
        global_best_pos=randomised_pos[0],
        individual_fitness=[float('inf') for _ in range(swarm_size)],
        individual_pos=deepcopy(randomised_pos),
        individual_best_pos=deepcopy(randomised_pos),
        individual_vel=[tuple(0 for _ in range(param_sum)) for _ in range(swarm_size)],
        swarm_size=swarm_size,
        v_max=h_create_v_max(swarm_extras)
    )
    random.shuffle(res.individual_best_pos)
    return res


@final
class PSOCont:
    def __init__(self, arg_con: PsoOptimArgs, swarm_extras: HyperExtras,
                 hyper_state: PSOHyperSt, /) -> None:
        super().__init__()
        self._swarm_extras = swarm_extras
        self._arg_con = arg_con
        self._hyper_state = hyper_state
        self._swarm = _pso_init_swarm(self._arg_con.swarm_size, self._swarm_extras)
        self._dp_m = DPMatrix(arg_con.memory, arg_con.memory_repeats)

    def update_swarm_extras(self, swarm_extras: HyperExtras, /) -> None:
        self._swarm_extras = swarm_extras

    def hyper_optim(self) \
            -> Generator[List[Tuple[float, ...]], List[Tuple[float, Tuple[float, ...]]], None]:
        run_cnt = 0
        while run_cnt < self._arg_con.repeats and h_check_end_fitness(
                self._swarm.global_fitness, self._arg_con.end_fitness
        ):
            puf_y = self._dp_m.cr_hyper_params(
                self._swarm.individual_pos[0: self._swarm.swarm_size]
            )
            puf_r = []
            if puf_y:
                puf_r = yield puf_y
            for swarm_id, fit in enumerate(self._dp_m.up_hyper_params(puf_r)):
                fitness = fit[0]
                if fitness < self._swarm.global_fitness:
                    self._swarm.global_best_pos = fit[1]
                    self._swarm.global_fitness = fitness
                if fitness < self._swarm.individual_fitness[swarm_id]:
                    self._swarm.individual_fitness[swarm_id] = fitness
                    self._swarm.individual_best_pos[swarm_id] = fit[1]

            run_cnt += 1
            _pso_update_swarm(self._swarm, self._hyper_state, self._swarm_extras.search_space_type)
