# -*- coding: utf-8 -*-
"""
This implementation was inspired by the approach is described in the paper:
    A hybrid approach of dimension partition and velocity control
    to enhance performance of particle swarm optimization
author:
    Hsiao, Yu-Ting
    and Lee, Wei-Po
    and Wang, Ruei-Yang

.. moduleauthor:: Artur Lissin
"""
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple, List, Generator, Union, Type, Dict, final

from bann.b_hyper_optim.dynamic_programming.dp_matrix import DPMatrix
from bann.b_hyper_optim.optimiser.fun_const.hyper_fun import h_create_random_params, \
    h_create_v_max, h_check_end_fitness
from bann.b_hyper_optim.optimiser.fun_const.pso_fun import pso_calc_new_vel, pso_calc_new_pos
from bann.b_hyper_optim.optimiser.fun_const.pso_const import HyperExtras, SwarmPopulation, \
    PSOHyperSt, DaPsoOptimArgs


@final
@dataclass
class _AdaptiveSwarmPopulation(SwarmPopulation):
    individual_fitness_pre: List[float]
    individual_fitness_cur: List[float]
    dimensions_cnt: int
    iter_cnt: int
    group_b_start_pos: int


def _check_new_v_max(v_max_old: float, v_max_new: float,
                     v_max_type: Union[Type[float], Type[int]], /) -> float:
    v_old_abs = abs(v_max_old)
    v_new_abs = abs(v_max_new)
    if not v_new_abs:
        return v_old_abs
    if v_max_type == int and v_new_abs < 1:
        return 1.0
    return v_new_abs


@final
class _AdaptiveVelocityControl:
    def __init__(self, dimensions_cnt: int, alpha: float, prob: float, /) -> None:
        super().__init__()
        self.__half_v_max = 0.5
        self.__move_variation: List[float] = [0.0 for _ in range(dimensions_cnt)]
        self.__alpha = alpha
        self.__prob = prob
        self.__new_limit = 20.0

    def slow_down_swarm(self, swarm: _AdaptiveSwarmPopulation,
                        swarm_extras: HyperExtras, /) -> None:
        counter = 0
        for particle_index in range(swarm.group_b_start_pos):
            if swarm.individual_fitness_pre[particle_index] \
                    < swarm.individual_fitness_cur[particle_index]:
                counter += 1
        if counter > int(swarm.group_b_start_pos / 2):
            swarm.v_max = tuple(
                _check_new_v_max(
                    v_max_i, self.__alpha * v_max_i, swarm_extras.search_space_type[dim]
                )
                for dim, v_max_i in enumerate(swarm.v_max)
            )

    def speed_up_swarm_rule_1(self, swarm: _AdaptiveSwarmPopulation, particle_id: int,
                              swarm_extras: HyperExtras, /) -> None:
        if swarm.iter_cnt > 4:
            swarm_ind = swarm.individual_pos[particle_id]
            if particle_id < swarm.group_b_start_pos:
                for dimension, swarm_ind_dim in enumerate(swarm_ind):
                    dif_m = abs(swarm.global_best_pos[dimension] - swarm_ind_dim)
                    if self.__move_variation[dimension] < dif_m:
                        self.__move_variation[dimension] = dif_m
            if particle_id == swarm.group_b_start_pos - 1:
                swarm.v_max = tuple(
                    abs(v_max_dim) if v_max_dim >= self.__move_variation[dim_v] * self.__half_v_max
                    else _check_new_v_max(
                        v_max_dim, self.__half_v_max * self.__move_variation[dim_v],
                        swarm_extras.search_space_type[dim_v]
                    )
                    for dim_v, v_max_dim in enumerate(swarm.v_max)
                )
                self.__move_variation = [0 for _ in range(len(self.__move_variation))]

    def speed_up_swarm_rule_2(self, swarm: _AdaptiveSwarmPopulation, /) -> Tuple[float, ...]:
        random.seed()
        if self.__prob > random.random():
            return tuple(v_max_dim * self.__new_limit for v_max_dim in swarm.v_max)
        return swarm.v_max


def _da_pso_init_swarm(swarm_size: int, swarm_extras: HyperExtras, /) \
        -> _AdaptiveSwarmPopulation:
    param_sum = len(swarm_extras.search_space_type)
    group_b = int((3 * swarm_size) / 4)
    randomised_pos = [h_create_random_params(swarm_extras) for _ in range(swarm_size)]
    res = _AdaptiveSwarmPopulation(
        global_fitness=float('inf'),
        global_best_pos=randomised_pos[0],
        individual_fitness=[float('inf') for _ in range(swarm_size)],
        individual_fitness_pre=[float('inf') for _ in range(group_b)],
        individual_fitness_cur=[float('inf') for _ in range(group_b)],
        individual_pos=deepcopy(randomised_pos),
        individual_best_pos=deepcopy(randomised_pos),
        individual_vel=[tuple(0 for _ in range(param_sum)) for _ in range(swarm_size)],
        swarm_size=swarm_size,
        dimensions_cnt=param_sum,
        iter_cnt=0,
        v_max=h_create_v_max(swarm_extras),
        group_b_start_pos=group_b
    )
    random.shuffle(res.individual_best_pos)
    return res


def _update_group_b_vector(b_pos_ind: int, dim_to_keep: List[int],
                           particle_swarm: _AdaptiveSwarmPopulation, /) -> None:
    list_new_pos = [
        tuple(
            value if dim not in dim_to_keep
            else particle_swarm.individual_pos[swarm_id][dim]
            for dim, value in enumerate(particle_swarm.global_best_pos)
        )
        for swarm_id in range(b_pos_ind, particle_swarm.swarm_size)
    ]
    particle_swarm.individual_pos[b_pos_ind:particle_swarm.swarm_size] = list_new_pos
    list_new_best_pos = [
        tuple(
            value if dim not in dim_to_keep
            else particle_swarm.individual_best_pos[swarm_id][dim]
            for dim, value in enumerate(particle_swarm.global_best_pos)
        )
        for swarm_id in range(b_pos_ind, particle_swarm.swarm_size)
    ]
    particle_swarm.individual_best_pos[b_pos_ind: particle_swarm.swarm_size] = list_new_best_pos


def _update_fitness(fitness: float, swarm_id: int, particle_swarm: _AdaptiveSwarmPopulation,
                    fixed_pos: Tuple[float, ...], /) -> None:
    if fitness < particle_swarm.global_fitness:
        particle_swarm.global_best_pos = fixed_pos
        particle_swarm.global_fitness = fitness
    if fitness < particle_swarm.individual_fitness[swarm_id]:
        particle_swarm.individual_fitness[swarm_id] = fitness
        particle_swarm.individual_best_pos[swarm_id] = fixed_pos


@final
@dataclass
class _CapValMaxArgs:
    dim_up: List[int]
    v_index: int
    vel_pos: int
    v_max: float
    type_val: Union[Type[float], Type[int]]


def _cap_val_max(cont: _CapValMaxArgs, particle_swarm: _AdaptiveSwarmPopulation,
                 hyper_state: PSOHyperSt, /) -> float:
    if cont.vel_pos in cont.dim_up or -1 in cont.dim_up:
        new_vel = pso_calc_new_vel(cont.v_index, cont.vel_pos, particle_swarm, hyper_state)
        if cont.v_max < abs(new_vel):
            new_vel = cont.v_max if new_vel >= 0 else -1 * cont.v_max
        if cont.type_val == int:
            new_vel = int(new_vel)
        return new_vel
    return 0


@final
@dataclass
class _DaPsoUpdateCon:
    hyper_state: PSOHyperSt
    swarm_extras: HyperExtras
    avc: _AdaptiveVelocityControl


def _da_pso_update(particle_index: int, dim_cur: List[int],
                   particle_swarm: _AdaptiveSwarmPopulation, cont: _DaPsoUpdateCon, /) -> None:
    vel_vector = cont.avc.speed_up_swarm_rule_2(particle_swarm)
    tuple_new_vel = tuple(
        _cap_val_max(
            _CapValMaxArgs(
                v_index=particle_index,
                vel_pos=vel_pos,
                v_max=vel_vector[vel_pos],
                type_val=cont.swarm_extras.search_space_type[vel_pos],
                dim_up=[-1] if particle_index < particle_swarm.group_b_start_pos else dim_cur
            ), particle_swarm, cont.hyper_state)
        for vel_pos, vel in enumerate(particle_swarm.individual_vel[particle_index])
    )
    particle_swarm.individual_vel[particle_index] = tuple_new_vel
    tuple_new_pos = tuple(
        pso_calc_new_pos(particle_index, vel_pos, particle_swarm)
        if particle_index < particle_swarm.group_b_start_pos or vel_pos in dim_cur
        else particle_swarm.individual_pos[particle_index][vel_pos]
        for vel_pos, vel in enumerate(particle_swarm.individual_vel[particle_index])
    )
    particle_swarm.individual_pos[particle_index] = tuple_new_pos


@final
@dataclass
class OptimStepArgs:
    avc: _AdaptiveVelocityControl
    dim: List[int]
    hyper_state: PSOHyperSt
    swarm_extras: HyperExtras
    start: int
    stop: int


def _optim_step(particle_swarm: _AdaptiveSwarmPopulation,
                erg_puf: List[Tuple[float, Tuple[float, ...]]],
                cont: OptimStepArgs, /) -> None:
    for erg_pos, swarm_id in enumerate(range(cont.start, cont.stop)):
        cont.avc.speed_up_swarm_rule_1(particle_swarm, swarm_id, cont.swarm_extras)
        if swarm_id < particle_swarm.group_b_start_pos:
            particle_swarm.individual_fitness_pre[swarm_id] = \
                particle_swarm.individual_fitness_cur[swarm_id]
            particle_swarm.individual_fitness_cur[swarm_id] = erg_puf[erg_pos][0]
        _update_fitness(erg_puf[erg_pos][0], swarm_id, particle_swarm, erg_puf[erg_pos][1])
    for swarm_id in range(cont.start, cont.stop):
        if swarm_id == particle_swarm.group_b_start_pos:
            _update_group_b_vector(swarm_id, cont.dim, particle_swarm)
        _da_pso_update(
            swarm_id, cont.dim, particle_swarm,
            _DaPsoUpdateCon(
                hyper_state=cont.hyper_state,
                swarm_extras=cont.swarm_extras,
                avc=cont.avc
            )
        )


def _randomise_group_b(particle_swarm: _AdaptiveSwarmPopulation,
                       swarm_extras: HyperExtras, next_dim: List[int], /) -> None:
    start_pos = particle_swarm.group_b_start_pos
    randomised_group = [
        h_create_random_params(swarm_extras)
        for _ in range(start_pos + 1, particle_swarm.swarm_size)
    ]
    particle_swarm.individual_pos[start_pos] = tuple(
        int(round(
            new_pos + random.random()
            * (1 if random.random() < 0.5 else -1) * particle_swarm.v_max[dim]
        ))
        if swarm_extras.search_space_type[dim] == int else
        new_pos + random.random()
        * (1 if random.random() < 0.5 else -1) * particle_swarm.v_max[dim]
        for dim, new_pos in enumerate(particle_swarm.global_best_pos)
    )
    particle_swarm.individual_best_pos[start_pos] = particle_swarm.individual_pos[start_pos]
    particle_swarm.individual_pos[start_pos + 1: particle_swarm.swarm_size] = randomised_group
    particle_swarm.individual_best_pos[start_pos + 1: particle_swarm.swarm_size] = randomised_group
    particle_swarm.individual_fitness[start_pos: particle_swarm.swarm_size] = [
        float('inf')
        for _ in range(start_pos, particle_swarm.swarm_size)
    ]
    particle_swarm.individual_vel[start_pos: particle_swarm.swarm_size] = [
        tuple(0 for _ in particle_swarm.individual_vel[id_cur])
        for id_cur in range(start_pos, particle_swarm.swarm_size)
    ]
    _update_group_b_vector(start_pos, next_dim, particle_swarm)


def _only_the_fittest_survive(d_rate: float, swarm: _AdaptiveSwarmPopulation,
                              swarm_extras: HyperExtras, /) -> None:
    fitness_dict: Dict[float, List[int]] = {}
    die_out_cnt = int(swarm.group_b_start_pos * d_rate)
    for swarm_id in range(swarm.group_b_start_pos):
        list_v = fitness_dict.setdefault(swarm.individual_fitness[swarm_id], [])
        list_v.append(swarm_id)
    run_cnt = 0
    sorted_keys = iter(sorted(fitness_dict.keys())[::-1])
    while run_cnt < die_out_cnt:
        try:
            next_key = next(sorted_keys)
        except StopIteration:
            run_cnt = die_out_cnt
        else:
            dif = die_out_cnt - run_cnt
            if len(fitness_dict[next_key]) < dif:
                dif = len(fitness_dict[next_key])
            for value_index in range(dif):
                swarm.individual_pos[fitness_dict[next_key][value_index]] = \
                    h_create_random_params(swarm_extras)
                swarm.individual_vel[fitness_dict[next_key][value_index]] = tuple(
                    0 for _ in swarm.individual_vel[fitness_dict[next_key][value_index]]
                )
                run_cnt += 1


@final
@dataclass
class _RunCont:
    dimension: List[int]
    to_add: float
    inner_loop: float
    run_cnt: int = 0


def _create_dim_list(dims: List[int], prob: float, /) -> Tuple[List[int], ...]:
    random.seed()
    random.shuffle(dims)
    dim_sol: List[List[int]] = [[]]
    run_cnt = 0
    for d_ind, d_val in enumerate(dims):
        dim_sol[run_cnt].append(d_val)
        if random.random() >= prob and d_ind < len(dims) - 1:
            run_cnt += 1
            dim_sol.append([])
    return tuple(dim_sol)


@final
class DAPSOCont:
    def __init__(self, arg_con: DaPsoOptimArgs, swarm_extras: HyperExtras,
                 hyper_state: PSOHyperSt, /) -> None:
        super().__init__()
        self._swarm_extras = swarm_extras
        self._arg_con = arg_con
        self._hyper_state = hyper_state
        self._swarm = _da_pso_init_swarm(self._arg_con.swarm_size, self._swarm_extras)
        self._dp_m = DPMatrix(arg_con.memory, arg_con.memory_repeats)

    def update_swarm_extras(self, swarm_extras: HyperExtras, /) -> None:
        self._swarm_extras = swarm_extras

    def hyper_optim(self) \
            -> Generator[List[Tuple[float, ...]], List[Tuple[float, Tuple[float, ...]]], None]:
        if self._swarm.dimensions_cnt <= 1:
            print("Warning this algorithm should be used with more than one parameter!")
        avc = _AdaptiveVelocityControl(
            self._swarm.dimensions_cnt, self._arg_con.alpha, self._arg_con.s_prob
        )
        run_cont = _RunCont(
            dimension=list(range(self._swarm.dimensions_cnt)),
            to_add=self._arg_con.l_increase, inner_loop=self._arg_con.l_repeats
        )
        if run_cont.to_add < 0:
            run_cont.to_add = 0
        if self._arg_con.first_run is not None:
            self._swarm.individual_pos[0] = self._arg_con.first_run
        while h_check_end_fitness(
                self._swarm.global_fitness, self._arg_con.end_fitness
        ) and run_cont.run_cnt < self._arg_con.repeats:
            dim_sol = _create_dim_list(run_cont.dimension, self._arg_con.dim_prob)
            run_cont.run_cnt += 1
            for dim_list in dim_sol:
                self._swarm.iter_cnt += 1
                _randomise_group_b(self._swarm, self._swarm_extras, dim_list)
                for _ in range(int(run_cont.inner_loop)):
                    puf_y = self._dp_m.cr_hyper_params(
                        self._swarm.individual_pos[0: self._swarm.swarm_size]
                    )
                    puf_r = []
                    if puf_y:
                        puf_r = yield puf_y
                    _optim_step(
                        self._swarm, self._dp_m.up_hyper_params(puf_r), OptimStepArgs(
                            avc=avc,
                            dim=dim_list,
                            hyper_state=self._hyper_state,
                            swarm_extras=self._swarm_extras,
                            start=0,
                            stop=self._swarm.swarm_size
                        )
                    )
                    avc.slow_down_swarm(self._swarm, self._swarm_extras)
            _only_the_fittest_survive(
                1.0 - self._arg_con.survival_rate, self._swarm, self._swarm_extras
            )

            run_cont.inner_loop += run_cont.to_add
