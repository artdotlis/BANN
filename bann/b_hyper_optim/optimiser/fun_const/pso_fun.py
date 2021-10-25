# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import random

from bann.b_hyper_optim.optimiser.fun_const.pso_const import PSOHyperSt, SwarmPopulation


def pso_calc_new_vel(v_index: int, v_pos: int, particle_swarm: SwarmPopulation,
                     hyper_state: PSOHyperSt, /) -> float:
    random.seed()
    new_vel = \
        hyper_state.weight * particle_swarm.individual_vel[v_index][v_pos] \
        + hyper_state.c_2 * random.random() \
        * (particle_swarm.global_best_pos[v_pos]
           - particle_swarm.individual_pos[v_index][v_pos]) \
        + hyper_state.c_1 * random.random() \
        * (particle_swarm.individual_best_pos[v_index][v_pos]
           - particle_swarm.individual_pos[v_index][v_pos])
    return new_vel


def pso_calc_new_pos(ind_pos: int, dimension: int, particle_swarm: SwarmPopulation, /) -> float:
    return particle_swarm.individual_pos[ind_pos][dimension] \
           + particle_swarm.individual_vel[ind_pos][dimension]
