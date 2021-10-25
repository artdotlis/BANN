# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import random
from typing import Optional, Union, Type, Tuple

from bann.b_hyper_optim.errors.custom_erors import KnownHyperError
from bann.b_hyper_optim.optimiser.fun_const.pso_const import HyperExtras


def h_check_end_fitness(fitness_current: float, fitness_max: Optional[float], /) -> float:
    if fitness_max is None:
        return True
    if fitness_current > fitness_max:
        return True
    return False


def h_randomise_value(min_v: float, max_v: float,
                      value_type: Union[Type[float], Type[int]], /) -> float:
    if min_v > max_v:
        max_v = min_v
    if value_type == int:
        return random.randint(int(min_v), int(max_v))
    if value_type == float:
        return random.uniform(min_v, max_v)
    raise KnownHyperError("Wrong type should never happen!")


def h_create_random_params(swarm_extras: HyperExtras, /) -> Tuple[float, ...]:
    random.seed()
    return tuple(
        h_randomise_value(
            swarm_extras.search_space_min[index], swarm_extras.search_space_max[index],
            space_type
        )
        for index, space_type in enumerate(swarm_extras.search_space_type)
    )


def h_create_v_max(swarm_extras: HyperExtras, /) -> Tuple[float, ...]:
    erg_list = tuple(
        swarm_extras.search_space_max[index] - swarm_extras.search_space_min[index]
        for index in range(len(swarm_extras.search_space_type))
    )
    for min_max_dif in erg_list:
        if min_max_dif < 0:
            raise KnownHyperError("The min_max_dif should not be negative!")

    return erg_list
