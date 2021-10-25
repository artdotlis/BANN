# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import Generator, List, Tuple, final

from bann.b_hyper_optim.optimiser.fun_const.hyper_const import RandomOArgs
from bann.b_hyper_optim.optimiser.fun_const.hyper_fun import h_check_end_fitness, \
    h_create_random_params
from bann.b_hyper_optim.optimiser.fun_const.pso_const import HyperExtras


@final
class RandomCont:
    def __init__(self, arg_con: RandomOArgs, rand_extras: HyperExtras, /) -> None:
        super().__init__()
        self._random_extras = rand_extras
        self._arg_con = arg_con

    def update_random_extras(self, swarm_extras: HyperExtras, /) -> None:
        self._random_extras = swarm_extras

    def hyper_optim(self) \
            -> Generator[List[Tuple[float, ...]], List[Tuple[float, Tuple[float, ...]]], None]:
        run_cnt = 0
        fitness = float('inf')
        while run_cnt < self._arg_con.repeats and h_check_end_fitness(
                fitness, self._arg_con.end_fitness
        ):
            erg = [
                h_create_random_params(self._random_extras)
                for _ in range(self._arg_con.package)
            ]
            puf = yield erg
            for fit in puf:
                if fit[0] < fitness:
                    fitness = fit[0]
            run_cnt += 1
