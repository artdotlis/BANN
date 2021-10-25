# -*- coding: utf-8 -*-
"""
Explained in paper:
    https://link.springer.com/chapter/10.1007/978-3-642-45111-9_1

    title:
        The Best Genetic Algorithm I
    author:
        Kuri-Morales, Angel and Aldana-Bobadilla, Edwin

.. moduleauthor:: Artur Lissin
"""
from copy import deepcopy
import random as rnd
from typing import Generator, List, Tuple, Dict, Iterable, final

from bann.b_hyper_optim.dynamic_programming.dp_matrix import DPMatrix
from bann.b_hyper_optim.optimiser.fun_const.ga_const import GAPopulation, EGAOptimArgs
from bann.b_hyper_optim.optimiser.fun_const.hyper_fun import h_randomise_value, \
    h_create_random_params, h_create_v_max, h_check_end_fitness
from bann.b_hyper_optim.optimiser.fun_const.pso_const import HyperExtras


def _corrupt_or_random_gene(gene: float, genome_i: int, mut_r: float,
                            ga_extras: HyperExtras, /) -> float:
    rnd.seed()
    if rnd.random() <= mut_r:
        return h_randomise_value(
            ga_extras.search_space_min[genome_i], ga_extras.search_space_max[genome_i],
            ga_extras.search_space_type[genome_i]
        )
    return gene


def _random_mut_pop(pop: GAPopulation, mut_r: float, ga_extras: HyperExtras, /) -> None:
    for pop_i in range(pop.pop_size):
        pop.individual[pop_i] = tuple(
            _corrupt_or_random_gene(pop.individual[pop_i][genome_i], genome_i, mut_r, ga_extras)
            for genome_i in range(pop.genome)
        )


def _pop_gen(limit: int, to_sorted: Dict[float, List[int]], /) -> Iterable[int]:
    cnt = 0
    for s_key in sorted(to_sorted.keys()):
        for ind_i in to_sorted[s_key]:
            if cnt < limit:
                yield ind_i
                cnt += 1


def _sort_pop(population: GAPopulation, /) -> None:
    to_sort_dict: Dict[float, List[int]] = {}
    for ind_i, ind_fit in enumerate(population.individual_fitness):
        to_sort_dict.setdefault(ind_fit, []).append(ind_i)
    population.fittest_id = [new_ind for new_ind in _pop_gen(population.pop_size, to_sort_dict)]


def _ga_init_population(pop_size: int, ga_extras: HyperExtras, /) -> GAPopulation:
    randomised_pop = [h_create_random_params(ga_extras) for _ in range(pop_size)]
    res = GAPopulation(
        fittest_id=list(range(pop_size)),
        individual_fitness=[float('inf') for _ in range(pop_size)],
        individual=deepcopy(randomised_pop),
        pop_size=pop_size, genome=len(randomised_pop[0]),
        v_max=h_create_v_max(ga_extras),
        global_fitness=float('inf')
    )
    return res


def _update_fitness(pop: GAPopulation, f_pop_fit: List[Tuple[float, Tuple[float, ...]]], /) \
        -> None:
    for ind_i, fit in enumerate(f_pop_fit):
        fitness = fit[0]
        if fitness < pop.global_fitness:
            pop.global_fitness = fitness
        pop.individual_fitness[ind_i] = fitness
        pop.individual[ind_i] = fit[1]


def _cross_over(pop: GAPopulation, ind_1: int, ind_2: int, pos: int, /) -> None:
    semi_l = int(pop.genome / 2)
    first_child = tuple(
        pop.individual[ind_1][genome_i]
        if pos < genome_i <= pos + semi_l
        else pop.individual[ind_2][genome_i]
        for genome_i in range(pop.genome)
    )
    second_child = tuple(
        pop.individual[ind_2][genome_i]
        if pos < genome_i <= pos + semi_l
        else pop.individual[ind_1][genome_i]
        for genome_i in range(pop.genome)
    )
    pop.individual[ind_1] = first_child
    pop.individual[ind_2] = second_child


def _deterministic_sel_annular_crossover(pop: GAPopulation, cr_prob: float, /) -> None:
    rnd.seed()
    pop_half = int(pop.pop_size / 2)
    for pop_i in range(pop_half):
        if rnd.random() <= cr_prob:
            n_p_cr = rnd.randint(0, int(pop.genome / 2))
            _cross_over(pop, pop_i, pop.pop_size - pop_i - 1, n_p_cr)


def _full_elitism(pop: GAPopulation, /) -> None:
    pop.individual = deepcopy([pop.individual[ind_i] for _ in range(2) for ind_i in pop.fittest_id])
    pop.individual_fitness = deepcopy([
        pop.individual_fitness[ind_i] for _ in range(2) for ind_i in pop.fittest_id
    ])


@final
class EGACont:
    def __init__(self, arg_con: EGAOptimArgs, ga_extras: HyperExtras, /) -> None:
        super().__init__()
        self._ga_extras = ga_extras
        self._arg_con = arg_con
        self._pop = _ga_init_population(self._arg_con.population, self._ga_extras)
        self._dp_m = DPMatrix(arg_con.memory, arg_con.memory_repeats)

    def update_ga_extras(self, ga_extras: HyperExtras, /) -> None:
        self._ga_extras = ga_extras

    def hyper_optim(self) \
            -> Generator[List[Tuple[float, ...]], List[Tuple[float, Tuple[float, ...]]], None]:
        run_cnt = 0
        if self._arg_con.first_run is not None:
            self._pop.individual[0] = self._arg_con.first_run
        while run_cnt < self._arg_con.repeats and h_check_end_fitness(
                self._pop.global_fitness, self._arg_con.end_fitness
        ):
            # eval pop
            puf_y = self._dp_m.cr_hyper_params(self._pop.individual[0:self._pop.pop_size])
            puf_r = []
            if puf_y:
                puf_r = yield puf_y
            _update_fitness(self._pop, self._dp_m.up_hyper_params(puf_r))
            # sort fitness
            if run_cnt >= 1:
                _sort_pop(self._pop)
            # duplicate
            _full_elitism(self._pop)
            # annular crossover
            _deterministic_sel_annular_crossover(self._pop, self._arg_con.cross_over_r)
            # mutate
            _random_mut_pop(self._pop, self._arg_con.mut_r, self._ga_extras)
            run_cnt += 1
