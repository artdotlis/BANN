# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import Optional, Generator, Dict, List, Tuple, final

from bann.b_container.functions.dict_str_repr import dict_string_repr
from bann.b_hyper_optim.optimiser.fun_const.pso_const import DaPsoOptimArgs, PSOHyperSt
from bann.b_hyper_optim.fun_const_wr.hyper_const import FlatTupleT
from bann.b_hyper_optim.fun_const_wr.hyper_fun import h_create_flat_params, \
    h_map_f_dict_to_tuple, h_create_hyper_space, h_map_tuple_to_dict, h_map_dict_to_tuple, \
    h_print_to_logger
from bann.b_hyper_optim.optimiser.da_pso import DAPSOCont
from bann.b_container.functions.check_hyper_arguments import check_dicts_consistency
from bann.b_container.states.general.hyper_optim.hyper_optim_pso import DAPSOHyperState
from bann.b_hyper_optim.errors.custom_erors import KnownPSOError
from bann.b_container.states.general.interface.hyper_optim_state import OptimHyperState
from bann.b_hyper_optim.hyper_optim_interface import \
    HyperOptimInterfaceArgs, HyperOptimReturnElem, HyperOptimInterface
from rewowr.public.functions.syncout_dep_functions import logger_print_to_console
from rewowr.public.interfaces.logger_interface import SyncStdoutInterface


_GenOutA = Generator[
    List[Dict[str, HyperOptimReturnElem]],
    List[Tuple[float, Dict[str, HyperOptimReturnElem]]],
    None
]


@final
class DAPSOHyperWr(HyperOptimInterface):
    def __init__(self) -> None:
        super().__init__()
        self.__hyper_state: Optional[OptimHyperState] = None

    @property
    def hyper_state(self) -> DAPSOHyperState:
        if self.__hyper_state is None or not isinstance(self.__hyper_state, DAPSOHyperState):
            raise KnownPSOError("Hyper state was not set properly!")
        return self.__hyper_state

    def set_hyper_state(self, state: OptimHyperState, /) -> None:
        if not isinstance(state, DAPSOHyperState):
            raise KnownPSOError(
                f"Expected type {DAPSOHyperState.__name__} got {type(state).__name__}"
            )
        self.__hyper_state = state

    def hyper_optim(self, sync_out: SyncStdoutInterface, args: HyperOptimInterfaceArgs, /) \
            -> _GenOutA:
        check_dicts_consistency(args)
        h_print_to_logger("DAPSOHyper", sync_out, self.hyper_state)
        flat_params: FlatTupleT = h_create_flat_params(args.hyper_args)
        if flat_params.sum_el <= 1:
            logger_print_to_console(
                sync_out, "Warning this algorithm should be used with more than one parameter!"
            )

        f_param: Optional[Tuple[float, ...]] = None
        if self.hyper_state.get_kwargs().include_first_run:
            f_param = h_map_f_dict_to_tuple(args.hyper_args, flat_params)
        da_pso = DAPSOCont(
            DaPsoOptimArgs(
                l_repeats=self.hyper_state.get_kwargs().l_repeats,
                first_run=f_param,
                swarm_size=self.hyper_state.get_kwargs().swarm,
                end_fitness=self.hyper_state.get_kwargs().end_fitness,
                repeats=self.hyper_state.get_kwargs().repeats,
                alpha=self.hyper_state.get_kwargs().alpha,
                s_prob=self.hyper_state.get_kwargs().speed_prob,
                survival_rate=self.hyper_state.get_kwargs().survival_rate,
                dim_prob=self.hyper_state.get_kwargs().dim_prob,
                memory=self.hyper_state.get_kwargs().memory,
                memory_repeats=self.hyper_state.get_kwargs().memory_repeats,
                l_increase=self.hyper_state.get_kwargs().l_increase
            ),
            h_create_hyper_space(
                args.hyper_max_args, args.hyper_min_args, args.min_max_types,
                flat_params
            ),
            PSOHyperSt(
                c_1=self.hyper_state.get_kwargs().c_1,
                c_2=self.hyper_state.get_kwargs().c_2,
                weight=self.hyper_state.get_kwargs().weight
            )
        )
        gen = da_pso.hyper_optim()
        try:
            l_new_params: List[Tuple[float, ...]] = next(gen)
        except StopIteration:
            raise KnownPSOError("Generator could not be started!")
        running = True
        while running:
            erg = [
                h_map_tuple_to_dict(args.state_type, param, flat_params)
                for param in l_new_params
            ]
            for param in erg:
                logger_print_to_console(
                    sync_out, f"Created new hyper-params:\n\t{dict_string_repr(param)}"
                )
            puf = yield erg
            da_pso.update_swarm_extras(h_create_hyper_space(
                args.hyper_max_args, args.hyper_min_args, args.min_max_types,
                flat_params
            ))
            try:
                l_new_params = gen.send(
                    [(elem[0], h_map_dict_to_tuple(elem[1], flat_params)) for elem in puf]
                )
            except StopIteration:
                running = False
