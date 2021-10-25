# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import Dict, Tuple, List, final, Optional, Generator

from bann.b_container.functions.check_hyper_arguments import check_dicts_consistency
from bann.b_container.functions.dict_str_repr import dict_string_repr
from bann.b_container.states.general.hyper_optim.hyper_optim_ega import EGAHyperState
from bann.b_hyper_optim.errors.custom_erors import KnownEGAError
from bann.b_hyper_optim.fun_const_wr.hyper_const import FlatTupleT
from bann.b_hyper_optim.fun_const_wr.hyper_fun import h_print_to_logger, h_create_flat_params, \
    h_map_f_dict_to_tuple, h_create_hyper_space, h_map_tuple_to_dict, h_map_dict_to_tuple
from bann.b_hyper_optim.optimiser.ega import EGACont
from bann.b_hyper_optim.optimiser.fun_const.ga_const import EGAOptimArgs
from bann.b_container.states.general.interface.hyper_optim_state import OptimHyperState
from bann.b_hyper_optim.hyper_optim_interface import HyperOptimReturnElem, HyperOptimInterface, \
    HyperOptimInterfaceArgs

from rewowr.public.functions.syncout_dep_functions import logger_print_to_console
from rewowr.public.interfaces.logger_interface import SyncStdoutInterface

_GenAT = Generator[
    List[Dict[str, HyperOptimReturnElem]],
    List[Tuple[float, Dict[str, HyperOptimReturnElem]]],
    None
]


@final
class EGAHyperWr(HyperOptimInterface):

    def __init__(self) -> None:
        super().__init__()
        self.__hyper_state: Optional[OptimHyperState] = None

    @property
    def hyper_state(self) -> EGAHyperState:
        if self.__hyper_state is None or not isinstance(self.__hyper_state, EGAHyperState):
            raise KnownEGAError("Hyper state was not set properly!")
        return self.__hyper_state

    def set_hyper_state(self, state: OptimHyperState, /) -> None:
        if not isinstance(state, EGAHyperState):
            raise KnownEGAError(
                f"Expected type {EGAHyperState.__name__} got {type(state).__name__}"
            )
        self.__hyper_state = state

    def hyper_optim(self, sync_out: SyncStdoutInterface, args: HyperOptimInterfaceArgs, /) \
            -> _GenAT:
        check_dicts_consistency(args)
        h_print_to_logger("EGAHyper", sync_out, self.hyper_state)
        flat_params: FlatTupleT = h_create_flat_params(args.hyper_args)
        f_param: Optional[Tuple[float, ...]] = None
        if self.hyper_state.get_kwargs().include_first_run:
            f_param = h_map_f_dict_to_tuple(args.hyper_args, flat_params)
        ega = EGACont(
            EGAOptimArgs(
                population=self.hyper_state.get_kwargs().population,
                first_run=f_param, cross_over_r=self.hyper_state.get_kwargs().co_rate,
                mut_r=self.hyper_state.get_kwargs().m_rate,
                end_fitness=self.hyper_state.get_kwargs().end_fitness,
                repeats=self.hyper_state.get_kwargs().repeats,
                memory=self.hyper_state.get_kwargs().memory,
                memory_repeats=self.hyper_state.get_kwargs().memory_repeats
            ),
            h_create_hyper_space(
                args.hyper_max_args, args.hyper_min_args, args.min_max_types,
                flat_params
            )
        )
        gen = ega.hyper_optim()
        try:
            l_new_params: List[Tuple[float, ...]] = next(gen)
        except StopIteration:
            raise KnownEGAError("Generator could not be started!")
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
            ega.update_ga_extras(h_create_hyper_space(
                args.hyper_max_args, args.hyper_min_args, args.min_max_types,
                flat_params
            ))
            try:
                l_new_params = gen.send(
                    [(elem[0], h_map_dict_to_tuple(elem[1], flat_params)) for elem in puf]
                )
            except StopIteration:
                running = False
