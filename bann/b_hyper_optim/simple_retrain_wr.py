# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import Optional, Generator, List, Tuple, Dict, final

from bann.b_hyper_optim.optimiser.fun_const.hyper_const import SimplyTrainOArgs
from bann.b_hyper_optim.trainer.simply_train_gen import SimplyTrainCont
from bann.b_container.functions.check_hyper_arguments import check_dicts_consistency
from bann.b_hyper_optim.fun_const_wr.hyper_fun import h_print_to_logger, h_create_flat_params, \
    h_map_tuple_to_dict, h_map_f_dict_to_tuple
from bann.b_hyper_optim.errors.custom_erors import KnownSimplyTrainError
from bann.b_container.states.general.hyper_optim.simply_retrain import SimplyTrainState
from bann.b_hyper_optim.hyper_optim_interface import HyperOptimInterface, \
    HyperOptimInterfaceArgs, HyperOptimReturnElem
from bann.b_container.states.general.interface.hyper_optim_state import OptimHyperState
from rewowr.public.functions.syncout_dep_functions import logger_print_to_console
from rewowr.public.interfaces.logger_interface import SyncStdoutInterface


_GenAT = Generator[
    List[Dict[str, HyperOptimReturnElem]],
    List[Tuple[float, Dict[str, HyperOptimReturnElem]]],
    None
]


@final
class SimplyTrainWr(HyperOptimInterface):

    def __init__(self) -> None:
        super().__init__()
        self.__hyper_state: Optional[OptimHyperState] = None

    @property
    def hyper_state(self) -> SimplyTrainState:
        if self.__hyper_state is None or not isinstance(self.__hyper_state, SimplyTrainState):
            raise KnownSimplyTrainError("Hyper state was not set properly!")
        return self.__hyper_state

    def set_hyper_state(self, state: OptimHyperState, /) -> None:
        if not isinstance(state, SimplyTrainState):
            raise KnownSimplyTrainError(
                f"Expected type {SimplyTrainState.__name__} got {type(state).__name__}"
            )
        self.__hyper_state = state

    def hyper_optim(self, sync_out: SyncStdoutInterface, args: HyperOptimInterfaceArgs, /) \
            -> _GenAT:
        check_dicts_consistency(args)
        h_print_to_logger("SimplyTrain", sync_out, self.hyper_state)
        flat_params = h_create_flat_params(args.hyper_args)
        simply_train_gen = SimplyTrainCont(
            SimplyTrainOArgs(
                repeats=self.hyper_state.get_kwargs().repeats,
                package=self.hyper_state.get_kwargs().package
            ),
            h_map_f_dict_to_tuple(args.hyper_args, flat_params)
        )
        for results_copy in simply_train_gen.simply_copy_params():
            logger_print_to_console(
                sync_out, f"Created {len(results_copy)} copies of hyper-params"
            )
            _ = yield [
                h_map_tuple_to_dict(args.state_type, param, flat_params)
                for param in results_copy
            ]
