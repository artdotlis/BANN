# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from copy import deepcopy
from typing import Optional, Tuple, Dict, Iterable

from bann.b_container.errors.custom_erors import KnownPrepareError
from bann.b_container.states.framework.general.prepare.local_prepare import LocalPrepTState
from bann.b_container.states.framework.interface.prepare_state import PrepareState
from bann.b_frameworks.pytorch.interfaces.local_prep_train_test import LocalPrepareInterface
from bann.b_test_train_prepare.pytorch.prepare_interface import PrepareInterface, \
    PrepareInterfaceArgs

from pan.public.constants.train_net_stats_constants import TrainNNStatsElementType

from rewowr.public.interfaces.logger_interface import SyncStdoutInterface


class LocalPrepare(PrepareInterface):

    def __init__(self) -> None:
        super().__init__()
        self.__fitness: Optional[Tuple[float, float]] = None
        self.__pr_state: Optional[LocalPrepTState] = None
        self.__state_dict: Optional[Dict] = None

    @property
    def p_state_dict(self) -> Dict:
        if self.__state_dict is None:
            raise KnownPrepareError("Prepare Trainig-run was not finished")
        return self.__state_dict

    @property
    def fitness(self) -> Tuple[float, float]:
        if self.__fitness is None:
            raise KnownPrepareError("The Trainer was not started!")
        return self.__fitness

    def run_train(self, sync_out: SyncStdoutInterface, args: PrepareInterfaceArgs, /) \
            -> Iterable[TrainNNStatsElementType]:
        self.__state_dict = None
        module = args.trainer_args.module
        if not isinstance(module, LocalPrepareInterface):
            raise KnownPrepareError(
                f"Expected {LocalPrepareInterface.__name__} got {type(module).__name__}"
            )
        module.pr_print_to_logger(sync_out)
        for new_tr_args in module.l_prepare(args.trainer_args):
            cp_tr = deepcopy(args.trainer)
            yield from cp_tr.train(sync_out, new_tr_args)
            if self.__fitness is None or self.__fitness[0] > cp_tr.fitness[0]:
                self.__fitness = cp_tr.fitness
                self.__state_dict = cp_tr.tr_state_dict

    def set_prepare_state(self, state: PrepareState, /) -> None:
        if not isinstance(state, LocalPrepTState):
            raise KnownPrepareError(
                f"Expected type {LocalPrepTState.__name__} got {type(state).__name__}"
            )
        self.__pr_state = state
