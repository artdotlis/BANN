# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import Iterable, Tuple, Optional, Dict

from bann.b_container.errors.custom_erors import KnownPrepareError
from bann.b_container.states.framework.general.prepare.no_prepare import NoPrepTState
from bann.b_container.states.framework.interface.prepare_state import PrepareState
from bann.b_test_train_prepare.pytorch.prepare_interface import PrepareInterfaceArgs
from bann.b_test_train_prepare.pytorch.prepare_interface import PrepareInterface

from pan.public.constants.train_net_stats_constants import TrainNNStatsElementType

from rewowr.public.interfaces.logger_interface import SyncStdoutInterface


class NoPrepare(PrepareInterface):

    def __init__(self) -> None:
        super().__init__()
        self.__fitness: Optional[Tuple[float, float]] = None
        self.__pr_state: Optional[NoPrepTState] = None
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
        yield from args.trainer.train(sync_out, args.trainer_args)
        self.__fitness = args.trainer.fitness
        self.__state_dict = args.trainer.tr_state_dict

    def set_prepare_state(self, state: PrepareState, /) -> None:
        if not isinstance(state, NoPrepTState):
            raise KnownPrepareError(
                f"Expected type {NoPrepTState.__name__} got {type(state).__name__}"
            )
        self.__pr_state = state
