# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from copy import deepcopy
from typing import final, Optional, Tuple, Dict, Iterable

from bann.b_container.states.framework.interface.train_state import TrainState
from bann.b_container.states.framework.pytorch.p_train.p_local_train import LocalTState
from bann.b_frameworks.pytorch.interfaces.local_prep_train_test import LocalTrainInterface, \
    LocalTrainerArgs
from bann.b_test_train_prepare.errors.custom_errors import KnownTrainerError
from bann.b_test_train_prepare.pytorch.trainer_interface import TrainerInterface, \
    TrainerInterfaceArgs

from pan.public.constants.train_net_stats_constants import TrainNNStatsElementType

from rewowr.public.interfaces.logger_interface import SyncStdoutInterface


@final
class LocalTrainer(TrainerInterface):

    def __init__(self) -> None:
        super().__init__()
        self.__fitness: Optional[Tuple[float, float]] = None
        self.__train_state: Optional[LocalTState] = None
        self.__log = True
        self.__state_dict: Optional[Dict] = None

    @property
    def tr_state_dict(self) -> Dict:
        if self.__state_dict is None:
            raise KnownTrainerError("Training was not finished")
        return self.__state_dict

    def deactivate_log(self) -> None:
        self.__log = False

    @property
    def train_state(self) -> LocalTState:
        if self.__train_state is None or not isinstance(self.__train_state, LocalTState):
            raise KnownTrainerError("Train state was not set properly!")
        return self.__train_state

    def set_train_state(self, state: TrainState, /) -> None:
        if not isinstance(state, LocalTState):
            raise KnownTrainerError(
                f"Expected type {LocalTState.__name__} got {type(state).__name__}"
            )
        self.__train_state = state

    @property
    def fitness(self) -> Tuple[float, float]:
        if self.__fitness is None:
            raise KnownTrainerError("The Trainer was not started!")
        return self.__fitness

    def train(self, sync_out: SyncStdoutInterface,
              args: TrainerInterfaceArgs, /) -> Iterable[TrainNNStatsElementType]:
        self.__state_dict = None
        module = args.module
        if not isinstance(module, LocalTrainInterface):
            raise KnownTrainerError(
                f"Expected {LocalTrainInterface.__name__} got {type(module).__name__}"
            )
        if not (args.optimizer is None and args.criterion is None and args.scheduler is None):
            raise KnownTrainerError("Optimizer, scheduler or criterion is not None")
        if self.__log:
            module.tr_print_to_logger(sync_out)
        yield from module.l_train(LocalTrainerArgs(
            input_train=args.input_train, input_eval=args.input_eval, id_file=args.id_file,
            dump=args.dump, cuda=args.cuda, truth_fun_id=args.truth_fun_id,
            hyper_str=args.hyper_str
        ))
        self.__state_dict = deepcopy(module.l_state_dict)
        self.__fitness = module.l_fitness
