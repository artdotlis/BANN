# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import abc
from dataclasses import dataclass
from typing import Tuple, Iterable, final, Dict

from bann.b_container.states.framework.interface.prepare_state import PrepareState
from bann.b_test_train_prepare.pytorch.trainer_interface import TrainerInterfaceArgs, \
    TrainerInterface

from pan.public.constants.train_net_stats_constants import TrainNNStatsElementType

from rewowr.public.interfaces.logger_interface import SyncStdoutInterface


@final
@dataclass
class PrepareInterfaceArgs:
    trainer: TrainerInterface
    trainer_args: TrainerInterfaceArgs


class PrepareInterface(abc.ABC):
    @property
    @abc.abstractmethod
    def p_state_dict(self) -> Dict:
        raise NotImplementedError('Interface!')

    @property
    @abc.abstractmethod
    def fitness(self) -> Tuple[float, float]:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def run_train(self, sync_out: SyncStdoutInterface, args: PrepareInterfaceArgs, /) \
            -> Iterable[TrainNNStatsElementType]:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def set_prepare_state(self, state: PrepareState, /) -> None:
        raise NotImplementedError('Interface!')
