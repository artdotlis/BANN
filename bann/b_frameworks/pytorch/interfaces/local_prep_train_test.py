# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import abc
from dataclasses import dataclass
from typing import Iterable, Tuple, final, Dict

from torch.utils.data import Dataset

from bann.b_test_train_prepare.pytorch.trainer_interface import TrainerInterfaceArgs

from pan.public.constants.net_tree_id_constants import ANNTreeIdType
from pan.public.constants.test_net_stats_constants import TestNNStatsElementType
from pan.public.constants.train_net_stats_constants import TrainNNStatsElementType

from rewowr.public.interfaces.logger_interface import SyncStdoutInterface


class LocalPrepareInterface(abc.ABC):
    @abc.abstractmethod
    def l_prepare(self, trainer_args: TrainerInterfaceArgs, /) -> Iterable[TrainerInterfaceArgs]:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def pr_print_to_logger(self, sync_out: SyncStdoutInterface, /) -> None:
        raise NotImplementedError('Interface!')


@final
@dataclass
class LocalTesterArgs:
    input_test: Tuple[Dataset, ...]
    id_file: ANNTreeIdType
    cuda: bool
    truth_fun_id: str


class LocalTestInterface(abc.ABC):
    @abc.abstractmethod
    def l_test(self, args: LocalTesterArgs, /) -> Tuple[TestNNStatsElementType, ...]:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def te_print_to_logger(self, sync_out: SyncStdoutInterface, /) -> None:
        raise NotImplementedError('Interface!')


@final
@dataclass
class LocalTrainerArgs:
    input_train: Tuple[Dataset, ...]
    input_eval: Tuple[Dataset, ...]
    id_file: ANNTreeIdType
    dump: bool
    cuda: bool
    truth_fun_id: str
    hyper_str: str


class LocalTrainInterface(abc.ABC):
    @abc.abstractmethod
    def l_train(self, args: LocalTrainerArgs, /) -> Iterable[TrainNNStatsElementType]:
        raise NotImplementedError('Interface!')

    @property
    @abc.abstractmethod
    def l_fitness(self) -> Tuple[float, float]:
        raise NotImplementedError('Interface!')

    @property
    @abc.abstractmethod
    def l_state_dict(self) -> Dict:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def tr_print_to_logger(self, sync_out: SyncStdoutInterface, /) -> None:
        raise NotImplementedError('Interface!')
