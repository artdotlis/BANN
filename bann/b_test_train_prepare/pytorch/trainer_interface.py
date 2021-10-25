# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import abc
from typing import Optional, Iterable, Tuple, final, Dict
from dataclasses import dataclass
from torch.utils.data import Dataset

from bann.b_frameworks.pytorch.net_model_interface import NetModelInterface
from bann.b_container.states.framework.interface.train_state import TrainState
from bann.b_container.states.framework.pytorch.criterion_param import CriterionAlgWr
from bann.b_container.states.framework.pytorch.lr_scheduler_param import LrSchAlgWr
from bann.b_container.states.framework.pytorch.optim_param import OptimAlgWr
from pan.public.constants.net_tree_id_constants import ANNTreeIdType
from pan.public.constants.train_net_stats_constants import TrainNNStatsElementType
from rewowr.public.interfaces.logger_interface import SyncStdoutInterface


@final
@dataclass
class TrainerInterfaceArgs:
    module: NetModelInterface
    input_train: Tuple[Dataset, ...]
    input_eval: Tuple[Dataset, ...]
    id_file: ANNTreeIdType
    dump: bool
    cuda: bool
    optimizer: Optional[OptimAlgWr]
    scheduler: Optional[LrSchAlgWr]
    criterion: Optional[CriterionAlgWr]
    truth_fun_id: str
    hyper_str: str


class TrainerInterface(abc.ABC):
    @property
    @abc.abstractmethod
    def tr_state_dict(self) -> Dict:
        raise NotImplementedError('Interface!')

    @property
    @abc.abstractmethod
    def fitness(self) -> Tuple[float, float]:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def train(self, sync_out: SyncStdoutInterface, args: TrainerInterfaceArgs, /) \
            -> Iterable[TrainNNStatsElementType]:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def set_train_state(self, state: TrainState, /) -> None:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def deactivate_log(self) -> None:
        raise NotImplementedError('Interface!')
