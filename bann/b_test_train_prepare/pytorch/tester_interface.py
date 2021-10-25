# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import abc
from dataclasses import dataclass
from typing import Tuple, final

from torch.utils.data import Dataset

from bann.b_container.states.framework.interface.test_state import TestState
from bann.b_frameworks.pytorch.net_model_interface import NetModelInterface
from pan.public.constants.net_tree_id_constants import ANNTreeIdType
from pan.public.constants.test_net_stats_constants import TestNNStatsElementType
from rewowr.public.interfaces.logger_interface import SyncStdoutInterface


@final
@dataclass
class TesterInterfaceArgs:
    module: NetModelInterface
    input_test: Tuple[Dataset, ...]
    id_file: ANNTreeIdType
    cuda: bool
    truth_fun_id: str


class BTesterInterface(abc.ABC):

    @abc.abstractmethod
    def test(self, sync_out: SyncStdoutInterface, args: TesterInterfaceArgs, /) \
            -> Tuple[TestNNStatsElementType, ...]:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def set_test_state(self, state: TestState, /) -> None:
        raise NotImplementedError('Interface!')
