# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import final, Optional, Tuple

from bann.b_container.states.framework.interface.test_state import TestState
from bann.b_container.states.framework.pytorch.p_test.p_local_test import TesterLocal
from bann.b_frameworks.pytorch.interfaces.local_prep_train_test import LocalTestInterface, \
    LocalTesterArgs
from bann.b_test_train_prepare.errors.custom_errors import KnownTesterError
from bann.b_test_train_prepare.pytorch.tester_interface import BTesterInterface, \
    TesterInterfaceArgs

from pan.public.constants.test_net_stats_constants import TestNNStatsElementType

from rewowr.public.interfaces.logger_interface import SyncStdoutInterface


@final
class LocalTester(BTesterInterface):

    def __init__(self) -> None:
        super().__init__()
        self.__test_state: Optional[TesterLocal] = None

    @property
    def test_state(self) -> TesterLocal:
        if self.__test_state is None or not isinstance(self.__test_state, TesterLocal):
            raise KnownTesterError("Test state was not set properly!")
        return self.__test_state

    def set_test_state(self, state: TestState, /) -> None:
        if not isinstance(state, TesterLocal):
            raise KnownTesterError(
                f"Expected type {TesterLocal.__name__} got {type(state).__name__}"
            )
        self.__test_state = state

    def test(self, sync_out: SyncStdoutInterface, args: TesterInterfaceArgs, /) -> \
            Tuple[TestNNStatsElementType, ...]:
        module = args.module
        if not isinstance(module, LocalTestInterface):
            raise KnownTesterError(
                f"Expected {LocalTestInterface.__name__} got {type(module).__name__}"
            )
        module.te_print_to_logger(sync_out)
        return module.l_test(LocalTesterArgs(
            input_test=args.input_test, id_file=args.id_file, cuda=args.cuda,
            truth_fun_id=args.truth_fun_id
        ))
