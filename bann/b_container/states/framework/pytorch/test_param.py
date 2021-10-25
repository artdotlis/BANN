# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from enum import Enum
from typing import Dict, Callable, Type, Tuple, List, final, Final
from dataclasses import dataclass

from bann.b_container.states.framework.pytorch.p_test.p_local_test import TesterLocal, \
    get_test_local_state_types
from bann.b_test_train_prepare.pytorch.p_test.local_test import LocalTester
from bann.b_container.states.framework.pytorch.p_test.p_test_reg import TesterReg, \
    get_test_reg_state_types
from bann.b_test_train_prepare.pytorch.p_test.reg_test import RegTester
from bann.b_container.constants.fr_string import FrStPName
from bann.b_test_train_prepare.pytorch.p_test.class_test import ClassTester
from bann.b_container.states.framework.pytorch.p_test.p_test_class import \
    get_test_class_state_types, TesterClass
from bann.b_container.states.framework.interface.test_state import TestState
from bann.b_container.states.framework.pytorch.p_test.p_test_general import TesterGeneral, \
    get_simple_test_state_types
from bann.b_container.errors.custom_erors import KnownTestStateError
from bann.b_test_train_prepare.pytorch.tester_interface import BTesterInterface
from bann.b_test_train_prepare.pytorch.p_test.simple_test import SimpleTester
from pan.public.interfaces.config_constants import ExtraArgsNet
from rewowr.public.functions.check_re_wr_wo_arguments import check_parse_type


@final
@dataclass
class _LibElem:
    state_types: Dict[str, Tuple[Callable[[str], object], str]]
    test_state: Type[TestState]


@final
class TestAlgWr:

    def __init__(self, tester_type: Type[BTesterInterface],
                 tester_state_type: Type[TestState],
                 tester_type_name: str,
                 tester_state: TestState, /) -> None:
        super().__init__()
        self.__tester_type: Type[BTesterInterface] = tester_type
        if not isinstance(tester_state, tester_state_type):
            raise KnownTestStateError(
                f"The expected tester type is {tester_state_type.__name__}"
                + f" got {type(tester_state).__name__}!"
            )
        self.__tester_state: TestState = tester_state
        self.__tester_state_type: Type[TestState] = tester_state_type
        self.__tester: BTesterInterface = self.tester_type()
        self.__tester.set_test_state(self.__tester_state)
        self.__tester_type_name = tester_type_name

    @staticmethod
    def param_name() -> str:
        return FrStPName.TE.value

    @property
    def tester_state_type(self) -> Type[TestState]:
        return self.__tester_state_type

    @property
    def test_state(self) -> TestState:
        return self.__tester_state

    @property
    def tester(self) -> BTesterInterface:
        return self.__tester

    @property
    def tester_type(self) -> Type[BTesterInterface]:
        return self.__tester_type

    @property
    def tester_type_name(self) -> str:
        return self.__tester_type_name


_TestAlg: Final[Dict[Type, Callable[[TestState], TestAlgWr]]] = {
    TesterGeneral:
        lambda state: TestAlgWr(SimpleTester, TesterGeneral, TestLibName.SIMPL.value, state),
    TesterClass: lambda state: TestAlgWr(ClassTester, TesterClass, TestLibName.CLASS.value, state),
    TesterReg: lambda state: TestAlgWr(RegTester, TesterReg, TestLibName.REG.value, state),
    TesterLocal: lambda state: TestAlgWr(LocalTester, TesterLocal, TestLibName.LOCAL.value, state)
}


@final
class TestLibName(Enum):
    SIMPL = 'SimpleTester'
    CLASS = 'ClassTester'
    REG = 'RegTester'
    LOCAL = 'LocalTester'


_TestLib: Final[Dict[str, _LibElem]] = {
    TestLibName.SIMPL.value: _LibElem(
        state_types=get_simple_test_state_types(), test_state=TesterGeneral
    ),
    TestLibName.CLASS.value: _LibElem(
        state_types=get_test_class_state_types(), test_state=TesterClass
    ),
    TestLibName.REG.value: _LibElem(
        state_types=get_test_reg_state_types(), test_state=TesterReg
    ),
    TestLibName.LOCAL.value: _LibElem(
        state_types=get_test_local_state_types(), test_state=TesterLocal
    )
}


def get_test_lib_keys() -> List[str]:
    return list(_TestLib.keys())


def get_test_state_params(state_id: str, /) -> Dict[str, str]:
    all_params = _TestLib.get(state_id, None)
    if all_params is None:
        return {}
    return {
        param_name: param_type[1]
        for param_name, param_type in all_params.state_types.items()
    }


def create_test_state(test_local: str, extra_args: ExtraArgsNet, /) -> TestState:
    tester = TestAlgWr.param_name()
    if tester in extra_args.arguments:
        tester = extra_args.arguments[tester]
    else:
        tester = test_local

    all_params = _TestLib.get(tester, None)
    if all_params is None:
        raise KnownTestStateError(f"The test algorithm {tester} is not defined!")

    set_params = {}
    for param_to_find, param_type in all_params.state_types.items():
        if param_to_find in extra_args.arguments:
            set_params[param_to_find] = check_parse_type(
                extra_args.arguments[param_to_find], param_type[0]
            )

    erg = all_params.test_state()
    erg.set_kwargs(set_params)
    return erg


def init_test_alg(test_state: TestState, /) -> TestAlgWr:
    test_alg_wr = _TestAlg.get(type(test_state), None)
    if test_alg_wr is None:
        raise KnownTestStateError(
            f"Could not find the test_type algorithm with the state {type(test_state).__name__}!"
        )
    return test_alg_wr(test_state)
