# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Callable, final, Final, List, Pattern

from bann.b_container.functions.check_arg_complex import get_comma_set, check_pattern
from bann.b_test_train_prepare.pytorch.p_test.functions.steps import C_STEP_SIZE
from bann.b_test_train_prepare.pytorch.p_test.libs.class_test_lib import get_class_test_list, \
    get_class_test_list_comb, get_class_test_pattern_list
from bann.b_container.functions.dict_str_repr import dict_json_repr
from bann.b_container.errors.custom_erors import KnownTestStateError
from bann.b_container.states.framework.interface.test_state import TestState
from bann.b_container.states.framework.pytorch.p_test.p_test_general import PTestGenCon, \
    get_t_general_main_state_types


_ALL_TESTS: Final[List[str]] = get_class_test_list()
_ALL_TESTS_REGEX: Final[List[Pattern[str]]] = get_class_test_pattern_list()
_ALL_TESTS_COM: Final[List[str]] = get_class_test_list_comb()


@final
@dataclass
class _TestClassCon(PTestGenCon):
    class_num: int = 2
    step_cnt: int = 20
    tests: Tuple[str, ...] = tuple()


@final
class TesterClass(TestState[_TestClassCon]):

    def __init__(self) -> None:
        super().__init__()
        self.__kwargs: Optional[_TestClassCon] = None

    def get_kwargs_repr(self) -> str:
        return dict_json_repr(self.get_kwargs().__dict__, self.get_pre_arg())

    def set_kwargs(self, args_dict: Dict, /) -> None:
        if self.__kwargs is not None:
            raise KnownTestStateError("Kwargs already set!")
        self.__kwargs = super().parse_dict(_TestClassCon, args_dict)

    def get_kwargs(self) -> _TestClassCon:
        if self.__kwargs is None:
            raise KnownTestStateError("Kwargs not set!")
        return self.__kwargs


_TestStateTypes: Final[Dict[str, Tuple[Callable[[str], object], str]]] = {
    f'{TestState.get_pre_arg()}class_num': (
        lambda val: int(val) if int(val) >= 2 else 2,
        "int (>=2)"
    ),
    f'{TestState.get_pre_arg()}step_cnt': (
        lambda val: int(val) if 10 <= int(val) <= C_STEP_SIZE else 20,
        "int (10<=x<=10000)"
    ),
    f'{TestState.get_pre_arg()}tests': (
        lambda ar_v: get_comma_set(
            ar_v,
            lambda val: str(val)
            if val in _ALL_TESTS or check_pattern(val, _ALL_TESTS_REGEX) else _ALL_TESTS[0]
        ),
        f"str,... ({','.join(_ALL_TESTS_COM)})"
    )
}


def get_test_class_state_types() -> Dict[str, Tuple[Callable[[str], object], str]]:
    bann_gen_test = get_t_general_main_state_types()
    merged_dict = {**_TestStateTypes}
    for key, value in bann_gen_test.items():
        if key in merged_dict:
            raise KnownTestStateError(f"Duplicated key {key} in {TesterClass.__name__}")
        merged_dict[key] = value
    return merged_dict
