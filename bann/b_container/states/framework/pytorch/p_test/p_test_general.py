# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import Dict, Optional, Callable, Tuple, Final, final

from dataclasses import dataclass

from bann.b_container.functions.check_arg_complex import get_comma_one_two_tuple
from bann.b_container.functions.dict_str_repr import dict_json_repr
from bann.b_frameworks.pytorch.act_fun_lib import get_framework_act_lib
from bann.b_pan_integration.framwork_key_lib import FrameworkKeyLib
from bann.b_container.states.framework.interface.test_state import TestStateKwargs, TestState
from bann.b_container.errors.custom_erors import KnownTestStateError

_FRAMEWORK: Final[str] = FrameworkKeyLib.PYTORCH.value
_MODE: Final[Tuple[str, ...]] = get_framework_act_lib(_FRAMEWORK).act_names_b()


@dataclass
class PTestGenCon(TestStateKwargs):
    last_layer: Tuple[str, Optional[int]] = (_MODE[0], 1)
    batch_size: int = 100
    num_workers: int = 0
    k_folds: int = 10
    n_repeats: int = 0


@final
@dataclass
class _PSimpleTestCon(PTestGenCon):
    print_results: bool = False


@final
class TesterGeneral(TestState[_PSimpleTestCon]):

    def __init__(self) -> None:
        super().__init__()
        self.__kwargs: Optional[_PSimpleTestCon] = None

    def get_kwargs_repr(self) -> str:
        return dict_json_repr(self.get_kwargs().__dict__, self.get_pre_arg())

    def set_kwargs(self, args_dict: Dict, /) -> None:
        if self.__kwargs is not None:
            raise KnownTestStateError("Kwargs already set!")
        self.__kwargs = super().parse_dict(_PSimpleTestCon, args_dict)

    def get_kwargs(self) -> _PSimpleTestCon:
        if self.__kwargs is None:
            raise KnownTestStateError("Kwargs not set!")
        return self.__kwargs


_TestStateTypes: Final[Dict[str, Tuple[Callable[[str], object], str]]] = {
    f'{TestState.get_pre_arg()}batch_size': (
        lambda val: int(val) if int(val) >= 1 else 100,
        "int (>=1)"
    ),
    f'{TestState.get_pre_arg()}num_workers': (
        lambda val: int(val) if int(val) >= 0 else 0,
        "int (>=0)"
    ),
    f'{TestState.get_pre_arg()}last_layer': (
        lambda val: get_comma_one_two_tuple(
            val, lambda val_m: str(val_m) if val_m in _MODE else _MODE[0], int
        ),
        f"str,[int] ({','.join(_MODE)}, optional dim)"
    ),
    f'{TestState.get_pre_arg()}k_folds': (
        lambda val: int(val) if int(val) >= 2 else 10,
        "int (>=2)"
    ),
    f'{TestState.get_pre_arg()}n_repeats': (
        lambda val: int(val) if int(val) >= 0 else 0,
        "int (>=0)"
    )
}


_SimpleTestStateTypes: Final[Dict[str, Tuple[Callable[[str], object], str]]] = {
    f'{TestState.get_pre_arg()}print_results': (
        lambda print_results: print_results == 'T', "T for True else False"
    )
}


def get_simple_test_state_types() -> Dict[str, Tuple[Callable[[str], object], str]]:
    bann_gen_test = get_t_general_main_state_types()
    merged_dict = {**_SimpleTestStateTypes}
    for key, value in bann_gen_test.items():
        if key in merged_dict:
            raise KnownTestStateError(f"Duplicated key {key} in {TesterGeneral.__name__}")
        merged_dict[key] = value
    return merged_dict


def get_t_general_main_state_types() -> Dict[str, Tuple[Callable[[str], object], str]]:
    return _TestStateTypes
