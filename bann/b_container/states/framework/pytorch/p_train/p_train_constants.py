# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Tuple, Dict, Callable, Final, Optional

from bann.b_container.functions.check_arg_complex import get_comma_one_two_tuple, \
    check_arg_tuple_single
from bann.b_container.states.framework.interface.train_state import TrainStateKwargs, TrainState
from bann.b_pan_integration.framwork_key_lib import FrameworkKeyLib
from bann.b_frameworks.pytorch.act_fun_lib import get_framework_act_lib

_FRAMEWORK: Final[str] = FrameworkKeyLib.PYTORCH.value
_MODE: Final[Tuple[str, ...]] = get_framework_act_lib(_FRAMEWORK).act_names_b()


@dataclass
class TrainGenCon(TrainStateKwargs):
    torch_thread: int = 0
    over_fit: float = 0
    eval_ll: Tuple[str, Optional[int]] = (_MODE[0], 1)
    train_ll: Tuple[str, Optional[int]] = (_MODE[0], 1)
    report_size: int = 100
    batch_size: Tuple[int, ...] = (100,)
    epoch_size: int = 100
    shuffle: bool = True
    num_workers: int = 0
    drop_last: bool = False
    end_criterion: float = 1e-8
    plot_data: bool = True
    write_data: bool = True
    # Max Min settings
    max_epoch_size: int = 250000
    max_batch_size: int = 250000
    min_batch_size: int = 10
    min_epoch_size: int = 4


_TrainStateTypes: Final[Dict[str, Tuple[Callable[[str], object], str]]] = {
    f'{TrainState.get_pre_arg()}plot_data': (
        lambda val: val == 'T',
        "True if T else False"
    ),
    f'{TrainState.get_pre_arg()}write_data': (
        lambda val: val == 'T',
        "True if T else False"
    ),
    f'{TrainState.get_pre_arg()}torch_thread': (
        lambda val: int(val) if int(val) >= 0 else 0,
        "int (>=0)"
    ),
    f'{TrainState.get_pre_arg()}over_fit': (
        lambda val: float(val) if float(val) >= 0.0 else 0.0,
        "float (>=0)"
    ),
    f'{TrainState.get_pre_arg()}report_size': (
        lambda val: int(val) if int(val) >= 1 else 100,
        "int (>=1)"
    ),
    f'{TrainState.get_pre_arg()}batch_size': (lambda val: check_arg_tuple_single(
        val, lambda val1: int(val1) if int(val1) >= 1 else 100
    ), "Tuple[int (>=1), ...] as str,..."),
    f'{TrainState.get_pre_arg()}epoch_size': (
        lambda val: int(val) if int(val) >= 1 else 100,
        "int (>=1)"
    ),
    f'{TrainState.get_pre_arg()}shuffle': (
        lambda val: val == 'T',
        "True if T else False"
    ),
    f'{TrainState.get_pre_arg()}num_workers': (
        lambda val: int(val) if int(val) >= 0 else 0,
        "int (>=0)"
    ),
    f'{TrainState.get_pre_arg()}drop_last': (
        lambda val: val == 'T',
        "True if T else False"
    ),
    f'{TrainState.get_pre_arg()}end_criterion': (
        lambda val: float(val) if float(val) > 0 else 1e-8,
        "float (>0)"
    ),
    f'{TrainState.get_pre_arg()}max_epoch_size': (
        lambda val: int(val) if int(val) >= 100 else 100,
        "int (>=100)"
    ),
    f'{TrainState.get_pre_arg()}min_epoch_size': (
        lambda val: int(val) if int(val) >= 4 else 10,
        "int (>=4)"
    ),
    f'{TrainState.get_pre_arg()}max_batch_size': (
        lambda val: int(val) if int(val) >= 10 else 10,
        "int (>=10)"
    ),
    f'{TrainState.get_pre_arg()}min_batch_size': (
        lambda val: int(val) if int(val) >= 1 else 10,
        "int (>=1)"
    ),
    f'{TrainState.get_pre_arg()}eval_ll': (
        lambda val: get_comma_one_two_tuple(
            val, lambda val_m: str(val_m) if val_m in _MODE else _MODE[0], int
        ),
        f"str,[int] ({','.join(_MODE)}, optional dim)"
    ),
    f'{TrainState.get_pre_arg()}train_ll': (
        lambda val: get_comma_one_two_tuple(
            val, lambda val_m: str(val_m) if val_m in _MODE else _MODE[0], int
        ),
        f"str,[int] ({','.join(_MODE)}, optional dim)"
    )
}


def get_train_general_state_types() -> Dict[str, Tuple[Callable[[str], object], str]]:
    return _TrainStateTypes
