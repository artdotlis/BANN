# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Optional, Union, Dict, List, final, Final, Tuple, Callable

from bann.b_container.constants.fr_string import FrStPName
from bann.b_container.states.framework.interface.prepare_state import PrepareState
from bann.b_container.states.framework.pytorch.prepare_param import PrepareAlgWr, \
    init_prepare_state, create_prepare_state, get_prepare_state_params, get_prepare_lib_keys
from bann.b_container.states.framework.interface.lr_scheduler import LRSchedulerState
from bann.b_container.states.framework.interface.optim_state import MainOptimSt
from bann.b_container.states.framework.interface.train_state import TrainState
from bann.b_container.states.framework.pytorch.lr_scheduler_param import LrSchAlgWr, \
    init_lr_sch_alg, create_lr_sch_state, get_lr_sch_state_params, get_lr_sch_lib_keys
from bann.b_container.states.framework.pytorch.optim_param import OptimAlgWr, init_optim_alg, \
    create_optim_state, get_optim_state_params, get_optim_lib_keys
from bann.b_container.states.framework.pytorch.train_param import TrainAlgWr, init_train_alg, \
    create_train_state, get_train_state_params, get_train_lib_keys
from bann.b_pan_integration.framwork_key_lib import FrameworkKeyLib
from bann.b_container.errors.custom_erors import KnownFrameworkError
from bann.b_container.states.framework.pytorch.criterion_param import CriterionAlgWr, \
    create_criterion_state, init_criterion_alg, get_criterion_state_params, get_criterion_lib_keys
from bann.b_container.states.framework.interface.criterion_state import CriterionState
from bann.b_container.states.framework.interface.test_state import TestState
from bann.b_container.states.framework.pytorch.test_param import init_test_alg, create_test_state, \
    TestAlgWr, get_test_state_params, get_test_lib_keys
from pan.public.interfaces.config_constants import ExtraArgsNet

_LrSchAlgWrType = Union[
    LrSchAlgWr
]
_OptimWrType = Union[
    OptimAlgWr
]
_TrainWrType = Union[
    TrainAlgWr
]
_TestWrType = Union[
    TestAlgWr
]
_CriterionWrType = Union[
    CriterionAlgWr
]
_PrepareWrType = Union[
    PrepareAlgWr
]

# --------------------------------------------------------------------------------------------------

_LrSchedulerInitT: Final = Callable[[Tuple[LRSchedulerState, ...]], Optional[_LrSchAlgWrType]]
_LrSchedulerParamT: Final = Callable[[ExtraArgsNet], Tuple[LRSchedulerState, ...]]
_OptimInitT: Final = Callable[[Tuple[MainOptimSt, ...]], Optional[_OptimWrType]]
_OptimParamT: Final = Callable[[ExtraArgsNet], Tuple[MainOptimSt, ...]]
_CriterionInitT: Final = Callable[[CriterionState], _CriterionWrType]
_CriterionParamT: Final = Callable[[ExtraArgsNet], Optional[CriterionState]]
_TrainInitT: Final = Callable[[TrainState], _TrainWrType]
_TrainParamT: Final = Callable[[str, ExtraArgsNet], TrainState]
_PrepareInitT: Final = Callable[[PrepareState], _PrepareWrType]
_PrepareParamT: Final = Callable[[str, ExtraArgsNet], PrepareState]
_TestInitT: Final = Callable[[TestState], _TestWrType]
_TestParamT: Final = Callable[[str, ExtraArgsNet], TestState]

# --------------------------------------------------------------------------------------------------

_LibSateType: Final = Callable[[str], Dict[str, str]]

# --------------------------------------------------------------------------------------------------


@final
@dataclass
class _FrameWorkContainer:
    lr_scheduler_init: _LrSchedulerInitT
    lr_scheduler_param: _LrSchedulerParamT
    optim_init: _OptimInitT
    optim_param: _OptimParamT
    criterion_init: _CriterionInitT
    criterion_param: _CriterionParamT
    train_init: _TrainInitT
    train_param: _TrainParamT
    test_init: _TestInitT
    test_param: _TestParamT
    prepare_init: _PrepareInitT
    prepare_param: _PrepareParamT


@final
@dataclass
class _FrameWorkStateContainer:
    lr_scheduler_param: _LibSateType
    optim_param: _LibSateType
    criterion_param: _LibSateType
    train_param: _LibSateType
    test_param: _LibSateType
    prepare_param: _LibSateType


_FrameWorkLib: Final[Dict[str, _FrameWorkContainer]] = {
    FrameworkKeyLib.PYTORCH.value: _FrameWorkContainer(
        lr_scheduler_init=init_lr_sch_alg,
        lr_scheduler_param=create_lr_sch_state,
        optim_init=init_optim_alg,
        optim_param=create_optim_state,
        train_init=init_train_alg,
        train_param=create_train_state,
        test_init=init_test_alg,
        test_param=create_test_state,
        criterion_init=init_criterion_alg,
        criterion_param=create_criterion_state,
        prepare_init=init_prepare_state,
        prepare_param=create_prepare_state
    )
}

_FrameWorkStateLib: Final[Dict[str, _FrameWorkStateContainer]] = {
    FrameworkKeyLib.PYTORCH.value: _FrameWorkStateContainer(
        lr_scheduler_param=get_lr_sch_state_params,
        optim_param=get_optim_state_params,
        train_param=get_train_state_params,
        test_param=get_test_state_params,
        criterion_param=get_criterion_state_params,
        prepare_param=get_prepare_state_params
    )
}


def get_framework_lib(framework: str, /) -> _FrameWorkContainer:
    framework_fun = _FrameWorkLib.get(framework, None)
    if framework_fun is None:
        raise KnownFrameworkError(f"The framework {framework} is not defined!")
    return framework_fun


def get_framework_states_param_names(framework: str, param_name: str,
                                     param_value: str, /) -> Dict[str, str]:
    framework_fun = _FrameWorkStateLib.get(framework, None)
    if framework_fun is not None:
        if param_name == FrStPName.LRSCH.value:
            return framework_fun.lr_scheduler_param(param_value)
        if param_name == FrStPName.OPTIM.value:
            return framework_fun.optim_param(param_value)
        if param_name == FrStPName.CRIT.value:
            return framework_fun.criterion_param(param_value)
        if param_name == FrStPName.TR.value:
            return framework_fun.train_param(param_value)
        if param_name == FrStPName.TE.value:
            return framework_fun.test_param(param_value)
        if param_name == FrStPName.PR.value:
            return framework_fun.prepare_param(param_value)
    return {}


def get_all_framework_names() -> List[str]:
    return list(_FrameWorkLib.keys())


@final
@dataclass
class _FrameWorkStateNameContainer:
    lr_scheduler_param: List[str]
    optim_param: List[str]
    criterion_param: List[str]
    train_param: List[str]
    test_param: List[str]
    prepare_param: List[str]


_FrameWorkStateNameLib: Final[Dict[str, _FrameWorkStateNameContainer]] = {
    FrameworkKeyLib.PYTORCH.value: _FrameWorkStateNameContainer(
        lr_scheduler_param=get_lr_sch_lib_keys(),
        optim_param=get_optim_lib_keys(),
        criterion_param=get_criterion_lib_keys(),
        train_param=get_train_lib_keys(),
        test_param=get_test_lib_keys(),
        prepare_param=get_prepare_lib_keys()
    )
}


def get_framework_states() -> Dict[str, _FrameWorkStateNameContainer]:
    return _FrameWorkStateNameLib
