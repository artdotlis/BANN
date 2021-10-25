# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Optional, Dict, List, final, Final, Callable

from bann.b_container.constants.gen_strings import GenStPaName
from bann.b_container.states.general.interface.hyper_optim_state import OptimHyperState
from bann.b_container.states.general.optim_hyper_param import HyperAlgWr, init_hyper_state, \
    create_hyper_state, get_hyper_state_params, get_hyper_lib_keys
from bann.b_container.states.general.init_param import init_init_state, create_init_state, \
    get_init_state_params, get_init_lib_keys
from bann.b_container.states.general.init_var_alg_wr import InitVarAlgWr
from bann.b_container.states.general.interface.init_state import InitState
from bann.b_container.states.general.interface.net_state import NetState
from bann.b_container.states.general.net_param import create_net_state, get_net_state_params, \
    get_net_lib_keys
from pan.public.interfaces.config_constants import ExtraArgsNet


_HyperInitT: Final = Callable[[OptimHyperState], HyperAlgWr]
_HyperParamT: Final = Callable[[ExtraArgsNet], Optional[OptimHyperState]]
_InitInitT: Final = Callable[[InitState], InitVarAlgWr]
_InitParamT: Final = Callable[[str, ExtraArgsNet], InitState]
_NetParamT: Final = Callable[[str, ExtraArgsNet], NetState]

# --------------------------------------------------------------------------------------------------

_LibSateGeneralType: Final = Callable[[str], Dict[str, str]]

# --------------------------------------------------------------------------------------------------


@final
@dataclass
class _GeneralContainer:
    hyper_init: _HyperInitT
    hyper_param: _HyperParamT
    init_init: _InitInitT
    init_param: _InitParamT
    net_param: _NetParamT


_GENERAL_LIB: Final[_GeneralContainer] = _GeneralContainer(
    hyper_init=init_hyper_state,
    hyper_param=create_hyper_state,
    init_init=init_init_state,
    init_param=create_init_state,
    net_param=create_net_state
)


@final
@dataclass
class _GeneralStateContainer:
    hyper_param: _LibSateGeneralType
    init_param: _LibSateGeneralType
    net_param: _LibSateGeneralType


_GENERAL_STATE_LIB: Final[_GeneralStateContainer] = _GeneralStateContainer(
    hyper_param=get_hyper_state_params,
    init_param=get_init_state_params,
    net_param=get_net_state_params
)


@final
@dataclass
class _GeneralStateNameContainer:
    hyper_param: List[str]
    init_param: List[str]
    net_param: List[str]


_GENERAL_STATE_NAME_LIB: Final[_GeneralStateNameContainer] = _GeneralStateNameContainer(
    hyper_param=get_hyper_lib_keys(),
    init_param=get_init_lib_keys(),
    net_param=get_net_lib_keys()
)


def get_general_lib() -> _GeneralContainer:
    return _GENERAL_LIB


def get_general_states_param_names(param_name: str, param_value: str, /) -> Dict[str, str]:
    general_lib = _GENERAL_STATE_LIB
    if param_name == GenStPaName.HYPER.value:
        return general_lib.hyper_param(param_value)
    if param_name == GenStPaName.INIT.value:
        return general_lib.init_param(param_value)
    if param_name == GenStPaName.NET.value:
        return general_lib.net_param(param_value)
    return {}


def get_general_states() -> Dict[str, str]:
    return _GENERAL_STATE_NAME_LIB.__dict__
