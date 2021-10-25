# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from enum import Enum
from typing import Dict, Callable, Type, List, Final, final

from bann.b_container.states.general.init_var_alg_wr import InitVarAlgWr
from bann.b_container.states.general.interface.init_state import InitState, \
    InitLibElemCon
from bann.b_container.states.general.g_init.init_general import InitGeneralState, \
    get_init_general_state_types
from bann.b_container.errors.custom_erors import KnownInitStateError

from bann_ex_con.external_states import get_ex_init_lib, get_ex_init_alg

from pan.public.interfaces.config_constants import ExtraArgsNet
from rewowr.public.functions.check_re_wr_wo_arguments import check_parse_type

_InitAlg: Final[Dict[Type, Callable[[InitState], InitVarAlgWr]]] = {
    InitGeneralState: lambda state: InitVarAlgWr(InitGeneralState, state)
}


def _merged_init_alg() -> Dict[Type, Callable[[InitState], InitVarAlgWr]]:
    ext_lib = get_ex_init_alg()
    merged_dict = {**_InitAlg}
    for key, value in ext_lib.items():
        if key in merged_dict:
            raise KnownInitStateError(f"Init-alg duplicate: {key}")
        merged_dict[key] = value
    return merged_dict


@final
class InitLibName(Enum):
    GENERAL = 'General'


_InitLib: Final[Dict[str, InitLibElemCon]] = {
    InitLibName.GENERAL.value:
        InitLibElemCon(state_types=get_init_general_state_types(), init_state=InitGeneralState)
}


def _merged_init_lib() -> Dict[str, InitLibElemCon]:
    ext_lib = get_ex_init_lib()
    merged_dict = {**_InitLib}
    for key, value in ext_lib.items():
        if key in merged_dict:
            raise KnownInitStateError(f"Init State duplicate: {key}")
        merged_dict[key] = value
    return merged_dict


def get_init_lib_keys() -> List[str]:
    return list(_merged_init_lib().keys())


def get_init_state_params(state_id: str, /) -> Dict[str, str]:
    all_params = _merged_init_lib().get(state_id, None)
    if all_params is None:
        return {}
    return {
        param_name: param_type[1]
        for param_name, param_type in all_params.state_types.items()
    }


def create_init_state(init_type: str, extra_args: ExtraArgsNet, /) -> InitState:
    all_params = _merged_init_lib().get(init_type, None)
    if all_params is None:
        raise KnownInitStateError(f"The init state {init_type} is not defined!")
    set_params = {}
    for param_to_find, param_type in all_params.state_types.items():
        if param_to_find in extra_args.arguments:
            set_params[param_to_find] = check_parse_type(
                extra_args.arguments[param_to_find], param_type[0]
            )

    erg = all_params.init_state()
    erg.set_kwargs(set_params)
    return erg


def init_init_state(init_state: InitState, /) -> InitVarAlgWr:
    init_alg_wr = _merged_init_alg().get(type(init_state), None)
    if init_alg_wr is None:
        raise KnownInitStateError(
            f"Could not find the init_type algorithm with the state {type(init_state).__name__}!"
        )
    return init_alg_wr(init_state)
