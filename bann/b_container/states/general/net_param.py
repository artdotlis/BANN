# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from enum import Enum
from typing import Dict, List, final, Final

from bann.b_container.states.general.interface.net_state import NetState, NetLibElemCon
from bann.b_container.states.general.net.net_general import get_net_general_state_types, \
    NetGeneralState
from bann.b_container.errors.custom_erors import KnownNetStateError
from bann_ex_con.external_states import get_ex_net_lib

from pan.public.interfaces.config_constants import ExtraArgsNet
from rewowr.public.functions.check_re_wr_wo_arguments import check_parse_type


@final
class NetLibName(Enum):
    GENERAL = 'General'


_NetLib: Final[Dict[str, NetLibElemCon]] = {
    NetLibName.GENERAL.value:
        NetLibElemCon(state_types=get_net_general_state_types(), net_state=NetGeneralState)
}


def _merged_net_lib() -> Dict[str, NetLibElemCon]:
    ext_lib = get_ex_net_lib()
    merged_dict = {**_NetLib}
    for key, value in ext_lib.items():
        if key in merged_dict:
            raise KnownNetStateError(f"Net State duplicate: {key}")
        merged_dict[key] = value
    return merged_dict


def get_net_lib_keys() -> List[str]:
    return list(_merged_net_lib().keys())


def get_net_state_params(state_id: str, /) -> Dict[str, str]:
    all_params = _merged_net_lib().get(state_id, None)
    if all_params is None:
        return {}
    return {
        param_name: param_type[1]
        for param_name, param_type in all_params.state_types.items()
    }


def create_net_state(net_type: str, extra_args: ExtraArgsNet, /) -> NetState:
    all_params = _merged_net_lib().get(net_type, None)
    if all_params is None:
        raise KnownNetStateError(f"The net state {net_type} is not defined!")
    set_params = {}
    for param_to_find, param_type in all_params.state_types.items():
        if param_to_find in extra_args.arguments:
            set_params[param_to_find] = check_parse_type(
                extra_args.arguments[param_to_find], param_type[0]
            )

    erg = all_params.net_state()
    erg.set_kwargs(set_params)
    return erg
