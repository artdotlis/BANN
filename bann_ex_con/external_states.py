# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import Dict, Type, Callable, Final

from bann.b_container.states.general.init_var_alg_wr import InitVarAlgWr
from bann.b_container.states.general.interface.init_state import InitLibElemCon, InitState
from bann.b_container.states.general.interface.net_state import NetLibElemCon

from bann_ex_con.external_states_enum import ENetLibName

from bann_demo.pytorch.states.net_complex import get_net_complex_state_types, NetComplexState
from bann_demo.pytorch.states.net_rbm import get_net_rbm_state_types, NetRBMState
from bann_demo.pytorch.states.net_ae import get_net_ae_state_types, NetAEState


_ENetLib: Final[Dict[str, NetLibElemCon]] = {
    # TODO fill if needed
    # demo
    ENetLibName.COMPLEX.value: NetLibElemCon(
        state_types=get_net_complex_state_types(), net_state=NetComplexState
    ),
    ENetLibName.RBM.value: NetLibElemCon(
        state_types=get_net_rbm_state_types(), net_state=NetRBMState
    ),
    ENetLibName.AE.value: NetLibElemCon(
        state_types=get_net_ae_state_types(), net_state=NetAEState
    ),
    # e_networks
}


def get_ex_net_lib() -> Dict[str, NetLibElemCon]:
    return _ENetLib


_EInitLib: Final[Dict[str, InitLibElemCon]] = {
    # TODO fill if needed
    # e_networks
}
_EInitAlg: Final[Dict[Type, Callable[[InitState], InitVarAlgWr]]] = {
    # TODO fill if needed
    # e_networks
}


def get_ex_init_lib() -> Dict[str, InitLibElemCon]:
    return _EInitLib


def get_ex_init_alg() -> Dict[Type, Callable[[InitState], InitVarAlgWr]]:
    return _EInitAlg
