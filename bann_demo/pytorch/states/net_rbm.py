# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Callable, List, Final, final

from bann.b_container.errors.custom_erors import KnownNetStateError
from bann.b_container.states.general.interface.net_state import NetState
from bann.b_container.states.general.net.net_general import NetGeneralStateArgs, NetGeneralState, \
    get_net_general_state_types

from bann_demo.pytorch.networks.libs.rbm_lib import get_rbm_list

_MODE: Final[List[str]] = get_rbm_list()


@final
@dataclass
class _NetRBMStateArgs(NetGeneralStateArgs):
    g_sampling: int = 2
    hidden_cnt: int = 10
    rbm: str = _MODE[0]


@final
class NetRBMState(NetGeneralState):
    def __init__(self) -> None:
        super().__init__()
        self.__kwargs: Optional[_NetRBMStateArgs] = None

    def get_kwargs(self) -> _NetRBMStateArgs:
        if self.__kwargs is None:
            raise KnownNetStateError("Kwargs not set!")
        return self.__kwargs

    def set_kwargs(self, args_dict: Dict, /) -> None:
        if self.__kwargs is not None:
            raise KnownNetStateError("Kwargs already set!")
        self.__kwargs = super().parse_dict(_NetRBMStateArgs, args_dict)


_NetRBMStateTypes: Final[Dict[str, Tuple[Callable[[str], object], str]]] = {
    f'{NetState.get_pre_arg()}g_sampling': (
        lambda val: int(val) if int(val) >= 2 else 2,
        "int (>=2)"
    ),
    f'{NetState.get_pre_arg()}rbm': (
        lambda val: str(val) if val in _MODE else _MODE[0],
        f"str ({','.join(_MODE)})"
    ),
    f'{NetState.get_pre_arg()}hidden_cnt': (
        lambda val: int(val) if int(val) >= 1 else 1,
        "int (>=1)"
    )
}


def get_net_rbm_state_types() -> Dict[str, Tuple[Callable[[str], object], str]]:
    bann_gen_net = get_net_general_state_types()
    merged_dict = {**_NetRBMStateTypes}
    for key, value in bann_gen_net.items():
        if key in merged_dict:
            raise KnownNetStateError(f"Duplicated key {key} in {NetRBMState.__name__}")
        merged_dict[key] = value
    return merged_dict
