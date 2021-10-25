# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Callable, Final, List, final

from bann.b_container.errors.custom_erors import KnownNetStateError
from bann.b_container.states.general.interface.net_state import NetState
from bann.b_container.states.general.net.net_general import NetGeneralStateArgs, NetGeneralState, \
    get_net_general_state_types

from bann_demo.pytorch.networks.libs.ae_lib import get_ae_list

_MODE: Final[List[str]] = get_ae_list()


@final
@dataclass
class _NetAEStateArgs(NetGeneralStateArgs):
    layer_cnt: int = 1
    input_cnt: int = 1
    auto_encoder: str = _MODE[0]


@final
class NetAEState(NetGeneralState):

    def __init__(self) -> None:
        super().__init__()
        self.__kwargs: Optional[_NetAEStateArgs] = None

    def get_kwargs(self) -> _NetAEStateArgs:
        if self.__kwargs is None:
            raise KnownNetStateError("Kwargs not set!")
        return self.__kwargs

    def set_kwargs(self, args_dict: Dict, /) -> None:
        if self.__kwargs is not None:
            raise KnownNetStateError("Kwargs already set!")
        self.__kwargs = super().parse_dict(_NetAEStateArgs, args_dict)


_NetAEStateTypes: Final[Dict[str, Tuple[Callable[[str], object], str]]] = {
    f'{NetState.get_pre_arg()}layer_cnt': (
        lambda val: int(val) if int(val) >= 1 else 1,
        "int (>=1)"
    ),
    f'{NetState.get_pre_arg()}input_cnt': (
        lambda val: int(val) if int(val) >= 1 else 1,
        "int (>=1)"
    ),
    f'{NetState.get_pre_arg()}auto_encoder': (
        lambda val: str(val) if val in _MODE else _MODE[0],
        f"str ({','.join(_MODE)})"
    )
}


def get_net_ae_state_types() -> Dict[str, Tuple[Callable[[str], object], str]]:
    bann_gen_net = get_net_general_state_types()
    merged_dict = {**_NetAEStateTypes}
    for key, value in bann_gen_net.items():
        if key in merged_dict:
            raise KnownNetStateError(f"Duplicated key {key} in {NetAEState.__name__}")
        merged_dict[key] = value
    return merged_dict
