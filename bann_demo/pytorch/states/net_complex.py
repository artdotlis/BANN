# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Callable, final, Final

from bann.b_container.errors.custom_erors import KnownNetStateError
from bann.b_container.states.general.interface.net_state import NetState
from bann.b_container.states.general.net.net_general import NetGeneralStateArgs, NetGeneralState, \
    get_net_general_state_types


@final
@dataclass
class _NetComplexStateArgs(NetGeneralStateArgs):
    children_cnt: int = 0


@final
class NetComplexState(NetGeneralState):

    def __init__(self) -> None:
        super().__init__()
        self.__kwargs: Optional[_NetComplexStateArgs] = None

    def get_kwargs(self) -> _NetComplexStateArgs:
        if self.__kwargs is None:
            raise KnownNetStateError("Kwargs not set!")
        return self.__kwargs

    def set_kwargs(self, args_dict: Dict, /) -> None:
        if self.__kwargs is not None:
            raise KnownNetStateError("Kwargs already set!")
        self.__kwargs = super().parse_dict(_NetComplexStateArgs, args_dict)


_NetComplexStateTypes: Final[Dict[str, Tuple[Callable[[str], object], str]]] = {
    f'{NetState.get_pre_arg()}children_cnt': (
        lambda val: int(val) if int(val) >= 0 else 0,
        "int (>=0)"
    )
}


def get_net_complex_state_types() -> Dict[str, Tuple[Callable[[str], object], str]]:
    bann_gen_net = get_net_general_state_types()
    merged_dict = {**_NetComplexStateTypes}
    for key, value in bann_gen_net.items():
        if key in merged_dict:
            raise KnownNetStateError(f"Duplicated key {key} in {NetComplexState.__name__}")
        merged_dict[key] = value
    return merged_dict
