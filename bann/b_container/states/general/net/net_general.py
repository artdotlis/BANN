# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Optional, Dict, Callable, Tuple, Final, final

from bann.b_container.functions.dict_str_repr import dict_json_repr
from bann.b_container.states.general.interface.net_state import NetStateKwargs, NetState
from bann.b_container.errors.custom_erors import KnownNetStateError


@dataclass
class NetGeneralStateArgs(NetStateKwargs):
    save: bool = True
    dump: bool = False
    cuda: bool = False
    retrain: bool = True
    random: bool = False
    redraw: bool = False
    resample: bool = False
    process: int = 1


class NetGeneralState(NetState[NetGeneralStateArgs]):

    def __init__(self) -> None:
        super().__init__()
        self.__kwargs: Optional[NetGeneralStateArgs] = None

    @final
    def get_kwargs_repr(self) -> str:
        return dict_json_repr(self.get_kwargs().__dict__, self.get_pre_arg())

    def get_kwargs(self) -> NetGeneralStateArgs:
        if self.__kwargs is None:
            raise KnownNetStateError("Kwargs not set!")
        return self.__kwargs

    def set_kwargs(self, args_dict: Dict, /) -> None:
        if self.__kwargs is not None:
            raise KnownNetStateError("Kwargs already set!")
        self.__kwargs = super().parse_dict(NetGeneralStateArgs, args_dict)


_NetGeneralStateTypes: Final[Dict[str, Tuple[Callable[[str], object], str]]] = {
    f'{NetState.get_pre_arg()}save': (lambda save: save == 'T', "T for True else False"),
    f'{NetState.get_pre_arg()}dump': (lambda dump: dump == 'T', "T for True else False"),
    f'{NetState.get_pre_arg()}cuda': (lambda cuda: cuda == 'T', "T for True else False"),
    f'{NetState.get_pre_arg()}retrain': (lambda retrain: retrain == 'T', "T for True else False"),
    f'{NetState.get_pre_arg()}random': (lambda random: random == 'T', "T for True else False"),
    f'{NetState.get_pre_arg()}redraw': (lambda redraw: redraw == 'T', "T for True else False"),
    f'{NetState.get_pre_arg()}resample': (lambda resam: resam == 'T', "T for True else False"),
    f'{NetState.get_pre_arg()}process': (lambda val: int(val) if int(val) >= 1 else 1, "int (>=1)")
}


def get_net_general_state_types() -> Dict[str, Tuple[Callable[[str], object], str]]:
    return _NetGeneralStateTypes
