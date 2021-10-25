# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import final, Dict, Tuple, Callable

from bann.b_container.functions.dict_str_repr import dict_json_repr
from bann.b_container.states.framework.interface.prepare_state import PrepareState, \
    PrepareStateKwargs


@final
class NoPrepTState(PrepareState[PrepareStateKwargs]):
    def __init__(self) -> None:
        super().__init__()

    def get_kwargs_repr(self) -> str:
        return dict_json_repr(self.get_kwargs().__dict__, self.get_pre_arg())

    def get_kwargs(self) -> PrepareStateKwargs:
        return PrepareStateKwargs()

    def set_kwargs(self, args_dict: Dict, /) -> None:
        pass


def get_no_prepare_state_types() -> Dict[str, Tuple[Callable[[str], object], str]]:
    return {}
