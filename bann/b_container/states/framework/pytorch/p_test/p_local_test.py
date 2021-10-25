# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import final, Optional, Dict, Callable, Tuple

from bann.b_container.errors.custom_erors import KnownTestStateError
from bann.b_container.functions.dict_str_repr import dict_json_repr
from bann.b_container.states.framework.interface.test_state import TestState
from bann.b_container.states.framework.pytorch.p_test.p_test_general import PTestGenCon


@final
class TesterLocal(TestState[PTestGenCon]):

    def __init__(self) -> None:
        super().__init__()
        self.__kwargs: Optional[PTestGenCon] = None

    def get_kwargs_repr(self) -> str:
        return dict_json_repr(self.get_kwargs().__dict__, self.get_pre_arg())

    def set_kwargs(self, args_dict: Dict, /) -> None:
        if self.__kwargs is not None:
            raise KnownTestStateError("Kwargs already set!")
        self.__kwargs = super().parse_dict(PTestGenCon, args_dict)

    def get_kwargs(self) -> PTestGenCon:
        if self.__kwargs is None:
            raise KnownTestStateError("Kwargs not set!")
        return self.__kwargs


def get_test_local_state_types() -> Dict[str, Tuple[Callable[[str], object], str]]:
    return {}
