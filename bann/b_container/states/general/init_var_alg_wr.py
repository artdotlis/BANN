# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import Type, Optional, Tuple, final

from bann.b_container.errors.custom_erors import KnownInitStateError
from bann.b_container.states.general.interface.init_state import InitState, NetInitGlobalInterface


@final
class InitVarAlgWr:
    def __init__(self, init_type: Type[InitState], init_state: InitState, /) -> None:
        super().__init__()
        if not isinstance(init_state, init_type):
            raise KnownInitStateError(
                f"The expected init type is {init_type.__name__} got {type(init_state).__name__}!"
            )
        self.__net_init: Optional[NetInitGlobalInterface] = None
        self.__init_state: InitState = init_state
        self.__init_type: Type[InitState] = init_type

    @property
    def init_state_type(self) -> Type[InitState]:
        return self.__init_type

    @property
    def init_state(self) -> InitState:
        return self.__init_state

    @property
    def net_init(self) -> NetInitGlobalInterface:
        if self.__net_init is None:
            raise KnownInitStateError("The init state was not initialised!")
        return self.__net_init

    def init_init(self, net_init: NetInitGlobalInterface, /) -> None:
        if self.__net_init is not None:
            raise KnownInitStateError("The init state was already initialised!")
        if net_init.update_init_type() != self.init_state_type:
            raise KnownInitStateError(
                f"The expected init type is {self.init_state_type.__name__}"
                + f" got {net_init.update_init_type().__name__}!"
            )
        self.__net_init = net_init

    def update_init(self, new_params: Tuple[float, ...], param_type: str, /) -> None:
        if self.__net_init is None:
            raise KnownInitStateError("The init state was not initialised!")
        if self.__net_init.update_init_type().__name__ != param_type \
                or param_type != self.init_state_type.__name__:
            raise KnownInitStateError(
                f"The expected init type is {self.init_state_type.__name__}"
                + f"got {param_type}!"
            )
        if self.__init_state.get_hyper_param():
            self.__init_state.set_new_hyper_param(new_params)
            self.__net_init.update_init(self.__init_state)
