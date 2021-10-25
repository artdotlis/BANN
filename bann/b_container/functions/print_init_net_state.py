# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import TypeVar

from bann.b_container.states.general.interface.init_state import InitState
from bann.b_container.states.general.interface.net_state import NetState
from bann.b_container.functions.dict_str_repr import dict_string_repr
from rewowr.public.functions.syncout_dep_functions import logger_print_to_console
from rewowr.public.interfaces.logger_interface import SyncStdoutInterface

_TypeNet = TypeVar('_TypeNet', bound=NetState)
_TypeInit = TypeVar('_TypeInit', bound=InitState)
_TypeState = TypeVar('_TypeState', NetState, InitState)


def _print_to_logger(start_str: str, sync_out: SyncStdoutInterface, states: _TypeState, /) -> None:
    output_string = f"The arguments given to {start_str}:\n"
    output_string += f"\n\t{dict_string_repr(states.get_kwargs().__dict__)}\n"
    logger_print_to_console(sync_out, output_string)


def print_init_net_states(net_state: _TypeNet, initializer: _TypeInit,
                          sync_out: SyncStdoutInterface, /) -> None:
    _print_to_logger("net-state", sync_out, net_state)
    _print_to_logger("init-state", sync_out, initializer)
