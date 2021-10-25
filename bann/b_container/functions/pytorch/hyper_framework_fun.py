# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import Dict

from bann.b_container.errors.custom_erors import KnownHyperOptimError
from bann.b_container.functions.pytorch.init_framework_fun import get_updater_dict, get_update_order
from bann.b_frameworks.pytorch.net_model_interface import NetModelInterface
from bann.b_container.functions.pytorch.init_framework_fun import get_state_dict, InitNetArgs
from bann.b_hyper_optim.hyper_optim_interface import \
    HyperOptimInterfaceArgs, HyperOptimReturnElem


def update_hyper_container(container: InitNetArgs,
                           hyper_param: HyperOptimInterfaceArgs, /) -> None:
    for state_lib, state in get_state_dict(container).items():
        if state is not None:
            hyper_param.hyper_args[state_lib] = state.get_hyper_param()
            hyper_param.hyper_max_args[state_lib] = state.hyper_max_values
            hyper_param.hyper_min_args[state_lib] = state.hyper_min_values
            hyper_param.min_max_types[state_lib] = state.type_values
            hyper_param.state_type[state_lib] = type(state).__name__


def create_hyper_arguments(container: InitNetArgs, /) -> HyperOptimInterfaceArgs:
    hyper_state_lib = get_state_dict(container)
    hyper_args = {}
    hyper_max_args = {}
    hyper_min_args = {}
    min_max_types = {}
    type_states = {}
    for state_lib, state in hyper_state_lib.items():
        if state is not None:
            hyper_args[state_lib] = state.get_hyper_param()
            hyper_max_args[state_lib] = state.hyper_max_values
            hyper_min_args[state_lib] = state.hyper_min_values
            min_max_types[state_lib] = state.type_values
            type_states[state_lib] = type(state).__name__
    return HyperOptimInterfaceArgs(
        state_type=type_states,
        hyper_args=hyper_args,
        hyper_max_args=hyper_max_args,
        hyper_min_args=hyper_min_args,
        min_max_types=min_max_types
    )


def update_hyper_params(model_net: NetModelInterface, container: InitNetArgs,
                        new_params: Dict[str, HyperOptimReturnElem], /) -> None:
    update_dict = get_updater_dict(container)
    for update_name in get_update_order(
            0 if container.scheduler_wr is None else container.scheduler_wr.lr_sch_chain,
            0 if container.optimizer_wr is None else container.optimizer_wr.optim_chain
    ):
        if update_name in new_params:
            if update_name not in update_dict:
                raise KnownHyperOptimError(
                    f"Could not update {update_name} in {new_params} or {update_dict}!"
                )
            to_update = new_params[update_name]
            update_dict[update_name](to_update.param, to_update.state_type, model_net)
