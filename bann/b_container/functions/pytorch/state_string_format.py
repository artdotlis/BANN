# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from bann.b_container.states.framework.pytorch.lr_scheduler_param import LrSchAlgWr, \
    create_lr_sch_json_param_output
from bann.b_container.functions.pytorch.init_framework_fun import InitNetArgs
from bann.b_container.states.framework.pytorch.optim_param import OptimAlgWr, \
    create_optim_json_param_output


def _create_output_lr_sch(scheduler_wr: LrSchAlgWr, /) -> str:
    state_str = f"\"{scheduler_wr.param_name()}\":\t"
    state_str += f"\"{','.join(name for name in scheduler_wr.lr_sch_type_name)}\""
    output = create_lr_sch_json_param_output(scheduler_wr)
    if output:
        state_str += f",\n\t{output}"
    return state_str


def _create_output_optim(optim_wr: OptimAlgWr, /) -> str:
    state_str = f"\"{optim_wr.param_name()}\":\t"
    state_str += f"\"{','.join(name for name in optim_wr.optim_type_name(False))}\""
    output = create_optim_json_param_output(optim_wr)
    if output:
        state_str += f",\n\t{output}"
    return state_str


def _check_if_empty(output: str, /) -> str:
    if not output:
        return ""
    return f"\n\t{output},"


def create_hyper_param_str(network_name: str, init_args: InitNetArgs, /) -> str:
    # net state
    output_string = f"Net {network_name} arguments:"
    output_string += _check_if_empty(init_args.net_state.get_kwargs_repr())
    # initializer
    output_string += _check_if_empty(init_args.initializer_wr.init_state.get_kwargs_repr())
    # hyper
    if init_args.hyper_optim_wr is not None:
        state_str = f"\"{init_args.hyper_optim_wr.param_name()}\":\t"
        state_str += f"\"{init_args.hyper_optim_wr.hyper_type_name}\""
        if init_args.hyper_optim_wr.hyper_state.get_kwargs_repr():
            state_str += f",\n\t{init_args.hyper_optim_wr.hyper_state.get_kwargs_repr()}"
        output_string += f"\n\t{state_str},"
    # prepare
    state_str = f"\"{init_args.prepare_wr.param_name()}\":\t"
    state_str += f"\"{init_args.prepare_wr.pr_type_name}\""
    if init_args.prepare_wr.pr_state.get_kwargs_repr():
        state_str += f",\n\t{init_args.prepare_wr.pr_state.get_kwargs_repr()}"
    output_string += f"\n\t{state_str},"
    # trainer
    state_str = f"\"{init_args.trainer_wr.param_name()}\":\t"
    state_str += f"\"{init_args.trainer_wr.trainer_type_name}\""
    if init_args.trainer_wr.train_state.get_kwargs_repr():
        state_str += f",\n\t{init_args.trainer_wr.train_state.get_kwargs_repr()}"
    output_string += f"\n\t{state_str},"
    # tester
    state_str = f"\"{init_args.tester_wr.param_name()}\":\t"
    state_str += f"\"{init_args.tester_wr.tester_type_name}\""
    if init_args.tester_wr.test_state.get_kwargs_repr():
        state_str += f",\n\t{init_args.tester_wr.test_state.get_kwargs_repr()}"
    output_string += f"\n\t{state_str},"
    # optim
    if init_args.optimizer_wr is not None:
        output_string += f"\n\t{_create_output_optim(init_args.optimizer_wr)},"
    # scheduler
    if init_args.scheduler_wr is not None:
        output_string += f"\n\t{_create_output_lr_sch(init_args.scheduler_wr)},"
    # criterion
    if init_args.criterion_wr is not None:
        state_str = f"\"{init_args.criterion_wr.param_name()}\":\t"
        state_str += f"\"{init_args.criterion_wr.criterion_type_name}\""
        if init_args.criterion_wr.criterion_state.get_kwargs_repr():
            state_str += f",\n\t{init_args.criterion_wr.criterion_state.get_kwargs_repr()}"
        output_string += f"\n\t{state_str},"

    return output_string.rstrip(',') + "\n"
