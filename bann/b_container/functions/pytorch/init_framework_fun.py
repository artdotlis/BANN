# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import re
from enum import Enum
from typing import Optional, Dict, List, Callable, Tuple, final, Final, Pattern

from dataclasses import dataclass

from bann.b_container.states.framework.pytorch.prepare_param import PrepareAlgWr
from bann.b_container.errors.custom_erors import KnownInitStateError
from bann.b_container.states.general.interface.init_state import NetInitGlobalInterface
from bann.b_frameworks.pytorch.net_model_interface import NetModelInterface
from bann.b_container.states.framework.pytorch.criterion_param import CriterionAlgWr
from bann.b_container.states.framework.pytorch.lr_scheduler_param import LrSchAlgWr
from bann.b_container.states.framework.pytorch.optim_param import OptimAlgWr
from bann.b_container.states.framework.pytorch.test_param import TestAlgWr
from bann.b_container.states.framework.pytorch.train_param import TrainAlgWr
from bann.b_container.states.general.init_var_alg_wr import InitVarAlgWr
from bann.b_container.states.general.net.net_general import NetGeneralState
from bann.b_container.states.general.optim_hyper_param import HyperAlgWr


@final
class ArgNMap(Enum):
    initializer_wr = 'init'
    trainer_wr = 'train'


@final
class ArgNMapI(Enum):
    scheduler_wr = 'lr_scheduler'
    optimizer_wr = 'optim'


def _get_value_to_name_dict() -> Dict[str, str]:
    res = {elem.value: elem.name for elem in ArgNMap}
    for elem in ArgNMapI:
        res[elem.value] = elem.name
    return res


_ArgPatter: Final[Pattern[str]] = re.compile(r'(.+)_(\d+)')


def _parse_arg_map_i(name: str, /) -> Tuple[str, int]:
    res = _ArgPatter.search(name)
    if res is None:
        return name, 0
    return res.group(1), int(res.group(2))


@final
@dataclass
class InitNetArgs:
    # general
    net_state: NetGeneralState
    initializer_wr: InitVarAlgWr
    hyper_optim_wr: Optional[HyperAlgWr]
    # pytorch
    tester_wr: TestAlgWr
    trainer_wr: TrainAlgWr
    prepare_wr: PrepareAlgWr
    # extra
    optimizer_wr: Optional[OptimAlgWr]
    scheduler_wr: Optional[LrSchAlgWr]
    criterion_wr: Optional[CriterionAlgWr]


def _get_state_dict_i(container: InitNetArgs, /) -> Dict:
    if container.scheduler_wr is None:
        res = {f"{ArgNMapI.scheduler_wr.value}_0": None}
    else:
        res = {
            f"{ArgNMapI.scheduler_wr.value}_{lr_i}": lr_sch
            for lr_i, lr_sch in enumerate(container.scheduler_wr.lr_sch_state)
        }
    if container.optimizer_wr is None:
        res.update({f"{ArgNMapI.optimizer_wr.value}_0": None})
    else:
        res.update({
            f"{ArgNMapI.optimizer_wr.value}_{opt_i}": opt
            for opt_i, opt in enumerate(container.optimizer_wr.optim_state)
        })
    return res


def get_state_dict(container: InitNetArgs, /) -> Dict:
    res = {
        ArgNMap.trainer_wr.value: container.trainer_wr.train_state,
        ArgNMap.initializer_wr.value: container.initializer_wr.init_state,
        **_get_state_dict_i(container)
    }
    return res


def _create_update_func(container: InitNetArgs, cont_id: str, /) -> \
        Callable[[Tuple[float, ...], str, NetModelInterface], None]:
    parsed_id = _parse_arg_map_i(cont_id)

    def update_lr_scheduler(new_data: Tuple[float, ...],
                            new_type: str, _: NetModelInterface) -> None:
        if container.scheduler_wr is not None:
            if container.optimizer_wr is None:
                raise KnownInitStateError("Wrong update order!")
            container.scheduler_wr.update_lr_sch(
                parsed_id[1], new_data, new_type, container.optimizer_wr
            )

    def update_optim(new_data: Tuple[float, ...], new_type: str,
                     model: NetModelInterface) -> None:
        if container.optimizer_wr is not None:
            container.optimizer_wr.update_optim(parsed_id[1], new_data, new_type, model)

    def update_train(new_data: Tuple[float, ...], new_type: str, _: NetModelInterface) -> None:
        container.trainer_wr.update_trainer(new_data, new_type)

    def update_init(new_data: Tuple[float, ...], new_type: str, _: NetModelInterface) -> None:
        container.initializer_wr.update_init(new_data, new_type)

    updater_dict = {
        ArgNMap.trainer_wr.value: update_train,
        ArgNMap.initializer_wr.value: update_init
    }
    if container.scheduler_wr is not None:
        for lr_sch_i in range(container.scheduler_wr.lr_sch_chain):
            updater_dict[f"{ArgNMapI.scheduler_wr.value}_{lr_sch_i}"] = update_lr_scheduler
    if container.optimizer_wr is not None:
        for opt_i in range(container.optimizer_wr.optim_chain):
            updater_dict[f"{ArgNMapI.optimizer_wr.value}_{opt_i}"] = update_optim
    reverse_dict = _get_value_to_name_dict()

    def updater(new_data: Tuple[float, ...], new_type: str, model: NetModelInterface) -> None:
        if reverse_dict[parsed_id[0]] in container.__dict__ \
                and updater_dict.get(cont_id, None) is not None:
            updater_dict[cont_id](new_data, new_type, model)

    return updater


def get_updater_dict(container: InitNetArgs, /) -> Dict:
    res = {elem.value: _create_update_func(container, elem.value) for elem in ArgNMap}
    if container.scheduler_wr is not None:
        for lr_i, lr_sch in enumerate(container.scheduler_wr.lr_sch_state):
            new_id = f"{ArgNMapI.scheduler_wr.value}_{lr_i}"
            res[new_id] = _create_update_func(container, new_id)
    if container.optimizer_wr is not None:
        for opt_i, opt in enumerate(container.optimizer_wr.optim_state):
            new_id = f"{ArgNMapI.optimizer_wr.value}_{opt_i}"
            res[new_id] = _create_update_func(container, new_id)
    return res


def _create_init_func(container: InitNetArgs, cont_id: str, /) -> \
        Callable[[NetModelInterface, NetInitGlobalInterface], None]:
    parsed_id = _parse_arg_map_i(cont_id)

    def init_lr_scheduler(_model_net: NetModelInterface,
                          _init_updater: NetInitGlobalInterface) -> None:
        if container.scheduler_wr is not None:
            if container.optimizer_wr is None:
                raise KnownInitStateError("Wrong init order!")
            container.scheduler_wr.init_lr_sch(parsed_id[1], container.optimizer_wr)

    def init_optim(model_net: NetModelInterface, _: NetInitGlobalInterface) -> None:
        if container.optimizer_wr is not None:
            container.optimizer_wr.init_optim(parsed_id[1], model_net)

    def init_train(_model_net: NetModelInterface,
                   _init_updater: NetInitGlobalInterface) -> None:
        pass

    def init_init(_: NetModelInterface, init_updater: NetInitGlobalInterface) -> None:
        container.initializer_wr.init_init(init_updater)

    init_wrapper_dict = {
        ArgNMap.trainer_wr.value: init_train,
        ArgNMap.initializer_wr.value: init_init
    }
    if container.scheduler_wr is not None:
        for lr_sch_i in range(container.scheduler_wr.lr_sch_chain):
            init_wrapper_dict[f"{ArgNMapI.scheduler_wr.value}_{lr_sch_i}"] = init_lr_scheduler
    if container.optimizer_wr is not None:
        for opt_i in range(container.optimizer_wr.optim_chain):
            init_wrapper_dict[f"{ArgNMapI.optimizer_wr.value}_{opt_i}"] = init_optim
    reverse_dict = _get_value_to_name_dict()

    def initializer(model_net: NetModelInterface, init_updater: NetInitGlobalInterface) -> None:
        if reverse_dict[parsed_id[0]] in container.__dict__ \
                and init_wrapper_dict.get(cont_id, None) is not None:
            init_wrapper_dict[cont_id](model_net, init_updater)

    return initializer


def get_init_wrapper_dict(container: InitNetArgs, /) -> Dict:
    res = {elem.value: _create_init_func(container, elem.value) for elem in ArgNMap}
    if container.scheduler_wr is not None:
        for lr_i, lr_sch in enumerate(container.scheduler_wr.lr_sch_state):
            new_id = f"{ArgNMapI.scheduler_wr.value}_{lr_i}"
            res[new_id] = _create_init_func(container, new_id)
    if container.optimizer_wr is not None:
        for opt_i, opt in enumerate(container.optimizer_wr.optim_state):
            new_id = f"{ArgNMapI.optimizer_wr.value}_{opt_i}"
            res[new_id] = _create_init_func(container, new_id)
    return res


def get_update_order(scheduler_cnt: int, optim_cnt: int, /) -> List[str]:
    order = [
        ArgNMap.initializer_wr.value,
        *[f"{ArgNMapI.optimizer_wr.value}_{opt_i}" for opt_i in range(optim_cnt)]
    ]
    order_scheduler = [f"{ArgNMapI.scheduler_wr.value}_{lr_i}" for lr_i in range(scheduler_cnt)]
    return [*order, *order_scheduler, ArgNMap.trainer_wr.value]


def get_init_order(scheduler_cnt: int,  optim_cnt: int, /) -> List[str]:
    order = [
        ArgNMap.initializer_wr.value,
        *[f"{ArgNMapI.optimizer_wr.value}_{opt_i}" for opt_i in range(optim_cnt)]
    ]
    order_scheduler = [f"{ArgNMapI.scheduler_wr.value}_{lr_i}" for lr_i in range(scheduler_cnt)]
    return [*order, *order_scheduler, ArgNMap.trainer_wr.value]


def init_wrapper(model_net: NetModelInterface, init_updater: NetInitGlobalInterface,
                 container: InitNetArgs, /) -> None:
    init_dict = get_init_wrapper_dict(container)
    for init_name in get_init_order(
            0 if container.scheduler_wr is None else container.scheduler_wr.lr_sch_chain,
            0 if container.optimizer_wr is None else container.optimizer_wr.optim_chain
    ):
        if init_name not in init_dict:
            raise KnownInitStateError(f"Could not update {init_name}!")
        init_dict[init_name](model_net, init_updater)
