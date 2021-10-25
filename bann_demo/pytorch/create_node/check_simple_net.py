# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import Type, TypeVar, Final, final

from bann.b_container.states.framework.pytorch.prepare_param import PrepLibName
from bann.b_container.states.framework.pytorch.test_param import TestLibName
from bann.b_container.states.framework.pytorch.train_param import TrainLibName
from bann.b_container.states.general.init_param import InitLibName
from bann.b_frameworks.pytorch.network_nodes.simple_net_abstract import SimpleAnnNet, \
    get_simple_net_framework
from bann.b_container.functions.pytorch.init_framework_fun import InitNetArgs
from bann.b_container.states.general.net.net_general import NetGeneralState
from bann.b_container.states.framework.framework_lib import get_framework_lib
from bann.b_container.states.general.general_lib import get_general_lib

from bann_demo.pytorch.networks.errors.custom_erors import KnownSimpleCheckError

from pan.public.interfaces.config_constants import CheckCreateNetElem, ExtraArgsNet
from pan.public.interfaces.pub_net_interface import NodeANNDataElemInterface

_FRAMEWORK: Final[str] = get_simple_net_framework()
_NetType = TypeVar('_NetType', bound=SimpleAnnNet)


@final
class CheckSimpleNet(CheckCreateNetElem):

    def __init__(self, dict_id: str, framework: str, net_type: Type[_NetType],
                 net_state: str, /) -> None:
        super().__init__()
        self.__dict_id = dict_id
        if framework != _FRAMEWORK:
            raise KnownSimpleCheckError(f"Expected framework {_FRAMEWORK} got {framework}!")
        self.__framework = framework
        if not issubclass(net_type, SimpleAnnNet) or net_type == SimpleAnnNet:
            raise KnownSimpleCheckError(
                f"Expected sub class {SimpleAnnNet.__name__} got {net_type.__name__}"
            )
        self.__net_type = net_type
        self.__net_state = net_state

    def check_args_func(self, extra_args: ExtraArgsNet, /) -> NodeANNDataElemInterface:
        frame_work_lib = get_framework_lib(self.framework)
        general_lib = get_general_lib()
        net_state = general_lib.net_param(self.__net_state, extra_args)
        if not isinstance(net_state, NetGeneralState):
            raise KnownSimpleCheckError(
                f"Expected type {NetGeneralState.__name__} got {type(net_state).__name__}"
            )
        hyper_optim_state = general_lib.hyper_param(extra_args)
        hyper_optim = None if hyper_optim_state is None else general_lib.hyper_init(
            hyper_optim_state
        )
        optimizer_state = frame_work_lib.optim_param(extra_args)
        optimizer = None if not optimizer_state else frame_work_lib.optim_init(
            optimizer_state
        )
        criterion_state = frame_work_lib.criterion_param(extra_args)
        criterion = None if criterion_state is None else frame_work_lib.criterion_init(
            criterion_state
        )
        scheduler = frame_work_lib.lr_scheduler_init(frame_work_lib.lr_scheduler_param(extra_args))
        return self.__net_type(InitNetArgs(
            net_state=net_state,
            initializer_wr=general_lib.init_init(
                general_lib.init_param(InitLibName.GENERAL.value, extra_args)
            ),
            hyper_optim_wr=hyper_optim,
            tester_wr=frame_work_lib.test_init(
                frame_work_lib.test_param(TestLibName.SIMPL.value, extra_args)
            ),
            trainer_wr=frame_work_lib.train_init(
                frame_work_lib.train_param(TrainLibName.HOGWILD.value, extra_args)
            ),
            optimizer_wr=optimizer,
            scheduler_wr=scheduler,
            criterion_wr=criterion,
            prepare_wr=frame_work_lib.prepare_init(
                frame_work_lib.prepare_param(PrepLibName.PASS.value, extra_args)
            )
        ))

    @property
    def dict_id(self) -> str:
        return self.__dict_id

    @property
    def framework(self) -> str:
        return self.__framework

    @property
    def net_type(self) -> Type[NodeANNDataElemInterface]:
        return self.__net_type
