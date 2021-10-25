# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Callable, Type, Tuple, List, final, Final

from bann.b_container.states.framework.pytorch.p_train.p_local_train import LocalTState, \
    get_train_local_state_types
from bann.b_test_train_prepare.pytorch.p_train.local_trainer import LocalTrainer
from bann.b_container.states.framework.pytorch.p_train.p_train_gan import GANTState, \
    get_train_gan_state_types
from bann.b_test_train_prepare.pytorch.p_train.net_trainer_gan import GANTrainer
from bann.b_container.constants.fr_string import FrStPName
from bann.b_container.states.framework.pytorch.p_train.p_train_glw import GLWPTState
from bann.b_test_train_prepare.pytorch.p_train.glw_pretraining import GLWPreTrainer
from bann.b_container.states.framework.pytorch.p_train.p_train_single_thread import \
    SingleThreadTState
from bann.b_test_train_prepare.pytorch.p_train.net_trainer_1thread import SingThreadTrainer
from bann.b_container.states.framework.interface.train_state import TrainState
from bann.b_container.errors.custom_erors import KnownTrainStateError
from bann.b_container.states.framework.pytorch.p_train.p_train_hogwild import \
    HogwildTState, get_train_hogwild_state_types
from bann.b_container.states.framework.pytorch.p_train.p_train_constants import \
    get_train_general_state_types
from bann.b_test_train_prepare.pytorch.p_train.net_trainer_hogwild import HogwildTrainer
from bann.b_test_train_prepare.pytorch.trainer_interface import TrainerInterface
from pan.public.interfaces.config_constants import ExtraArgsNet
from rewowr.public.functions.check_re_wr_wo_arguments import check_parse_type


@final
@dataclass
class _LibElem:
    state_types: Dict[str, Tuple[Callable[[str], object], str]]
    train_state: Type[TrainState]


@final
class TrainAlgWr:
    def __init__(self, trainer_type: Type[TrainerInterface],
                 trainer_state_type: Type[TrainState],
                 trainer_type_name: str,
                 trainer_state: TrainState, /) -> None:
        super().__init__()
        self.__trainer_type: Type[TrainerInterface] = trainer_type
        if not isinstance(trainer_state, trainer_state_type):
            raise KnownTrainStateError(
                f"The expected trainer type is {trainer_state_type.__name__}"
                + f" got {type(trainer_state).__name__}!"
            )
        self.__train_state: TrainState = trainer_state
        self.__trainer_state_type: Type[TrainState] = trainer_state_type
        self.__trainer: TrainerInterface = self.trainer_type()
        self.__trainer.set_train_state(self.__train_state)
        self.__trainer_type_name = trainer_type_name

    @staticmethod
    def param_name() -> str:
        return FrStPName.TR.value

    @property
    def trainer_state_type(self) -> Type[TrainState]:
        return self.__trainer_state_type

    @property
    def train_state(self) -> TrainState:
        return self.__train_state

    @property
    def trainer(self) -> TrainerInterface:
        return self.__trainer

    @property
    def trainer_type(self) -> Type[TrainerInterface]:
        return self.__trainer_type

    @property
    def trainer_type_name(self) -> str:
        return self.__trainer_type_name

    def update_trainer(self, new_params: Tuple[float, ...], param_type: str, /) -> None:
        if param_type != self.trainer_state_type.__name__:
            raise KnownTrainStateError(
                f"The expected trainer type is {self.trainer_state_type.__name__}"
                + f" got {param_type}!"
            )
        self.__train_state.set_new_hyper_param(new_params)
        self.__trainer = self.trainer_type()
        self.__trainer.set_train_state(self.__train_state)


_TrainAlg: Final[Dict[Type, Callable[[TrainState], TrainAlgWr]]] = {
    HogwildTState:
        lambda state: TrainAlgWr(HogwildTrainer, HogwildTState, TrainLibName.HOGWILD.value, state),
    SingleThreadTState:
        lambda state: TrainAlgWr(
            SingThreadTrainer, SingleThreadTState, TrainLibName.SINGLETHREAD.value, state
        ),
    GLWPTState:
        lambda state: TrainAlgWr(GLWPreTrainer, GLWPTState, TrainLibName.GLWPRE.value, state),
    GANTState:
        lambda state: TrainAlgWr(GANTrainer, GANTState, TrainLibName.GAN.value, state),
    LocalTState:
        lambda state: TrainAlgWr(LocalTrainer, LocalTState, TrainLibName.LOCAL.value, state)
}


@final
class TrainLibName(Enum):
    HOGWILD = 'HogwildTrainer'
    SINGLETHREAD = '1ThreadTrainer'
    GLWPRE = 'GLW-Pretrainer'
    GAN = 'GANTrainer'
    LOCAL = 'LocalTrainer'


_TrainLib: Final[Dict[str, _LibElem]] = {
    TrainLibName.HOGWILD.value: _LibElem(
        state_types=get_train_hogwild_state_types(), train_state=HogwildTState
    ),
    TrainLibName.SINGLETHREAD.value: _LibElem(
        state_types=get_train_general_state_types(), train_state=SingleThreadTState
    ),
    TrainLibName.GLWPRE.value: _LibElem(
        state_types=get_train_general_state_types(), train_state=GLWPTState
    ),
    TrainLibName.GAN.value: _LibElem(
        state_types=get_train_gan_state_types(), train_state=GANTState
    ),
    TrainLibName.LOCAL.value: _LibElem(
        state_types=get_train_local_state_types(), train_state=LocalTState
    )
}


def get_train_lib_keys() -> List[str]:
    return list(_TrainLib.keys())


def get_train_state_params(state_id: str, /) -> Dict[str, str]:
    all_params = _TrainLib.get(state_id, None)
    if all_params is None:
        return {}
    return {
        param_name: param_type[1]
        for param_name, param_type in all_params.state_types.items()
    }


def create_train_state(trainer_local: str, extra_args: ExtraArgsNet, /) -> TrainState:
    trainer = TrainAlgWr.param_name()
    if trainer in extra_args.arguments:
        trainer = extra_args.arguments[trainer]
    else:
        trainer = trainer_local

    all_params = _TrainLib.get(trainer, None)
    if all_params is None:
        raise KnownTrainStateError(f"The train algorithm {trainer} is not defined!")

    set_params = {}
    for param_to_find, param_type in all_params.state_types.items():
        if param_to_find in extra_args.arguments:
            set_params[param_to_find] = check_parse_type(
                extra_args.arguments[param_to_find], param_type[0]
            )

    erg = all_params.train_state()
    erg.set_kwargs(set_params)
    return erg


def init_train_alg(train_state: TrainState, /) -> TrainAlgWr:
    train_alg_wr = _TrainAlg.get(type(train_state), None)
    if train_alg_wr is None:
        raise KnownTrainStateError(
            f"Could not find the train_type algorithm with the state {type(train_state).__name__}!"
        )
    return train_alg_wr(train_state)
