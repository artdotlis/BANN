# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Callable, Type, Optional, Tuple, List, final, Final

from bann.b_container.states.general.hyper_optim.hyper_optim_ega import EGAHyperState, \
    get_ega_hyper_state_types
from bann.b_hyper_optim.ega_wr import EGAHyperWr
from bann.b_container.constants.gen_strings import GenStPaName
from bann.b_container.states.general.hyper_optim.simply_retrain import SimplyTrainState, \
    get_simply_train_state_types
from bann.b_hyper_optim.simple_retrain_wr import SimplyTrainWr
from bann.b_hyper_optim.da_pso_hyper_wr import DAPSOHyperWr
from bann.b_hyper_optim.pso_hyper_wr import PSOHyperWr
from bann.b_container.states.general.hyper_optim.hyper_optim_pso import get_pso_hyper_state_types, \
    PSOHyperState, DAPSOHyperState, get_da_pso_hyper_state_types
from bann.b_container.states.general.interface.hyper_optim_state import OptimHyperState
from bann.b_container.errors.custom_erors import KnownHyperOptimError
from bann.b_hyper_optim.hyper_optim_interface import HyperOptimInterface
from bann.b_container.states.general.hyper_optim.hyper_optim_random import RandomHyperState, \
    get_random_hyper_state_types
from bann.b_hyper_optim.random_hyper_wr import RandomHyperWr
from pan.public.interfaces.config_constants import ExtraArgsNet
from rewowr.public.functions.check_re_wr_wo_arguments import check_parse_type


@final
@dataclass
class _LibElem:
    state_types: Dict[str, Tuple[Callable[[str], object], str]]
    hyper_state: Type[OptimHyperState]


@final
class HyperAlgWr:
    def __init__(self, hyper_type: Type[HyperOptimInterface],
                 hyper_state_type: Type[OptimHyperState],
                 hyper_state_type_name: str,
                 hyper_state: OptimHyperState, /) -> None:
        super().__init__()
        self.__hyper_type: Type[HyperOptimInterface] = hyper_type
        if not isinstance(hyper_state, hyper_state_type):
            raise KnownHyperOptimError(
                f"The expected hyper-optim type is {hyper_state_type.__name__}"
                + f" got {type(hyper_state).__name__}!"
            )
        self.__hyper_state: OptimHyperState = hyper_state
        self.__hyper_state_type: Type[OptimHyperState] = hyper_state_type
        self.__hyper: HyperOptimInterface = self.hyper_type()
        self.__hyper.set_hyper_state(self.__hyper_state)
        self.__hyper_type_name = hyper_state_type_name

    @staticmethod
    def param_name() -> str:
        return GenStPaName.HYPER.value

    @property
    def stop_file(self) -> Optional[Path]:
        return self.__hyper_state.get_kwargs().stop_fp

    @property
    def stop_iterations(self) -> int:
        return self.__hyper_state.get_kwargs().stop_it

    @property
    def stop_time_min(self) -> int:
        return self.__hyper_state.get_kwargs().stop_min

    @property
    def hyper_state_type(self) -> Type[OptimHyperState]:
        return self.__hyper_state_type

    @property
    def hyper_state(self) -> OptimHyperState:
        return self.__hyper_state

    @property
    def hyper(self) -> HyperOptimInterface:
        return self.__hyper

    @property
    def hyper_type(self) -> Type[HyperOptimInterface]:
        return self.__hyper_type

    @property
    def hyper_type_name(self) -> str:
        return self.__hyper_type_name


_HyperAlg: Final[Dict[Type, Callable[[OptimHyperState], HyperAlgWr]]] = {
    RandomHyperState:
        lambda state: HyperAlgWr(RandomHyperWr, RandomHyperState, HyperLibName.RANDOM.value, state),
    PSOHyperState:
        lambda state: HyperAlgWr(PSOHyperWr, PSOHyperState, HyperLibName.PSO.value, state),
    DAPSOHyperState:
        lambda state: HyperAlgWr(DAPSOHyperWr, DAPSOHyperState, HyperLibName.DAPSO.value, state),
    SimplyTrainState:
        lambda state: HyperAlgWr(SimplyTrainWr, SimplyTrainState, HyperLibName.STR.value, state),
    EGAHyperState:
        lambda state: HyperAlgWr(EGAHyperWr, EGAHyperState, HyperLibName.EGA.value, state)
}


@final
class HyperLibName(Enum):
    RANDOM = 'RandomHyper'
    PSO = 'PSOHyper'
    DAPSO = 'DAPSOHyper'
    STR = 'SimplyTrain'
    EGA = 'EGAHyper'


_HyperLib: Final[Dict[str, _LibElem]] = {
    HyperLibName.STR.value: _LibElem(
        state_types=get_simply_train_state_types(), hyper_state=SimplyTrainState
    ),
    HyperLibName.RANDOM.value: _LibElem(
        state_types=get_random_hyper_state_types(), hyper_state=RandomHyperState
    ),
    HyperLibName.PSO.value: _LibElem(
        state_types=get_pso_hyper_state_types(), hyper_state=PSOHyperState
    ),
    HyperLibName.DAPSO.value: _LibElem(
        state_types=get_da_pso_hyper_state_types(), hyper_state=DAPSOHyperState
    ),
    HyperLibName.EGA.value: _LibElem(
        state_types=get_ega_hyper_state_types(), hyper_state=EGAHyperState
    )
}


def get_hyper_lib_keys() -> List[str]:
    return list(_HyperLib.keys())


def get_hyper_state_params(state_id: str, /) -> Dict[str, str]:
    all_params = _HyperLib.get(state_id, None)
    if all_params is None:
        return {}
    return {
        param_name: param_type[1]
        for param_name, param_type in all_params.state_types.items()
    }


def create_hyper_state(extra_args: ExtraArgsNet, /) -> Optional[OptimHyperState]:
    hyper_optim = HyperAlgWr.param_name()
    if hyper_optim in extra_args.arguments:
        all_params = _HyperLib.get(extra_args.arguments[hyper_optim], None)
        if all_params is None:
            raise KnownHyperOptimError(
                f"The hyper state {extra_args.arguments[hyper_optim]} is not defined!"
            )
        set_params = {}
        for param_to_find, param_type in all_params.state_types.items():
            if param_to_find in extra_args.arguments:
                set_params[param_to_find] = check_parse_type(
                    extra_args.arguments[param_to_find], param_type[0]
                )

        erg = all_params.hyper_state()
        erg.set_kwargs(set_params)
        return erg
    return None


def init_hyper_state(hyper_state: OptimHyperState, /) -> HyperAlgWr:
    hyper_alg_wr = _HyperAlg.get(type(hyper_state), None)
    if hyper_alg_wr is None:
        raise KnownHyperOptimError(
            f"Could not find the hyper algorithm with the state {type(hyper_alg_wr).__name__}!"
        )
    return hyper_alg_wr(hyper_state)
