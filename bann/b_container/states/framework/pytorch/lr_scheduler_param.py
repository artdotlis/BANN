# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import re
from enum import Enum
from typing import Dict, Callable, Type, Optional, TypeVar, Generic, Tuple, List, final, Final, \
    Pattern
from dataclasses import dataclass
import torch

from bann.b_container.constants.fr_string import FrStPName, StateNLib
from bann.b_container.states.framework.pytorch.lr_schedules.cyclic_lr import \
    get_cyclic_lr_state_types, CyclicLRState
from bann.b_container.states.framework.interface.lr_scheduler import LRSchedulerState
from bann.b_container.errors.custom_erors import KnownLRSchedulerStateError
from bann.b_container.states.framework.pytorch.lr_schedules.step_lr import StepLRState, \
    get_step_lr_state_types
from bann.b_container.states.framework.pytorch.optim_param import OptimAlgWr, OptimizerAlias
from bann.b_container.states.framework.pytorch.lr_schedules.reduce_lr_on_plateau import \
    ReduceLROnPlateauState, get_reduce_lr_on_plateau_state_types
from pan.public.interfaces.config_constants import ExtraArgsNet
from rewowr.public.functions.check_re_wr_wo_arguments import check_parse_type


@final
@dataclass
class _LibElem:
    state_types: Dict[str, Tuple[Callable[[str], object], str]]
    lr_sch_state: Type[LRSchedulerState]


_LrSchVar = TypeVar(
    '_LrSchVar',
    torch.optim.lr_scheduler.StepLR,
    torch.optim.lr_scheduler.ReduceLROnPlateau,
    torch.optim.lr_scheduler.CyclicLR
)
_EVAL_CLASSES: Final = (
    torch.optim.lr_scheduler.ReduceLROnPlateau,
)
_NO_EVAL_CLASSES: Final = (
    torch.optim.lr_scheduler.StepLR,
    torch.optim.lr_scheduler.CyclicLR
)


@final
class _LrSchAlgWrChainElement(Generic[_LrSchVar]):
    def __init__(self, batch_name: Tuple[bool, str],
                 lr_sch_type: Type[_LrSchVar],
                 lr_sch_state_type: Type[LRSchedulerState],
                 lr_sch_state: LRSchedulerState, /) -> None:
        super().__init__()
        self.__lr_sch_type: Type[_LrSchVar] = lr_sch_type
        self.__lr_sch_type_name = batch_name[1]
        self.__batch: bool = batch_name[0]
        self.__lr_sch: Optional[_LrSchVar] = None
        if not isinstance(lr_sch_state, lr_sch_state_type):
            raise KnownLRSchedulerStateError(
                f"The expected scheduler type is {lr_sch_state_type.__name__}"
                + f" got {type(lr_sch_state).__name__}!"
            )
        self.__lr_sch_state: LRSchedulerState = lr_sch_state
        self.__lr_sch_state_type: Type[LRSchedulerState] = lr_sch_state_type

    def step_wrapper(self, loss: float, batch: bool, /) -> None:
        if batch == self.__batch:
            scheduler = self.lr_sch
            if isinstance(scheduler, _EVAL_CLASSES):
                scheduler.step(metrics=loss)
            elif isinstance(scheduler, _NO_EVAL_CLASSES):
                scheduler.step(epoch=None)
            else:
                raise KnownLRSchedulerStateError(
                    f"The given {type(scheduler).__name__} scheduler type is not defined!"
                )

    @property
    def lr_sch_state_type(self) -> Type[LRSchedulerState]:
        return self.__lr_sch_state_type

    @property
    def lr_sch_state(self) -> LRSchedulerState:
        return self.__lr_sch_state

    @property
    def lr_sch(self) -> _LrSchVar:
        if self.__lr_sch is None:
            raise KnownLRSchedulerStateError("The lr_sch algorithm was not initialised!")
        return self.__lr_sch

    @property
    def lr_sch_type(self) -> Type[_LrSchVar]:
        return self.__lr_sch_type

    @property
    def lr_sch_type_name(self) -> str:
        return self.__lr_sch_type_name

    def init_lr_sch(self, optimizer: OptimAlgWr, /) -> None:
        if self.__lr_sch is not None:
            raise KnownLRSchedulerStateError("The lr_sch algorithm was already initialised!")
        self.__lr_sch = self.lr_sch_type(
            optimizer.optim, **self.lr_sch_state.get_kwargs().get_scheduler_dict
        )

    def update_lr_sch(self, new_params: Tuple[float, ...], param_type: str,
                      optimizer: OptimAlgWr, /) -> None:
        if self.__lr_sch is None:
            raise KnownLRSchedulerStateError("The lr_sch algorithm was not initialised!")
        if param_type != self.lr_sch_state_type.__name__:
            raise KnownLRSchedulerStateError(
                f"The expected trainer type is {self.lr_sch_state_type.__name__}"
                + f" got {param_type}!"
            )
        self.__lr_sch_state.set_new_hyper_param(new_params)
        self.__lr_sch = self.lr_sch_type(
            optimizer.optim, **self.__lr_sch_state.get_kwargs().get_scheduler_dict
        )

    def update_only_optim(self, optimizer: OptimizerAlias, /) -> None:
        self.__lr_sch = self.lr_sch_type(
            optimizer, **self.__lr_sch_state.get_kwargs().get_scheduler_dict
        )


@final
class LrSchAlgWr:
    def __init__(self, chain_els: Tuple[_LrSchAlgWrChainElement, ...], /) -> None:
        super().__init__()
        if not chain_els:
            raise KnownLRSchedulerStateError("At least one scheduler is needed")
        self.__lr_sch_chain: Tuple[_LrSchAlgWrChainElement, ...] = chain_els

    @staticmethod
    def param_name() -> str:
        return FrStPName.LRSCH.value

    def update_lr_optim(self, opt_wr: OptimAlgWr, /) -> None:
        if opt_wr.sch_change:
            self.update_only_optim(opt_wr.optim)
            opt_wr.sch_changed()

    def step_wrapper(self, loss: float, batch: bool, /) -> None:
        for scheduler in self.__lr_sch_chain:
            scheduler.step_wrapper(loss, batch)

    @property
    def _lr_sch_chain(self) -> Tuple[_LrSchAlgWrChainElement, ...]:
        if not self.__lr_sch_chain:
            raise KnownLRSchedulerStateError("At least one scheduler is needed")
        return self.__lr_sch_chain

    @property
    def _lr_sch_state_type(self) -> Tuple[Type[LRSchedulerState], ...]:
        return tuple(scheduler.lr_sch_state_type for scheduler in self._lr_sch_chain)

    @property
    def lr_sch_type_name(self) -> Tuple[str, ...]:
        return tuple(scheduler.lr_sch_type_name for scheduler in self._lr_sch_chain)

    @property
    def lr_sch_state(self) -> Tuple[LRSchedulerState, ...]:
        return tuple(scheduler.lr_sch_state for scheduler in self._lr_sch_chain)

    @property
    def lr_sch_chain(self) -> int:
        return len(self._lr_sch_chain)

    @property
    def _lr_sch(self) -> Tuple[_LrSchVar, ...]:
        return tuple(scheduler.lr_sch for scheduler in self._lr_sch_chain)

    @property
    def _lr_sch_type(self) -> Tuple[Type[_LrSchVar], ...]:
        return tuple(scheduler.lr_sch_type for scheduler in self._lr_sch_chain)

    def init_lr_sch(self, index: int, optimizer: OptimAlgWr, /) -> None:
        self._lr_sch_chain[index].init_lr_sch(optimizer)

    def update_lr_sch(self, index: int, new_params: Tuple[float, ...],
                      param_type: str, optimizer: OptimAlgWr, /) -> None:
        self._lr_sch_chain[index].update_lr_sch(new_params, param_type, optimizer)

    def update_only_optim(self, optimizer: OptimizerAlias, /) -> None:
        for scheduler in self._lr_sch_chain:
            scheduler.update_only_optim(optimizer)


_LrSchAlg: Final[Dict[Type, Callable[[LRSchedulerState], _LrSchAlgWrChainElement]]] = {
    StepLRState:
        lambda state: _LrSchAlgWrChainElement[torch.optim.lr_scheduler.StepLR](
            (False, LrSchLibName.STEP.value), torch.optim.lr_scheduler.StepLR,
            StepLRState, state
        ),
    ReduceLROnPlateauState:
        lambda state: _LrSchAlgWrChainElement[torch.optim.lr_scheduler.ReduceLROnPlateau](
            (False, LrSchLibName.ROPL.value), torch.optim.lr_scheduler.ReduceLROnPlateau,
            ReduceLROnPlateauState, state
        ),
    CyclicLRState:
        lambda state: _LrSchAlgWrChainElement[torch.optim.lr_scheduler.CyclicLR](
            (True, LrSchLibName.CYCL.value), torch.optim.lr_scheduler.CyclicLR, CyclicLRState, state
        )
}


@final
class LrSchLibName(Enum):
    STEP = 'StepLR'
    ROPL = 'ReduceLROnPlateau'
    CYCL = 'CyclicLR'


_LrSchLib: Final[Dict[str, _LibElem]] = {
    LrSchLibName.STEP.value:
        _LibElem(state_types=get_step_lr_state_types(), lr_sch_state=StepLRState),
    LrSchLibName.ROPL.value:
        _LibElem(
            state_types=get_reduce_lr_on_plateau_state_types(),
            lr_sch_state=ReduceLROnPlateauState
        ),
    LrSchLibName.CYCL.value:
        _LibElem(state_types=get_cyclic_lr_state_types(), lr_sch_state=CyclicLRState)
}


def get_lr_sch_lib_keys() -> List[str]:
    return list(_LrSchLib.keys())


def get_lr_sch_state_params(state_id: str, /) -> Dict[str, str]:
    all_params = _LrSchLib.get(state_id, None)
    if all_params is None:
        return {}
    return {
        param_name: param_type[1]
        for param_name, param_type in all_params.state_types.items()
    }


_LrSchPrePattern: Final[Pattern[str]] = re.compile(f"^({StateNLib.LRSCH.value})(.+)$")


def fix_scheduler_pre_args(param_name: str, index: int, /) -> str:
    res = _LrSchPrePattern.search(param_name)
    if res is None:
        raise KnownLRSchedulerStateError(f"Param-name {param_name} could not be parsed!")
    return f"{res.group(1)}{index}_{res.group(2)}"


def create_lr_sch_json_param_output(scheduler_wr: LrSchAlgWr, /) -> str:
    return ',\n\t'.join(
        lr_sch_st.get_kwargs_repr(lr_sch_i)
        for lr_sch_i, lr_sch_st in enumerate(scheduler_wr.lr_sch_state, 1)
    )


def _create_lr_sch_state(scheduler: str, index: int,
                         extra_args: ExtraArgsNet, /) -> Optional[LRSchedulerState]:
    all_params = _LrSchLib.get(scheduler, None)
    if all_params is None:
        return None
    set_params = {}
    for param_to_find, param_type in all_params.state_types.items():
        mod_param_to_find = fix_scheduler_pre_args(param_to_find, index)
        if mod_param_to_find in extra_args.arguments:
            set_params[param_to_find] = check_parse_type(
                extra_args.arguments[mod_param_to_find], param_type[0]
            )
    erg = all_params.lr_sch_state()
    erg.set_kwargs(set_params)
    return erg


def create_lr_sch_state(extra_args: ExtraArgsNet, /) -> Tuple[LRSchedulerState, ...]:
    lr_sch = LrSchAlgWr.param_name()
    if lr_sch in extra_args.arguments:
        schedulers = extra_args.arguments[lr_sch].split(',')
        return tuple(lr_sch_res for lr_sch_res in (
            _create_lr_sch_state(lr_sch, lr_i, extra_args)
            for lr_i, lr_sch in enumerate(schedulers, 1)
        ) if lr_sch_res is not None)

    return tuple()


def init_lr_sch_alg(lr_sch_state: Tuple[LRSchedulerState, ...], /) -> Optional[LrSchAlgWr]:
    if not lr_sch_state:
        return None
    res = tuple(
        chain_e(lr_sch_state[chain_i]) if chain_e is not None else chain_e
        for chain_i, chain_e in enumerate(
            _LrSchAlg.get(type(sch_state), None) for sch_state in lr_sch_state
        )
    )
    for sch_state in lr_sch_state:
        if _LrSchAlg.get(type(sch_state), None) is None:
            raise KnownLRSchedulerStateError(
                f"Could not find the lr_sch_type algorithm: {type(sch_state).__name__}!"
            )
    return LrSchAlgWr(tuple(r_elm for r_elm in res if r_elm is not None))
