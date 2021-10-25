# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import re
from enum import Enum
from typing import Dict, Type, Callable, Optional, TypeVar, Generic, Tuple, Union, List, final, \
    Final, Pattern, Set
from dataclasses import dataclass

import torch
from torch import nn
from torch.optim import rmsprop, lbfgs, adamw, adamax, sparse_adam

from bann.b_container.states.framework.interface.pytorch.optim_per_parameter import \
    PerParameterAbc, PerParameterNetInterface
from bann.b_container.constants.fr_string import FrStPName, StateNLib
from bann.b_container.states.framework.pytorch.optim.adamax import ADAMaxState, \
    get_adamax_state_types
from bann.b_container.states.framework.pytorch.optim.adamw import ADAMWState, get_adamw_state_types
from bann.b_container.states.framework.pytorch.optim.lbfgs import LBFGSState, get_lbfgs_state_types
from bann.b_container.states.framework.pytorch.optim.rms_prop import get_rmsprop_state_types, \
    RMSpropState
from bann.b_container.errors.custom_erors import KnownOptimStateError
from bann.b_container.states.framework.interface.optim_state import MainOptimSt
from bann.b_container.states.framework.pytorch.optim.adam import get_adam_state_types, ADAMState
from bann.b_container.states.framework.pytorch.optim.sgd import get_sgd_state_types, SGDState
from bann.b_container.states.framework.pytorch.optim.sparse_adam import \
    get_adam_sparse_state_types, ADAMSparseState
from bann.b_container.states.framework.pytorch.optim.switcher import SWState, \
    get_switcher_state_types
from bann.b_frameworks.pytorch.net_model_interface import NetModelInterface
from pan.public.interfaces.config_constants import ExtraArgsNet
from rewowr.public.functions.check_re_wr_wo_arguments import check_parse_type


@final
@dataclass
class _LibElem:
    state_types: Dict[str, Tuple[Callable[[str], object], str]]
    optim_state: Type[MainOptimSt]


_OptimizerVar = TypeVar(
    '_OptimizerVar',
    torch.optim.SGD,
    torch.optim.Adam, adamw.AdamW, adamax.Adamax, sparse_adam.SparseAdam,
    rmsprop.RMSprop, lbfgs.LBFGS
)
OptimizerAlias = Union[
    torch.optim.SGD,
    torch.optim.Adam, adamw.AdamW, adamax.Adamax, sparse_adam.SparseAdam,
    rmsprop.RMSprop, lbfgs.LBFGS
]


@final
class _OptimAlgWrChainEl(Generic[_OptimizerVar]):

    def __init__(self, optim_type: Type[_OptimizerVar],
                 optim_state_type: Type[MainOptimSt],
                 optim_type_name: str,
                 optim_state: MainOptimSt, /) -> None:
        super().__init__()
        self.__optim_type: Type[_OptimizerVar] = optim_type
        self.__optim: Optional[_OptimizerVar] = None
        if not isinstance(optim_state, optim_state_type):
            raise KnownOptimStateError(
                f"The expected optim type is {optim_state_type.__name__}"
                + f" got {type(optim_state).__name__}!"
            )
        self.__optim_state: MainOptimSt = optim_state
        self.__optim_state_type: Type[MainOptimSt] = optim_state_type
        self.__optim_type_name = optim_type_name

    @property
    def optim_state_type(self) -> Type[MainOptimSt]:
        return self.__optim_state_type

    @property
    def optim_state(self) -> MainOptimSt:
        return self.__optim_state

    @property
    def optim(self) -> _OptimizerVar:
        if self.__optim is None:
            raise KnownOptimStateError("The optim algorithm was not initialised!")
        return self.__optim

    @property
    def optim_type(self) -> Type[_OptimizerVar]:
        return self.__optim_type

    @property
    def optim_type_name(self) -> str:
        return self.__optim_type_name

    def init_optim(self, model_param: NetModelInterface, /) -> None:
        if self.__optim is not None:
            raise KnownOptimStateError("The optim algorithm was already initialised!")
        self.update_only_model(model_param.get_net_com)

    def update_optim(self, new_params: Tuple[float, ...], param_type: str,
                     model_param: NetModelInterface, /) -> None:
        if self.__optim is None:
            raise KnownOptimStateError("The optim algorithm was not initialised!")
        if param_type != self.optim_state_type.__name__:
            raise KnownOptimStateError(
                f"The expected optim type is {self.optim_state_type.__name__}"
                + f" got {param_type}!"
            )
        self.__optim_state.set_new_hyper_param(new_params)
        self.update_only_model(model_param.get_net_com)

    def update_only_model(self, model_param: nn.Module, /) -> None:
        optim_l_state = self.optim_state
        if isinstance(optim_l_state, PerParameterAbc) and optim_l_state.layer_param_cnt(False) >= 1:
            if not isinstance(model_param, PerParameterNetInterface):
                raise KnownOptimStateError("Model does not support per-parameter options!")
            all_layers = model_param.layer_modules
            self.__optim = self.optim_type(
                [
                    {'params': layer_mod.parameters(), **{
                        p_name: p_val
                        for p_name, p_val in
                        optim_l_state.layer_params(False).get(layer_n, {}).items()
                    }}
                    for layer_n, layer_mod in all_layers.items()
                ],
                **self.optim_state.get_kwargs().get_optim_dict
            )
        else:
            self.__optim = self.optim_type(
                model_param.parameters(), **self.optim_state.get_kwargs().get_optim_dict
            )


@final
class OptimAlgWr:
    def __init__(self, chain_els: Tuple[_OptimAlgWrChainEl, ...],
                 switcher: Optional[SWState], /) -> None:
        super().__init__()
        if not chain_els:
            raise KnownOptimStateError("At least one optimizer is needed")
        self.__optim_chain: Tuple[_OptimAlgWrChainEl, ...] = chain_els
        self.__switcher = switcher
        self.__switcher_type = type(switcher)
        self.__sw_epoch: Optional[Set[int]] = None
        if switcher is not None:
            self.__sw_epoch = sorted(set(switcher.get_kwargs().sw_epoch))
            if len(self.__sw_epoch) != len(chain_els) - 1:
                raise KnownOptimStateError(
                    f"Required {len(chain_els) - 1} epochs defined in switcher "
                    + f"got {len(self.__sw_epoch)}!"
                )
        self.__cr_index: int = 0
        self.__sw_index: int = 0
        self.__sch_change: bool = False

    @staticmethod
    def param_name() -> str:
        return FrStPName.OPTIM.value

    @property
    def _optim_state_type(self) -> Tuple[Type[MainOptimSt], ...]:
        res: List[Type[MainOptimSt]] = [
            opt.optim_state_type for opt in self.__optim_chain
        ]
        if self.switcher_state is not None:
            res.append(self.__switcher_type)
        return tuple(res)

    def optim_type_name(self, switcher: bool, /) -> Tuple[str, ...]:
        res: List[str] = [
            opt.optim_type_name for opt in self.__optim_chain
        ]
        if self.switcher_state is not None and switcher:
            res.append(self.__switcher_type.__name__)
        return tuple(res)

    @property
    def optim_state(self) -> Tuple[MainOptimSt, ...]:
        res: List[MainOptimSt] = [
            opt.optim_state for opt in self.__optim_chain
        ]
        if self.switcher_state is not None:
            res.append(self.switcher_state)
        return tuple(res)

    @property
    def optim_chain(self) -> int:
        return len(self.__optim_chain) + (1 if self.switcher_state is not None else 0)

    @property
    def switcher_state(self) -> Optional[SWState]:
        return self.__switcher

    def update_epoch(self, epoch: int, model_param: nn.Module, /) -> None:
        if self.__sw_epoch is not None \
                and epoch > self.__sw_epoch[self.__sw_index] \
                and len(self.__sw_epoch) > self.__cr_index:
            self.__cr_index += 1
            self.__sw_index += (1 if len(self.__sw_epoch) - 1 > self.__sw_index else 0)
            self.__sch_change = True
            self.update_only_model(model_param)

    @property
    def sch_change(self) -> bool:
        return self.__sch_change

    def sch_changed(self) -> None:
        self.__sch_change = False

    @property
    def optim(self) -> _OptimizerVar:
        return self.__optim_chain[self.__cr_index].optim

    @property
    def _optim_type(self) -> Type[_OptimizerVar]:
        return self.__optim_chain[self.__cr_index].optim_type

    def init_optim(self, index: int, model_param: NetModelInterface, /) -> None:
        if index >= self.optim_chain:
            raise KnownOptimStateError(
                f"Out of boundary index: {index}; max allowed: {self.optim_chain - 1}"
            )
        if index < len(self.__optim_chain):
            self.__optim_chain[index].init_optim(model_param)

    def update_optim(self, index: int, new_params: Tuple[float, ...], param_type: str,
                     model_param: NetModelInterface, /) -> None:
        if index >= self.optim_chain:
            raise KnownOptimStateError(
                f"Out of boundary index: {index}; max allowed: {self.optim_chain - 1}"
            )
        if index < len(self.__optim_chain):
            self.__optim_chain[index].update_optim(new_params, param_type, model_param)
        else:
            self.__switcher.set_new_hyper_param(new_params)
            self.__sw_epoch = sorted(set(self.__switcher.get_kwargs().sw_epoch))

    def update_only_model(self, model_param: nn.Module, /) -> None:
        self.__optim_chain[self.__cr_index].update_only_model(model_param)


_OptimAlg: Final[Dict[Type, Callable[[MainOptimSt], _OptimAlgWrChainEl]]] = {
    SGDState:
        lambda state: _OptimAlgWrChainEl[torch.optim.SGD](torch.optim.SGD, SGDState,
                                                          OptimLibName.SGD.value, state),
    ADAMState:
        lambda state: _OptimAlgWrChainEl[torch.optim.Adam](torch.optim.Adam, ADAMState,
                                                           OptimLibName.ADAM.value, state),
    ADAMWState:
        lambda state: _OptimAlgWrChainEl[adamw.AdamW](adamw.AdamW, ADAMWState,
                                                      OptimLibName.ADAMW.value, state),
    ADAMaxState:
        lambda state: _OptimAlgWrChainEl[adamax.Adamax](adamax.Adamax, ADAMaxState,
                                                        OptimLibName.ADAMAX.value, state),
    ADAMSparseState:
        lambda state: _OptimAlgWrChainEl[sparse_adam.SparseAdam](
            sparse_adam.SparseAdam, ADAMSparseState, OptimLibName.ADAMSP.value, state
        ),
    RMSpropState:
        lambda state: _OptimAlgWrChainEl[rmsprop.RMSprop](
            rmsprop.RMSprop, RMSpropState, OptimLibName.RMSP.value, state
        ),
    LBFGSState:
        lambda state: _OptimAlgWrChainEl[lbfgs.LBFGS](
            lbfgs.LBFGS, LBFGSState, OptimLibName.LBFGS.value, state
        )
}


@final
class OptimLibName(Enum):
    SGD = 'SGD'
    ADAM = 'ADAM'
    ADAMW = 'ADAMW'
    ADAMAX = 'ADAMAX'
    ADAMSP = 'SparseADAM'
    RMSP = 'RMSprop'
    LBFGS = 'LBFGS'


_OptimLib: Final[Dict[str, _LibElem]] = {
    OptimLibName.SGD.value: _LibElem(state_types=get_sgd_state_types(), optim_state=SGDState),
    OptimLibName.ADAM.value: _LibElem(state_types=get_adam_state_types(), optim_state=ADAMState),
    OptimLibName.ADAMW.value: _LibElem(state_types=get_adamw_state_types(), optim_state=ADAMWState),
    OptimLibName.ADAMAX.value:
        _LibElem(state_types=get_adam_sparse_state_types(), optim_state=ADAMSparseState),
    OptimLibName.ADAMSP.value:
        _LibElem(state_types=get_adamax_state_types(), optim_state=ADAMaxState),
    OptimLibName.RMSP.value:
        _LibElem(state_types=get_rmsprop_state_types(), optim_state=RMSpropState),
    OptimLibName.LBFGS.value:
        _LibElem(state_types=get_lbfgs_state_types(), optim_state=LBFGSState)
}


def get_optim_lib_keys() -> List[str]:
    return list(_OptimLib.keys())


def get_optim_state_params(state_id: str, /) -> Dict[str, str]:
    all_params = _OptimLib.get(state_id, None)
    if all_params is None:
        return {}
    return {
        param_name: param_type[1]
        for param_name, param_type in all_params.state_types.items()
    }


_OptimPrePattern: Final[Pattern[str]] = re.compile(f"^({StateNLib.OPTIM.value})(.+)$")


def fix_optim_pre_args(param_name: str, index: int, /) -> str:
    res = _OptimPrePattern.search(param_name)
    if res is None:
        raise KnownOptimStateError(f"Param-name {param_name} could not be parsed!")
    return f"{res.group(1)}{index}_{res.group(2)}"


def create_optim_json_param_output(optim_wr: OptimAlgWr, /) -> str:
    return ',\n\t'.join(
        optim_st.get_kwargs_repr(optim_i)
        for optim_i, optim_st in enumerate(optim_wr.optim_state, 1)
        if isinstance(optim_st, MainOptimSt)
    )


def _create_optim_state(optim_name: str, index: int,
                        extra_args: ExtraArgsNet, /) -> Optional[MainOptimSt]:
    all_params = _OptimLib.get(optim_name, None)
    if all_params is None:
        return None
    set_params = {}
    for param_to_find, param_type in all_params.state_types.items():
        mod_param_to_find = fix_optim_pre_args(param_to_find, index)
        if mod_param_to_find in extra_args.arguments:
            set_params[param_to_find] = check_parse_type(
                extra_args.arguments[mod_param_to_find], param_type[0]
            )
    erg = all_params.optim_state()
    erg.set_kwargs(set_params)
    return erg


def _create_sw_state(extra_args: ExtraArgsNet, /) -> Optional[SWState]:
    sw_types = get_switcher_state_types()
    set_params = {}
    for param_to_find, param_type in sw_types.items():
        if param_to_find in extra_args.arguments:
            set_params[param_to_find] = check_parse_type(
                extra_args.arguments[param_to_find], param_type[0]
            )
    erg = SWState()
    erg.set_kwargs(set_params)
    return erg


def create_optim_state(extra_args: ExtraArgsNet, /) -> Tuple[MainOptimSt, ...]:
    optim = OptimAlgWr.param_name()
    if optim in extra_args.arguments:
        optim_names = extra_args.arguments[optim].split(',')
        optim_res: List[MainOptimSt] = [
            opt_res for opt_res in (
                _create_optim_state(opt, opt_i, extra_args)
                for opt_i, opt in enumerate(optim_names, 1)
            ) if opt_res is not None
        ]
        if len(optim_res) > 1:
            print("Detected switcher!\n")
            optim_res.append(_create_sw_state(extra_args))
        else:
            print("No switcher detected!\n")
        return tuple(optim_res)

    return tuple()


def init_optim_alg(optim_state: Tuple[MainOptimSt, ...], /) -> Optional[OptimAlgWr]:
    if not optim_state:
        return None
    res = tuple(
        chain_e(optim_state[chain_i]) if chain_e is not None else chain_e
        for chain_i, chain_e in enumerate(
            _OptimAlg.get(type(sch_state), None)
            for sch_state in optim_state
            if not isinstance(sch_state, SWState)
        )
    )
    for optimst in optim_state:
        if optimst is None:
            raise KnownOptimStateError(
                f"Could not find the optim algorithm with the state {type(optim_state).__name__}!"
            )
    switcher: Optional[SWState] = None
    last_v = optim_state[-1]
    if isinstance(last_v, SWState):
        switcher = last_v
    if switcher is None and len(res) > 1:
        raise KnownOptimStateError(
            f"Switcher was not defined, while having more than one ({len(res)}) optimizer!"
        )
    return OptimAlgWr(tuple(
        r_elm for r_elm in res
        if not (r_elm is None or isinstance(r_elm, SWState))
    ), switcher)
