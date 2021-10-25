# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from enum import Enum
from typing import Dict, Callable, Type, TypeVar, Optional, Generic, Tuple, Union, List, final, \
    Final
from dataclasses import dataclass
from torch import nn

from bann.b_frameworks.pytorch.p_criterion.l1_loss import L1Loss
from bann.b_container.constants.fr_string import FrStPName
from bann.b_container.states.framework.pytorch.criterion.log_cosh_loss_state import \
    get_log_cosh_loss_state_types, LogCoshLossState
from bann.b_frameworks.pytorch.p_criterion.log_cosh_loss import LogCoshLoss
from bann.b_container.states.framework.pytorch.criterion.smooth_l1_loss import \
    get_smooth_l1_loss_state_types, SmoothL1LossState
from bann.b_container.states.framework.pytorch.criterion.mse_loss import MSELossState, \
    get_mse_loss_state_types
from bann.b_container.states.framework.pytorch.criterion.nll_loss import get_nll_loss_state_types, \
    NLLLossState
from bann.b_container.errors.custom_erors import KnownCriterionStateError
from bann.b_container.states.framework.interface.criterion_state import CriterionState
from bann.b_container.states.framework.pytorch.criterion.l1_loss import get_l1_loss_state_types, \
    L1LossState
from bann.b_container.states.framework.pytorch.criterion.cross_entropy import \
    get_cross_entropy_state_types, CrossEntropyLossState
from pan.public.interfaces.config_constants import ExtraArgsNet
from rewowr.public.functions.check_re_wr_wo_arguments import check_parse_type


@final
@dataclass
class _LibElem:
    state_types: Dict[str, Tuple[Callable[[str], object], str]]
    criterion_state: Type[CriterionState]


_CriterionVar = TypeVar(
    '_CriterionVar', nn.CrossEntropyLoss, nn.NLLLoss, nn.MSELoss, nn.SmoothL1Loss,
    LogCoshLoss, L1Loss
)
CriterionAlias = Union[
    nn.CrossEntropyLoss, nn.NLLLoss, nn.MSELoss, nn.SmoothL1Loss, LogCoshLoss, L1Loss
]


@final
class CriterionAlgWr(Generic[_CriterionVar]):

    def __init__(self, criterion_type: Type[_CriterionVar],
                 criterion_state_type: Type[CriterionState],
                 criterion_type_name: str,
                 criterion_state: CriterionState, /) -> None:
        super().__init__()
        self.__criterion_type: Type[_CriterionVar] = criterion_type
        self.__criterion_type_name = criterion_type_name
        if not isinstance(criterion_state, criterion_state_type):
            raise KnownCriterionStateError(
                f"The expected criterion type is {criterion_state_type.__name__}"
                + f" got {type(criterion_type).__name__}!"
            )
        self.__criterion_state: CriterionState = criterion_state
        self.__criterion_state_type: Type[CriterionState] = criterion_state_type
        self.__criterion: _CriterionVar = self.criterion_type(
            **self.__criterion_state.get_kwargs().get_criterion_dict
        )

    @staticmethod
    def param_name() -> str:
        return FrStPName.CRIT.value

    @property
    def criterion_state_type(self) -> Type[CriterionState]:
        return self.__criterion_state_type

    @property
    def criterion_state(self) -> CriterionState:
        return self.__criterion_state

    @property
    def criterion_type(self) -> Type[_CriterionVar]:
        return self.__criterion_type

    @property
    def criterion_type_name(self) -> str:
        return self.__criterion_type_name

    @property
    def criterion(self) -> _CriterionVar:
        return self.__criterion


_CriterionAlg: Final[Dict[Type, Callable[[CriterionState], CriterionAlgWr]]] = {
    MSELossState:
        lambda state: CriterionAlgWr[nn.MSELoss](nn.MSELoss, MSELossState,
                                                 CriterionLibName.MSE.value, state),
    NLLLossState:
        lambda state: CriterionAlgWr[nn.NLLLoss](nn.NLLLoss, NLLLossState,
                                                 CriterionLibName.NLL.value, state),
    L1LossState: lambda state: CriterionAlgWr[L1Loss](L1Loss, L1LossState,
                                                      CriterionLibName.L1.value, state),
    CrossEntropyLossState:
        lambda state: CriterionAlgWr[nn.CrossEntropyLoss](
            nn.CrossEntropyLoss, CrossEntropyLossState, CriterionLibName.CROSSE.value, state
        ),
    SmoothL1LossState:
        lambda state: CriterionAlgWr[nn.SmoothL1Loss](
            nn.SmoothL1Loss, SmoothL1LossState, CriterionLibName.SMOOTHL1.value, state
        ),
    LogCoshLossState:
        lambda state: CriterionAlgWr[LogCoshLoss](
            LogCoshLoss, LogCoshLossState, CriterionLibName.LOGCOSH.value, state
        )
}


@final
class CriterionLibName(Enum):
    LOGCOSH = 'LogCoshLoss'
    SMOOTHL1 = 'SmoothL1Loss'
    MSE = 'MSELoss'
    NLL = 'NLLLoss'
    L1 = 'L1Loss'
    CROSSE = 'CrossEntropyLoss'


_CriterionLib: Final[Dict[str, _LibElem]] = {
    CriterionLibName.MSE.value:
        _LibElem(state_types=get_mse_loss_state_types(), criterion_state=MSELossState),
    CriterionLibName.NLL.value:
        _LibElem(state_types=get_nll_loss_state_types(), criterion_state=NLLLossState),
    CriterionLibName.L1.value:
        _LibElem(state_types=get_l1_loss_state_types(), criterion_state=L1LossState),
    CriterionLibName.CROSSE.value:
        _LibElem(
            state_types=get_cross_entropy_state_types(),
            criterion_state=CrossEntropyLossState
        ),
    CriterionLibName.SMOOTHL1.value:
        _LibElem(state_types=get_smooth_l1_loss_state_types(), criterion_state=SmoothL1LossState),
    CriterionLibName.LOGCOSH.value:
        _LibElem(state_types=get_log_cosh_loss_state_types(), criterion_state=LogCoshLossState)

}


def get_criterion_lib_keys() -> List[str]:
    return list(_CriterionLib.keys())


def get_criterion_state_params(state_id: str, /) -> Dict[str, str]:
    all_params = _CriterionLib.get(state_id, None)
    if all_params is None:
        return {}
    return {
        param_name: param_type[1]
        for param_name, param_type in all_params.state_types.items()
    }


def create_criterion_state(extra_args: ExtraArgsNet, /) -> Optional[CriterionState]:
    criterion = CriterionAlgWr.param_name()
    if criterion in extra_args.arguments:
        all_params = _CriterionLib.get(extra_args.arguments[criterion], None)
        if all_params is None:
            raise KnownCriterionStateError(
                f"The criterion algorithm {extra_args.arguments[criterion]} is not defined!"
            )
        set_params = {}
        for param_to_find, param_type in all_params.state_types.items():
            if param_to_find in extra_args.arguments:
                set_params[param_to_find] = check_parse_type(
                    extra_args.arguments[param_to_find], param_type[0]
                )

        erg = all_params.criterion_state()
        erg.set_kwargs(set_params)
        return erg

    return None


def init_criterion_alg(criterion_state: CriterionState, /) -> CriterionAlgWr:
    criterion_alg_wr = _CriterionAlg.get(type(criterion_state), None)
    if criterion_alg_wr is None:
        raise KnownCriterionStateError(
            f"Could not find the lr_sch_type algorithm: {type(criterion_state).__name__}!"
        )
    return criterion_alg_wr(criterion_state)
