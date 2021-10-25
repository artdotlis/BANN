# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from enum import Enum
from typing import final, Type, Dict, Tuple, Callable, Final, List

from bann.b_container.states.framework.general.prepare.local_prepare import LocalPrepTState, \
    get_local_prepare_state_types
from bann.b_test_train_prepare.pytorch.p_prepare.local_prepare import LocalPrepare
from bann.b_container.constants.fr_string import FrStPName
from bann.b_container.states.framework.general.prepare.cross_validate import CrossValTState, \
    get_prepare_cross_state_types
from bann.b_container.states.framework.general.prepare.no_prepare import NoPrepTState, \
    get_no_prepare_state_types
from bann.b_test_train_prepare.pytorch.p_prepare.cross_validation import CrossValidate
from bann.b_test_train_prepare.pytorch.p_prepare.no_prepare import NoPrepare
from bann.b_container.errors.custom_erors import KnownPrepareError
from bann.b_container.states.framework.interface.prepare_state import PrepareState
from bann.b_test_train_prepare.pytorch.prepare_interface import PrepareInterface

from pan.public.interfaces.config_constants import ExtraArgsNet

from rewowr.public.functions.check_re_wr_wo_arguments import check_parse_type


@final
@dataclass
class _LibElem:
    state_types: Dict[str, Tuple[Callable[[str], object], str]]
    prepare_state: Type[PrepareState]


@final
class PrepareAlgWr:
    def __init__(self, pr_type: Type[PrepareInterface], pr_state_type: Type[PrepareState],
                 pr_state_type_name: str, pr_state: PrepareState, /) -> None:
        super().__init__()
        self.__pr_type: Type[PrepareInterface] = pr_type
        if not isinstance(pr_state, pr_state_type):
            raise KnownPrepareError(
                f"The expected prepare type is {pr_state_type.__name__}"
                + f" got {type(pr_state).__name__}!"
            )
        self.__pr_state: PrepareState = pr_state
        self.__pr_state_type: Type[PrepareState] = pr_state_type
        self.__prepare: PrepareInterface = self.__init_prepare(self.__pr_type, self.__pr_state)
        self.__pr_type_name: str = pr_state_type_name

    @staticmethod
    def __init_prepare(pr_type: Type[PrepareInterface], pr_st: PrepareState) -> PrepareInterface:
        prep = pr_type()
        prep.set_prepare_state(pr_st)
        return prep

    def init_prepare(self) -> None:
        self.__prepare = self.__init_prepare(self.__pr_type, self.__pr_state)

    @staticmethod
    def param_name() -> str:
        return FrStPName.PR.value

    @property
    def pr_state_type(self) -> Type[PrepareState]:
        return self.__pr_state_type

    @property
    def pr_state(self) -> PrepareState:
        return self.__pr_state

    @property
    def prepare(self) -> PrepareInterface:
        return self.__prepare

    @property
    def pr_type(self) -> Type[PrepareInterface]:
        return self.__pr_type

    @property
    def pr_type_name(self) -> str:
        return self.__pr_type_name


_PrepAlg: Final[Dict[Type, Callable[[PrepareState], PrepareAlgWr]]] = {
    CrossValTState:
        lambda state: PrepareAlgWr(CrossValidate, CrossValTState, PrepLibName.CROSS.value, state),
    NoPrepTState:
        lambda state: PrepareAlgWr(NoPrepare, NoPrepTState, PrepLibName.PASS.value, state),
    LocalPrepTState:
        lambda state: PrepareAlgWr(LocalPrepare, LocalPrepTState, PrepLibName.LOCAL.value, state)
}


@final
class PrepLibName(Enum):
    CROSS = 'CrossValidate'
    PASS = 'NoPrep'
    LOCAL = 'Local'


_PrepLib: Final[Dict[str, _LibElem]] = {
    PrepLibName.CROSS.value: _LibElem(
        state_types=get_prepare_cross_state_types(), prepare_state=CrossValTState
    ),
    PrepLibName.PASS.value: _LibElem(
        state_types=get_no_prepare_state_types(), prepare_state=NoPrepTState
    ),
    PrepLibName.LOCAL.value: _LibElem(
        state_types=get_local_prepare_state_types(), prepare_state=LocalPrepTState
    )
}


def get_prepare_lib_keys() -> List[str]:
    return list(_PrepLib.keys())


def get_prepare_state_params(state_id: str, /) -> Dict[str, str]:
    all_params = _PrepLib.get(state_id, None)
    if all_params is None:
        return {}
    return {
        param_name: param_type[1]
        for param_name, param_type in all_params.state_types.items()
    }


def create_prepare_state(prepare_local: str, extra_args: ExtraArgsNet, /) -> PrepareState:
    prepare_str = PrepareAlgWr.param_name()
    if prepare_str not in extra_args.arguments:
        prepare_str = prepare_local

    if prepare_str in extra_args.arguments:
        all_params = _PrepLib.get(extra_args.arguments[prepare_str], None)
        if all_params is None:
            raise KnownPrepareError(
                f"The prepare state {extra_args.arguments[prepare_str]} is not defined!"
            )
        set_params = {}
        for param_to_find, param_type in all_params.state_types.items():
            if param_to_find in extra_args.arguments:
                set_params[param_to_find] = check_parse_type(
                    extra_args.arguments[param_to_find], param_type[0]
                )

        erg = all_params.prepare_state()
        erg.set_kwargs(set_params)
        return erg
    return _PrepLib[PrepLibName.PASS.value].prepare_state()


def init_prepare_state(pr_state: PrepareState, /) -> PrepareAlgWr:
    pr_alg_wr = _PrepAlg.get(type(pr_state), None)
    if pr_alg_wr is None:
        raise KnownPrepareError(
            f"Could not find the prepare algorithm with the state {type(pr_alg_wr).__name__}!"
        )
    return pr_alg_wr(pr_state)
