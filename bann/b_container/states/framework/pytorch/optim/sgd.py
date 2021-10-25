# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Dict, Callable, Tuple, Optional, Union, Type, final, Final, Any, List, TypeVar

from bann.b_container.functions.check_arg_complex import get_layer_wise_args
from bann.b_container.states.framework.interface.pytorch.optim_per_parameter import \
    PerParameterAbc, PerParameterStateKwargs
from bann.b_container.functions.dict_str_repr import dict_json_repr, LayerWiseArgsCon
from bann.b_container.states.framework.interface.optim_state import MainOptimSt, OptimStateKwargs
from bann.b_container.errors.custom_erors import KnownOptimStateError
from bann.b_container.functions.compare_min_max import CompareNumElem, compare_min_max_float
from bann.b_container.functions.compare_min_max import AlwaysCompareNumElem


@final
@dataclass
class _SDGStateKwargs(OptimStateKwargs):
    lr: float = 0.0001
    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False

    @property
    def get_optim_dict(self) -> Dict:
        return {
            'lr': self.lr,
            'momentum': self.momentum,
            'dampening': self.dampening,
            'weight_decay': self.weight_decay,
            'nesterov': self.nesterov
        }

    # Max Min settings
    max_lr: float = 10.


@final
@dataclass
class _LayerCon(PerParameterStateKwargs):
    lr: Optional[float] = None
    momentum: Optional[float] = None
    dampening: Optional[float] = None
    weight_decay: Optional[float] = None
    nesterov: Optional[bool] = None


_ComVar = TypeVar('_ComVar', AlwaysCompareNumElem, CompareNumElem)


@final
class SGDState(MainOptimSt[_SDGStateKwargs], PerParameterAbc):

    def __init__(self) -> None:
        super().__init__()
        self.__kwargs: Optional[_SDGStateKwargs] = None
        self.__l_kwargs: Dict[str, _LayerCon] = {}
        self.__l_key_order: List[str] = []

    @staticmethod
    def _order_args() -> List[str]:
        return ['lr', 'momentum', 'dampening', 'weight_decay', 'nesterov']

    @staticmethod
    def _not_to_optim() -> List[str]:
        return ['nesterov']

    def layer_params(self, optim: bool, /) -> Dict[str, Dict[str, Any]]:
        return self.layer_params_st(
            self.__l_kwargs, self._order_args(), self._not_to_optim(), optim
        )

    def get_kwargs(self) -> _SDGStateKwargs:
        if self.__kwargs is None:
            raise KnownOptimStateError("Kwargs not set!")
        return self.__kwargs

    def get_kwargs_repr(self, index: int, /) -> str:
        layer_dict = self.layer_params(False)
        return dict_json_repr({
            d_k: LayerWiseArgsCon(
                default=d_v, layer_wise={
                    l_k: p_value
                    for l_k, l_v in layer_dict.items() if d_k in l_v
                    for p_name, p_value in l_v.items() if p_name == d_k
                }
            )
            for d_k, d_v in self.get_kwargs().__dict__.items()
        }, f"{self.get_pre_arg()}{index}_")

    @property
    def type_values(self) -> Tuple[Union[Type[int], Type[float]], ...]:
        layer_dict = self.layer_params(True)
        erg = (
            float, float, float, float,
            *tuple(float for layer_v in layer_dict.values() for _ in layer_v)
        )
        return erg

    def _hyper_min_values(self, com_t: Type[_ComVar], l_key: int, nesterov: bool, /) \
            -> List[_ComVar]:
        erg = []
        if self.__l_kwargs[self.__l_key_order[l_key]].lr is not None:
            erg.append(com_t(False, 0))
        if self.__l_kwargs[self.__l_key_order[l_key]].momentum is not None:
            erg.append(
                com_t(False, 0)
                if (nesterov and self.__l_kwargs[self.__l_key_order[l_key]].nesterov is None) or
                self.__l_kwargs[self.__l_key_order[l_key]].nesterov is True else com_t(True, 0)
            )
        if self.__l_kwargs[self.__l_key_order[l_key]].dampening is not None:
            erg.append(com_t(True, 0))
        if self.__l_kwargs[self.__l_key_order[l_key]].weight_decay is not None:
            erg.append(com_t(True, 0))
        return erg

    @property
    def hyper_min_values(self) -> Tuple[AlwaysCompareNumElem, ...]:
        erg = (
            AlwaysCompareNumElem(False, 0),
            AlwaysCompareNumElem(False, 0) if self.get_kwargs().nesterov else
            AlwaysCompareNumElem(True, 0),
            AlwaysCompareNumElem(True, 0),
            AlwaysCompareNumElem(True, 0),
            *[
                value_e for l_i in range(len(self.__l_key_order))
                for value_e in self._hyper_min_values(
                    AlwaysCompareNumElem, l_i, self.get_kwargs().nesterov
                )
            ]
        )
        return erg

    def _hyper_max_values(self, l_key: int, nesterov: bool, /) -> List[AlwaysCompareNumElem]:
        erg = []
        if self.__l_kwargs[self.__l_key_order[l_key]].lr is not None:
            erg.append(AlwaysCompareNumElem(True, self.get_kwargs().max_lr))
        if self.__l_kwargs[self.__l_key_order[l_key]].momentum is not None:
            erg.append(AlwaysCompareNumElem(False, 1))
        if self.__l_kwargs[self.__l_key_order[l_key]].dampening is not None:
            erg.append((
                AlwaysCompareNumElem(True, 0)
                if (nesterov and self.__l_kwargs[self.__l_key_order[l_key]].nesterov is None) or
                self.__l_kwargs[self.__l_key_order[l_key]].nesterov is True
                else AlwaysCompareNumElem(False, 1)
            ))
        if self.__l_kwargs[self.__l_key_order[l_key]].weight_decay is not None:
            erg.append(AlwaysCompareNumElem(False, 1))
        return erg

    @property
    def hyper_max_values(self) -> Tuple[AlwaysCompareNumElem, ...]:
        erg = (
            AlwaysCompareNumElem(True, self.get_kwargs().max_lr),
            AlwaysCompareNumElem(False, 1),
            AlwaysCompareNumElem(True, 0) if self.get_kwargs().nesterov else
            AlwaysCompareNumElem(False, 1),
            AlwaysCompareNumElem(False, 1),
            *[
                value_e for l_i in range(len(self.__l_key_order))
                for value_e in self._hyper_max_values(l_i, self.get_kwargs().nesterov)
            ]
        )
        return erg

    @property
    def min_values(self) -> Tuple[CompareNumElem, ...]:
        erg = (
            CompareNumElem(False, 0),
            CompareNumElem(False, 0) if self.get_kwargs().nesterov else
            CompareNumElem(True, 0),
            CompareNumElem(True, 0),
            CompareNumElem(True, 0),
            *[
                value_e for l_i in range(len(self.__l_key_order))
                for value_e in self._hyper_min_values(
                    CompareNumElem, l_i, self.get_kwargs().nesterov
                )
            ]
        )
        return erg

    def _max_values(self, l_key: int, nesterov: bool, /) -> List[CompareNumElem]:
        erg = []
        if self.__l_kwargs[self.__l_key_order[l_key]].lr is not None:
            erg.append(CompareNumElem(True, None))
        if self.__l_kwargs[self.__l_key_order[l_key]].momentum is not None:
            erg.append(CompareNumElem(True, None))
        if self.__l_kwargs[self.__l_key_order[l_key]].dampening is not None:
            erg.append((
                CompareNumElem(True, 0)
                if (nesterov and self.__l_kwargs[self.__l_key_order[l_key]].nesterov is None) or
                self.__l_kwargs[self.__l_key_order[l_key]].nesterov is True
                else CompareNumElem(True, None)
            ))
        if self.__l_kwargs[self.__l_key_order[l_key]].weight_decay is not None:
            erg.append(CompareNumElem(True, None))
        return erg

    @property
    def max_values(self) -> Tuple[CompareNumElem, ...]:
        list_erg = [CompareNumElem(True, None) for _ in range(2)]
        list_erg.extend([
            CompareNumElem(True, 0) if self.get_kwargs().nesterov else
            CompareNumElem(True, None),
            CompareNumElem(True, None),
            *[
                value_e for l_i in range(len(self.__l_key_order))
                for value_e in self._max_values(l_i, self.get_kwargs().nesterov)
            ]

        ])
        return tuple(list_erg)

    def set_kwargs(self, args_dict: Dict, /) -> None:
        if self.__kwargs is not None:
            raise KnownOptimStateError("Kwargs already set!")
        sep_dict = self.sep_dict(args_dict)
        self.__kwargs = self.parse_dict(
            _SDGStateKwargs, {
                key: val[0] if isinstance(val, tuple) else val for key, val in sep_dict[0].items()
            }
        )
        for key, d_value in sep_dict[1].items():
            if not isinstance(d_value, dict):
                raise KnownOptimStateError(f"Expected dict got {type(d_value).__name__}")
            p_key = self.parse_str(key)
            for l_name, l_value in d_value.items():
                if l_name not in self.__l_key_order:
                    self.__l_key_order.append(l_name)
                if isinstance(l_value, tuple):
                    self.__l_kwargs.setdefault(l_name, _LayerCon()).__dict__[p_key] = l_value[0]
                else:
                    self.__l_kwargs.setdefault(l_name, _LayerCon()).__dict__[p_key] = l_value
        self.set_new_hyper_param(self.get_hyper_param())

    def get_hyper_param(self) -> Tuple[float, ...]:
        layer_dict = self.layer_params(True)
        return (
            self.get_kwargs().lr,
            self.get_kwargs().momentum,
            self.get_kwargs().dampening,
            self.get_kwargs().weight_decay,
            *[
                value_e for l_i in self.__l_key_order
                for value_e in layer_dict[l_i].values()
            ]
        )

    def layer_param_cnt(self, hyper: bool, /) -> int:
        cnt = 0
        for l_values in self.layer_params(hyper).values():
            for val in l_values.values():
                if isinstance(val, (set, tuple, list)):
                    raise KnownOptimStateError("Should never happen!")
                cnt += 1
        return cnt

    def set_new_hyper_param(self, params: Tuple[float, ...], /) -> None:
        all_p = 4 + self.layer_param_cnt(True)
        if len(params) != all_p:
            raise KnownOptimStateError(
                f"The argument tuple has {len(params)} elements but needed {all_p}!"
            )
        self.get_kwargs().lr = compare_min_max_float(
            params[0], self.min_values[0], self.max_values[0]
        )
        self.get_kwargs().momentum = compare_min_max_float(
            params[1], self.min_values[1], self.max_values[1]
        )
        self.get_kwargs().dampening = compare_min_max_float(
            params[2], self.min_values[2], self.max_values[2]
        )
        self.get_kwargs().weight_decay = compare_min_max_float(
            params[3], self.min_values[3], self.max_values[3]
        )
        self.layer_set_new_hyper_param(params[4:])

    def layer_set_new_hyper_param(self, params: Tuple[float, ...], /) -> None:
        cnt_index = 0
        layer_dict = self.layer_params(True)
        if params:
            for layer_n in self.__l_key_order:
                for l_key, l_value in layer_dict[layer_n].items():
                    self.__l_kwargs[layer_n].__dict__[l_key] = compare_min_max_float(
                        params[cnt_index], self.min_values[cnt_index + 4],
                        self.max_values[cnt_index + 4]
                    )
                    cnt_index += 1


_SGDStateTypes: Final[Dict[str, Tuple[Callable[[str], object], str]]] = {
    f'{MainOptimSt.get_pre_arg()}lr': (
        lambda a_v: get_layer_wise_args(lambda val: float(val) if float(val) > 0 else 0.0001, a_v),
        "float;layer_name:float;... (>0)"
    ),
    f'{MainOptimSt.get_pre_arg()}weight_decay': (
        lambda a_v: get_layer_wise_args(lambda val: float(val) if float(val) >= 0 else 0.0, a_v),
        "float;layer_name:float;... (>=0)"
    ),
    f'{MainOptimSt.get_pre_arg()}momentum': (
        lambda a_v: get_layer_wise_args(lambda val: float(val) if float(val) >= 0 else 0.0, a_v),
        "float;layer_name:float;... (>=0)"
    ),
    f'{MainOptimSt.get_pre_arg()}dampening': (
        lambda a_v: get_layer_wise_args(lambda val: float(val) if float(val) >= 0 else 0.0, a_v),
        "float;layer_name:float;... (>=0)"
    ),
    f'{MainOptimSt.get_pre_arg()}nesterov': (
        lambda a_v: get_layer_wise_args(lambda val: val == 'T', a_v),
        "True if T else False;layer_name:True if T else False;..."
    ),
    f'{MainOptimSt.get_pre_arg()}max_lr': (
        lambda val: float(val) if float(val) > 0 else 2.0,
        "float (>0)"
    )
}


def get_sgd_state_types() -> Dict[str, Tuple[Callable[[str], object], str]]:
    return _SGDStateTypes
