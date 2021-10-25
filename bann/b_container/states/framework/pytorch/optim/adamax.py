# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Tuple, Dict, Callable, Optional, Union, Type, final, Final, List, Any, TypeVar, \
    Iterable

from bann.b_container.states.framework.interface.pytorch.optim_per_parameter import \
    PerParameterAbc, PerParameterStateKwargs
from bann.b_container.functions.dict_str_repr import dict_json_repr, LayerWiseArgsCon
from bann.b_container.states.framework.interface.optim_state import MainOptimSt, OptimStateKwargs
from bann.b_container.errors.custom_erors import KnownOptimStateError
from bann.b_container.functions.compare_min_max import CompareNumElem, compare_min_max_float
from bann.b_container.functions.check_arg_complex import get_layer_wise_args, check_arg_tuple
from bann.b_container.functions.compare_min_max import AlwaysCompareNumElem


@final
@dataclass
class _ADAMaxStateKwargs(OptimStateKwargs):
    lr: float = 0.0001
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0

    @property
    def get_optim_dict(self) -> Dict:
        return {
            'lr': self.lr,
            'betas': self.betas,
            'eps': self.eps,
            'weight_decay': self.weight_decay
        }

    # Max Min settings
    max_lr: float = 10.
    min_betas: Tuple[float, float] = (0.9, 0.999)


@final
@dataclass
class _LayerCon(PerParameterStateKwargs):
    lr: Optional[float] = None
    betas: Optional[Tuple[float, float]] = None
    weight_decay: Optional[float] = None


_ComVar = TypeVar('_ComVar', AlwaysCompareNumElem, CompareNumElem)


@final
class ADAMaxState(MainOptimSt[_ADAMaxStateKwargs], PerParameterAbc):

    def __init__(self) -> None:
        super().__init__()
        self.__kwargs: Optional[_ADAMaxStateKwargs] = None
        self.__l_kwargs: Dict[str, _LayerCon] = {}
        self.__l_key_order: List[str] = []

    @staticmethod
    def _order_args() -> List[str]:
        return ['lr', 'betas', 'eps', 'weight_decay']

    @staticmethod
    def _not_to_optim() -> List[str]:
        return ['eps']

    def layer_params(self, optim: bool, /) -> Dict[str, Dict[str, Any]]:
        return self.layer_params_st(
            self.__l_kwargs, self._order_args(), self._not_to_optim(), optim
        )

    def get_kwargs(self) -> _ADAMaxStateKwargs:
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
            *tuple(float for _ in range(4)),
            *tuple(
                float for layer_v in layer_dict.values() for o_v in layer_v.values()
                for _ in (o_v if isinstance(o_v, tuple) else range(1))
            )
        )
        return erg

    def _hyper_min_values(self, com_t: Type[_ComVar], l_key: int, /) \
            -> List[_ComVar]:
        erg = []
        if self.__l_kwargs[self.__l_key_order[l_key]].lr is not None:
            erg.append(com_t(False, 0))
        if self.__l_kwargs[self.__l_key_order[l_key]].betas is not None:
            erg.extend([
                com_t(True, self.get_kwargs().min_betas[0]),
                com_t(True, self.get_kwargs().min_betas[1])
            ])
        if self.__l_kwargs[self.__l_key_order[l_key]].weight_decay is not None:
            erg.append(com_t(True, 0))
        return erg

    @property
    def hyper_min_values(self) -> Tuple[AlwaysCompareNumElem, ...]:
        erg = (
            AlwaysCompareNumElem(False, 0),
            AlwaysCompareNumElem(True, self.get_kwargs().min_betas[0]),
            AlwaysCompareNumElem(True, self.get_kwargs().min_betas[1]),
            AlwaysCompareNumElem(True, 0),
            *[
                value_e for l_i in range(len(self.__l_key_order))
                for value_e in self._hyper_min_values(AlwaysCompareNumElem, l_i)
            ]
        )
        return erg

    def _hyper_max_values(self, l_key: int, /) -> List[AlwaysCompareNumElem]:
        erg = []
        if self.__l_kwargs[self.__l_key_order[l_key]].lr is not None:
            erg.append(AlwaysCompareNumElem(True, self.get_kwargs().max_lr))
        if self.__l_kwargs[self.__l_key_order[l_key]].betas is not None:
            erg.extend([AlwaysCompareNumElem(False, 1), AlwaysCompareNumElem(False, 1)])
        if self.__l_kwargs[self.__l_key_order[l_key]].weight_decay is not None:
            erg.append(AlwaysCompareNumElem(False, 1))
        return erg

    @property
    def hyper_max_values(self) -> Tuple[AlwaysCompareNumElem, ...]:
        erg = (
            AlwaysCompareNumElem(True, self.get_kwargs().max_lr),
            AlwaysCompareNumElem(False, 1),
            AlwaysCompareNumElem(False, 1),
            AlwaysCompareNumElem(False, 1),
            *[
                value_e for l_i in range(len(self.__l_key_order))
                for value_e in self._hyper_max_values(l_i)
            ]
        )
        return erg

    @property
    def min_values(self) -> Tuple[CompareNumElem, ...]:
        erg = (
            CompareNumElem(False, 0),
            CompareNumElem(True, self.get_kwargs().min_betas[0]),
            CompareNumElem(True, self.get_kwargs().min_betas[1]),
            CompareNumElem(True, 0),
            *[
                value_e for l_i in range(len(self.__l_key_order))
                for value_e in self._hyper_min_values(CompareNumElem, l_i)
            ]
        )
        return erg

    def _max_values(self, l_key: int, /) -> List[CompareNumElem]:
        erg = []
        if self.__l_kwargs[self.__l_key_order[l_key]].lr is not None:
            erg.append(CompareNumElem(True, None))
        if self.__l_kwargs[self.__l_key_order[l_key]].betas is not None:
            erg.extend([CompareNumElem(False, 1), CompareNumElem(False, 1)])
        if self.__l_kwargs[self.__l_key_order[l_key]].weight_decay is not None:
            erg.append(CompareNumElem(True, None))
        return erg

    @property
    def max_values(self) -> Tuple[CompareNumElem, ...]:
        max_val = [
            CompareNumElem(True, None),
            CompareNumElem(False, 1),
            CompareNumElem(False, 1),
            CompareNumElem(True, None)
        ]
        max_val.extend(
            value_e for l_i in range(len(self.__l_key_order)) for value_e in self._max_values(l_i)
        )
        return tuple(max_val)

    def set_kwargs(self, args_dict: Dict, /) -> None:
        if self.__kwargs is not None:
            raise KnownOptimStateError("Kwargs already set!")
        sep_dict = self.sep_dict(args_dict)
        self.__kwargs = self.parse_dict(
            _ADAMaxStateKwargs, {
                key: val[0] if isinstance(val, tuple) and len(val) == 1 else val
                for key, val in sep_dict[0].items()
            }
        )
        for key, d_value in sep_dict[1].items():
            if not isinstance(d_value, dict):
                raise KnownOptimStateError(f"Expected dict got {type(d_value).__name__}")
            p_key = self.parse_str(key)
            for l_name, l_value in d_value.items():
                if l_name not in self.__l_key_order:
                    self.__l_key_order.append(l_name)
                if isinstance(l_value, tuple) and len(l_value) == 1:
                    self.__l_kwargs.setdefault(l_name, _LayerCon()).__dict__[p_key] = l_value[0]
                else:
                    self.__l_kwargs.setdefault(l_name, _LayerCon()).__dict__[p_key] = l_value
        self.set_new_hyper_param(self.get_hyper_param())

    @staticmethod
    def _get_iter_tuple(value_e: Union[float, Tuple[float, ...]], /) -> Tuple[float, ...]:
        if isinstance(value_e, tuple):
            return value_e
        return value_e,

    def _iterator_hyper(self) -> Iterable[float]:
        layer_dict = self.layer_params(True)
        enp_v: float
        for l_i in self.__l_key_order:
            for value_n, value_e in layer_dict[l_i].items():
                for enp_v in self._get_iter_tuple(value_e):
                    yield enp_v

    def get_hyper_param(self) -> Tuple[float, ...]:
        results: Tuple[float, ...] = (
            self.get_kwargs().lr,
            *self.get_kwargs().betas,
            self.get_kwargs().weight_decay,
            *list(self._iterator_hyper())
        )
        return results

    def layer_param_cnt(self, hyper: bool, /) -> int:
        cnt = 0
        for l_values in self.layer_params(hyper).values():
            for val in l_values.values():
                if isinstance(val, (set, tuple, list)):
                    cnt += len(val)
                else:
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
        self.get_kwargs().betas = (
            compare_min_max_float(params[1], self.min_values[1], self.max_values[1]),
            compare_min_max_float(params[2], self.min_values[2], self.max_values[2])
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
                    if l_key == 'betas':
                        erg_l = []
                        for _ in range(2):
                            erg_l.append(compare_min_max_float(
                                params[cnt_index], self.min_values[cnt_index + 4],
                                self.max_values[cnt_index + 4]
                            ))
                            cnt_index += 1
                        self.__l_kwargs[layer_n].__dict__[l_key] = tuple(erg_l)
                    else:
                        self.__l_kwargs[layer_n].__dict__[l_key] = compare_min_max_float(
                            params[cnt_index], self.min_values[cnt_index + 4],
                            self.max_values[cnt_index + 4]
                        )
                        cnt_index += 1


_ADAMaxStateTypes: Final[Dict[str, Tuple[Callable[[str], object], str]]] = {
    f'{MainOptimSt.get_pre_arg()}lr': (
        lambda a_v: get_layer_wise_args(lambda val: float(val) if float(val) > 0 else 0.0001, a_v),
        "float;layer_name:float;... (>0)"
    ),
    f'{MainOptimSt.get_pre_arg()}weight_decay': (
        lambda a_v: get_layer_wise_args(lambda val: float(val) if float(val) >= 0 else 0.0, a_v),
        "float;layer_name:float;... (>=0)"
    ),
    f'{MainOptimSt.get_pre_arg()}eps': (
        lambda val: float(val) if float(val) > 0 else 1e-8,
        "float (>=0)"
    ),
    f'{MainOptimSt.get_pre_arg()}betas': (
        lambda a_v: get_layer_wise_args(lambda val: float(val) if float(val) >= 0 else 0.0, a_v),
        "float,...;layer_name:float,...;... (>=0)"
    ),
    f'{MainOptimSt.get_pre_arg()}max_lr': (
        lambda val: float(val) if float(val) > 0 else 2.0,
        "float (>0)"
    ),
    f'{MainOptimSt.get_pre_arg()}min_betas': (
        lambda val: check_arg_tuple(val, (
            lambda val1: float(val1) if 0 <= float(val1) < 1 else 0.0,
            lambda val2: float(val2) if 0 <= float(val2) < 1 else 0.0,
        )),
        "float,... (0<=x<1)"
    )
}


def get_adamax_state_types() -> Dict[str, Tuple[Callable[[str], object], str]]:
    return _ADAMaxStateTypes
