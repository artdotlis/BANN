# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import abc
from dataclasses import dataclass
from typing import Dict, Any, final, Tuple, List, TypeVar

from torch import nn


@dataclass(init=False)
class PerParameterStateKwargs:
    def __init__(self, **_: Any) -> None:  # type: ignore
        super().__init__()


class PerParameterNetInterface(abc.ABC):
    @property
    @abc.abstractmethod
    def layer_modules(self) -> Dict[str, nn.Module]:
        raise NotImplementedError('Interface!')

    @property
    @abc.abstractmethod
    def layer_names(self) -> List[str]:
        raise NotImplementedError('Interface!')


_ConVar = TypeVar('_ConVar', bound=PerParameterStateKwargs)


class PerParameterAbc(abc.ABC):
    @staticmethod
    @final
    def layer_params_st(l_kwargs: Dict[str, _ConVar], order_args: List[str],
                        not_to_optim: List[str], optim: bool, /) -> Dict[str, Dict[str, Any]]:
        results_d: Dict[str, Dict] = {
            l_key: {
                l_p_n: l_value.__dict__[l_p_n] for l_p_n in order_args
                if l_value.__dict__[l_p_n] is not None
            }
            for l_key, l_value in l_kwargs.items()
        }
        if optim:
            to_rem: List[Tuple[str, str]] = [
                (l_key, l_p_n)
                for l_key, l_value in results_d.items()
                for l_p_n in l_value
                if l_p_n in not_to_optim
            ]
            for r_keys in to_rem:
                del results_d[r_keys[0]][r_keys[1]]
        to_rem_key: List[str] = [
            l_key for l_key, l_value in results_d.items() if not l_value
        ]
        for r_key in to_rem_key:
            del results_d[r_key]
        return results_d

    @abc.abstractmethod
    def layer_params(self, optim: bool, /) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def layer_set_new_hyper_param(self, params: Tuple[float, ...], /) -> None:
        raise NotImplementedError('Interface!')

    @abc.abstractmethod
    def layer_param_cnt(self, hyper: bool, /) -> int:
        raise NotImplementedError('Interface!')

    @staticmethod
    @final
    def sep_dict(args_dict: Dict, /) -> Tuple[Dict, Dict]:
        return {
            key: value[0]
            if isinstance(value, tuple) and len(value) == 2
            and isinstance(value[1], dict) else value
            for key, value in args_dict.items()
        }, {
            key: value[1]
            for key, value in args_dict.items()
            if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict)
        }
