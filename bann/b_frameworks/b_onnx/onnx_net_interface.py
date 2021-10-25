# -*- coding: utf-8 -*-
"""
'ONNX is an open format built to represent machine learning models.'
    See https://onnx.ai/

.. moduleauthor:: Artur Lissin
"""
import abc
from pathlib import Path
from typing import Generic, TypeVar, Tuple

_TypeDummy = TypeVar('_TypeDummy')
_TypeModel = TypeVar('_TypeModel')


class BOnnxInterface(Generic[_TypeDummy, _TypeModel], abc.ABC):

    @abc.abstractmethod
    def onnx_on(self) -> None:
        raise NotImplementedError("Interface!")

    @abc.abstractmethod
    def onnx_off(self) -> None:
        raise NotImplementedError("Interface!")

    @abc.abstractmethod
    def prepare_for_trace(self, extra_args: Tuple[int, ...], /) -> None:
        raise NotImplementedError("Interface!")

    @abc.abstractmethod
    def _create_dummy_input(self, torch_sizes: Tuple[int, ...], /) -> _TypeDummy:
        raise NotImplementedError("Interface!")

    @staticmethod
    @abc.abstractmethod
    def _onnx_get_name() -> str:
        raise NotImplementedError("Interface!")

    @abc.abstractmethod
    def _get_model(self) -> _TypeModel:
        raise NotImplementedError("Interface!")

    @abc.abstractmethod
    def _export_onnx(self, onnx_net: _TypeModel, dummy_input: _TypeDummy, output_n: str, /) -> None:
        raise NotImplementedError("Interface!")

    def prepare_and_dummy(self, torch_sizes: Tuple[int, ...],
                          prep_args: Tuple[int, ...], /) -> _TypeDummy:
        self.prepare_for_trace(prep_args)
        return self._create_dummy_input(torch_sizes)

    def save_onnx_output(self, output_dir: Path, torch_sizes: Tuple[int, ...],
                         prep_args: Tuple[int, ...], /) -> None:
        onnx_net = self._get_model()
        dummy_input = self.prepare_and_dummy(torch_sizes, prep_args)
        onnx_net_out_p = str(output_dir.joinpath(f"{self._onnx_get_name()}.onnx"))
        self._export_onnx(onnx_net, dummy_input, onnx_net_out_p)
