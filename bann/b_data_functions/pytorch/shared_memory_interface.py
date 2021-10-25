# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import abc
from copy import copy
from dataclasses import dataclass
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory

from typing import Tuple, List, Optional, final, TypeVar, Generic
from torch.utils.data import Dataset

import numpy as np  # type: ignore

from bann.b_data_functions.errors.custom_erors import KnownErrorBannData


@final
@dataclass
class TypeShapeCon:
    type: np.dtype = np.dtype('float')
    shape: Tuple[int, ...] = (4,)
    data: Optional[np.ndarray] = None
    shared_data: Optional[SharedMemory] = None


@final
class SmmConManger:
    def __init__(self) -> None:
        self.__smm: SharedMemoryManager = SharedMemoryManager()
        self.__started: bool = False
        self.__stopped: bool = False

    @property
    def smm(self) -> SharedMemoryManager:
        return self.__smm

    def smm_shutdown(self) -> None:
        if self.__started and not self.__stopped:
            self.__smm.shutdown()
            self.__stopped = True

    def smm_start(self) -> None:
        if not (self.__started or self.__stopped):
            self.__smm.start()
            self.__started = True


_TypD = TypeVar('_TypD')


class DataSetSharedMemoryA(abc.ABC, Dataset, Generic[_TypD]):

    def __init__(self, data_len: int, /) -> None:
        super().__init__()
        self.__subset: List[int] = []
        self.__subsets_locked: bool = False
        self.__smm: Optional[SharedMemoryManager] = None
        self.__data_len = data_len

    @final
    def __len__(self) -> int:
        return self.__data_len

    @final
    @property
    def subset(self) -> List[int]:
        return self.__subset

    @final
    def _set_subset(self, indices: List[int], /) -> None:
        self.__subset = indices
        if indices:
            self.__data_len = len(indices)

    @final
    def _lock_subsets(self) -> None:
        self.__subsets_locked = True

    @final
    def create_subsets(self, indices: List[int], /) -> 'DataSetSharedMemoryA':
        if self.__subsets_locked:
            raise KnownErrorBannData("subset of subset is prohibited")
        shallow_copy = copy(self)
        shallow_copy._set_subset(indices)
        shallow_copy._lock_subsets()
        shallow_copy._trim_shallow_copy(indices)
        return shallow_copy

    @abc.abstractmethod
    def _getitem(self, item: int, /) -> _TypD:
        raise NotImplementedError("Abstract method!")

    @final
    def __getitem__(self, item: int) -> _TypD:
        self.remap_shared_memory()
        return self._getitem(item)

    @final
    @property
    def used_smm(self) -> Optional[SharedMemoryManager]:
        return self.__smm

    @abc.abstractmethod
    def _trim_shallow_copy(self, indices: List[int], /) -> None:
        raise NotImplementedError("Abstract method!")

    @abc.abstractmethod
    def remap_shared_memory(self) -> None:
        raise NotImplementedError("Abstract method!")

    @abc.abstractmethod
    def _pre_send_empty(self) -> None:
        raise NotImplementedError("Abstract method!")

    @final
    def pre_send_empty(self) -> None:
        self.__smm = None
        self._pre_send_empty()

    @abc.abstractmethod
    def _move_data_to_shared_memory(self) -> None:
        raise NotImplementedError("Abstract method!")

    @final
    def move_data_to_shared_memory(self, smm: SharedMemoryManager, /) -> None:
        if self.__smm is not None:
            raise KnownErrorBannData("SharedMemoryManager already set")
        self.__smm = smm
        self._move_data_to_shared_memory()


def _generate_shared_mem_it(np_array: np.ndarray, cont: TypeShapeCon,
                            smm: SharedMemoryManager, /) -> SharedMemory:
    cont.shape = np_array.shape
    cont.type = np_array.dtype
    shm = smm.SharedMemory(size=np_array.nbytes)
    np_buffered = np.ndarray(np_array.shape, dtype=np_array.dtype, buffer=shm.buf)
    np_buffered[:] = np_array[:]
    return shm


def remap_shared_mem(data: TypeShapeCon, indices: List[int], /) -> None:
    # TODO (remove copy) at this point DataLoader doesn't work without copy
    if not (data.shared_data is None or data.shape is None or data.type is None):
        data_point = data.shared_data
        np_buffered_data = np.ndarray(data.shape, dtype=data.type, buffer=data_point.buf)
        if indices:
            data.data = np.array(list(np_buffered_data[index_i] for index_i in indices))
        else:
            data.data = copy(np_buffered_data)
        data.shared_data = None


def generate_shared_mem(data_type_shape: TypeShapeCon, smm: SharedMemoryManager, /) -> None:
    data_l = data_type_shape.data
    if data_type_shape.shared_data is None and data_l is None:
        raise KnownErrorBannData("Both data types are empty!")
    if data_l is not None:
        data_type_shape.shared_data = _generate_shared_mem_it(data_l, data_type_shape, smm)
        data_type_shape.data = None


def trim_shallow_copy(data_type_shape: TypeShapeCon, indices: List[int], /) -> TypeShapeCon:
    if data_type_shape.shared_data is None and data_type_shape.data is None:
        raise KnownErrorBannData("Both data types are empty!")
    new_con = TypeShapeCon(type=data_type_shape.type, shape=data_type_shape.shape)
    if indices:
        new_data = data_type_shape.data
        if new_data is not None:
            new_con.data = np.array(list(new_data[data_index] for data_index in indices))
        new_con.shared_data = data_type_shape.shared_data
        return new_con
    new_con.shared_data = data_type_shape.shared_data
    new_con.data = data_type_shape.data
    return new_con


def data_get_item(data: TypeShapeCon, index: int, /) -> np.ndarray:
    if data.data is not None:
        return np.array(data.data[index])
    raise KnownErrorBannData("Should never happen")


def data_shallow_copy_shared_mem(data: TypeShapeCon, /) -> TypeShapeCon:
    if data.shared_data is None:
        raise KnownErrorBannData("Shared data is empty!")
    new_con = TypeShapeCon(type=data.type, shape=data.shape)
    new_con.shared_data = data.shared_data
    return new_con
