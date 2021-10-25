# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import abc
import atexit
import math
import multiprocessing
import queue
import time
from copy import deepcopy

import pickle as rick
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import reduce
from multiprocessing.managers import SharedMemoryManager
from pathlib import Path
from typing import Union, Tuple, Dict, Optional, Iterable, TypeVar, List, Callable, final, \
    Final

import numpy as np  # type: ignore
from torch import nn
from torch.utils.data import Dataset

from bann.b_container.constants.file_names import TrainSubStrSuf
from bann.b_data_functions.pytorch.p_gen_fun import re_copy_model
from bann.b_test_train_prepare.pytorch.prepare_interface import PrepareInterfaceArgs, \
    PrepareInterface
from bann.b_data_functions.pytorch.shared_memory_interface import DataSetSharedMemoryA, SmmConManger
from bann.b_hyper_optim.fun_const_wr.hyper_fun import h_map_dict_to_tuple, h_create_flat_params, \
    h_create_hyper_space
from bann.b_container.states.general.net.net_general import NetGeneralState
from bann.b_frameworks.pytorch.pytorch_lego_const import LegoContInit
from bann.b_container.states.general.interface.init_state import InitState
from bann.b_frameworks.pytorch.net_model_interface import InitContainer
from bann.b_container.functions.pytorch.state_string_format import create_hyper_param_str
from bann.b_frameworks.errors.custom_erors import KnownSimpleAnnError
from bann.b_container.functions.pytorch.hyper_framework_fun import create_hyper_arguments, \
    update_hyper_params, update_hyper_container
from bann.b_test_train_prepare.pytorch.tester_interface import TesterInterfaceArgs
from bann.b_hyper_optim.hyper_optim_interface import HyperOptimReturnElem, \
    HyperOptimInterfaceArgs, HGenTA
from bann.b_test_train_prepare.pytorch.trainer_interface import TrainerInterfaceArgs, \
    TrainerInterface
from bann.b_frameworks.pytorch.net_model_interface import NetModelInterface, CurrentNetData
from bann.b_pan_integration.framwork_key_lib import FrameworkKeyLib
from bann.b_container.functions.pytorch.init_framework_fun import InitNetArgs

from pan.public.constants.net_tree_id_constants import ANNTreeIdType
from pan.public.constants.test_net_stats_constants import TestNNStatsElementType
from pan.public.constants.train_net_stats_constants import TrainNNStatsElementType, \
    create_train_net_stats_function, TrainNNStatsElemInfo, TrainNNStatsElementFiller, \
    TrainReturnFiller
from pan.public.interfaces.pub_net_interface import NodeANNDataElemInterface, NetSavable

from rewowr.public.functions.worker_ctx import get_worker_ctx
from rewowr.public.interfaces.logger_interface import SyncStdoutInterface
from rewowr.public.errors.custom_errors import KnownError
from rewowr.public.functions.decorator_functions import rewowr_process_wrapper, \
    ProcessWrapperFun


_FRAMEWORK: Final[str] = FrameworkKeyLib.PYTORCH.value


def get_simple_net_framework() -> str:
    return _FRAMEWORK


_ExtraType = TypeVar('_ExtraType')


def _create_hyper_params(args: HyperOptimInterfaceArgs, /) -> Dict[str, HyperOptimReturnElem]:
    return {
        name: HyperOptimReturnElem(
            param=args.hyper_args[name], state_type=args.state_type[name]
        )
        for name, elem in args.hyper_args.items()
    }


_TrFitParam = List[Tuple[float, Dict[str, HyperOptimReturnElem]]]
_TrFitAl = Tuple[_TrFitParam, List[float]]
_TrainArgs = List[Tuple[
    TrainerInterfaceArgs, HyperOptimInterfaceArgs, TrainerInterface, PrepareInterface
]]


def _calc_dev(variance: float, space: float, /) -> float:
    if not space:
        return 0.0
    res = variance / space
    if res < 1.0:
        return res
    return 1.0


def _create_deviation_tuple(hyper_param_t: _TrFitAl,
                            hyper_args: List[HyperOptimInterfaceArgs], /) -> List[float]:
    flat_params_l = [
        h_create_flat_params(hyper_args_e.hyper_args) for hyper_args_e in hyper_args
    ]
    if len(flat_params_l) <= 1 or sum(elem.sum_el for elem in flat_params_l) <= 1:
        return [0.0, 0.0]
    space_list = reduce(
        lambda num3, num4: list(map(lambda num5, num6: num5 + num6, num3, num4)), (
            list(map(
                lambda num1, num2: num1 - num2,
                h_space.search_space_max, h_space.search_space_min
            ))
            for h_space in (
                h_create_hyper_space(
                    hyper_args_e.hyper_max_args, hyper_args_e.hyper_min_args,
                    hyper_args_e.min_max_types, flat_params_l[h_index]
                )
                for h_index, hyper_args_e in enumerate(hyper_args)
            )
        )
    )
    flatten_param_list = [
        h_map_dict_to_tuple(hyper_param_e[1], flat_params_l[h_index])
        for h_index, hyper_param_e in enumerate(hyper_param_t[0])
    ]
    return [
        _calc_dev(float(elem), space_list[num1])
        for num1, elem in enumerate(
            np.std([flatten_p[h_index] for flatten_p in flatten_param_list])
            for h_index in range(len(flatten_param_list[0]))
        )
    ]


@final
@dataclass
class _ResQueueEl:
    fit_0: float
    fit_1: float
    in_id: int
    state_dict: bytes


@final
@dataclass
class _YieldQueueElWr:
    data: TrainNNStatsElementType


def _process_run_fun(index_id: int, prep: PrepareInterface, yield_queue: multiprocessing.Queue,
                     args: PrepareInterfaceArgs, sync_out: SyncStdoutInterface, /) -> None:
    for erg in prep.run_train(sync_out, args):
        yield_queue.put(_YieldQueueElWr(erg))
    yield_queue.put(_ResQueueEl(
        fit_0=prep.fitness[0], fit_1=prep.fitness[1], in_id=index_id,
        state_dict=rick.dumps(prep.p_state_dict, protocol=rick.HIGHEST_PROTOCOL)
    ))


@final
@dataclass
class _ProcessCon:
    ctx: multiprocessing.context.SpawnContext
    target: ProcessWrapperFun
    args: Tuple[
        SyncStdoutInterface, str, Callable[..., None],
        Tuple[
            int, PrepareInterface, multiprocessing.Queue, PrepareInterfaceArgs, SyncStdoutInterface
        ]
    ]


def _get_from_queue(sync_out: SyncStdoutInterface, pr_cont: List[_ProcessCon],
                    yr_queues: multiprocessing.Queue, res_dict: Dict[int, _ResQueueEl],
                    pr_cnt: int, /) -> Iterable[TrainNNStatsElementType]:
    pr_list = []
    id_el = 0
    while id_el < pr_cnt and not sync_out.error_occurred():
        buffer_c = pr_cont[id_el]
        pr_list.append(buffer_c.ctx.Process(target=buffer_c.target, args=buffer_c.args))
        pr_list[id_el].start()
        id_el += 1
    max_id = pr_cnt
    running = not sync_out.error_occurred()
    finished_ids: List[int] = []
    while running:
        try:
            erg_buf = yr_queues.get(True, 2)
        except queue.Empty:
            started = 0
            for id_el in range(max_id):
                if not (pr_list[id_el].is_alive() or id_el in finished_ids):
                    started += 1
                    finished_ids.append(id_el)
            id_el = max_id
            new_stop = max_id + started if max_id + started < len(pr_cont) else len(pr_cont)
            while id_el < new_stop and not sync_out.error_occurred():
                max_id += 1
                buffer_c = pr_cont[id_el]
                pr_list.append(buffer_c.ctx.Process(target=buffer_c.target, args=buffer_c.args))
                pr_list[id_el].start()
                id_el += 1
            if sync_out.error_occurred():
                running = False
            else:
                running = len(finished_ids) < len(pr_cont) or len(res_dict) < len(pr_cont)
        else:
            if isinstance(erg_buf, _YieldQueueElWr):
                yield erg_buf.data
            elif isinstance(erg_buf, _ResQueueEl):
                res_dict[erg_buf.in_id] = erg_buf
            elif isinstance(erg_buf, KnownError):
                raise erg_buf
            else:
                raise KnownSimpleAnnError(
                    f"Expected {_ResQueueEl.__name__}  or {_YieldQueueElWr.__name__} "
                    + f"got {type(erg_buf).__name__}"
                )


def _pre_send_empty(data_t: Tuple[Dataset, ...]) -> None:
    for data in data_t:
        if isinstance(data, DataSetSharedMemoryA):
            data.pre_send_empty()


def _optimise_in_parallel(sync_out: SyncStdoutInterface, process_cnt: int,
                          train_arguments: _TrainArgs,
                          tr_fit: _TrFitAl, /) -> Iterable[TrainNNStatsElementType]:
    ctx = get_worker_ctx()
    if not isinstance(ctx, multiprocessing.context.SpawnContext):
        raise KnownSimpleAnnError(f"Expected SpawnContext got {type(ctx).__name__}")
    yield_queue = ctx.Queue()
    for arg_el in train_arguments:
        _pre_send_empty(arg_el[0].input_train)
        _pre_send_empty(arg_el[0].input_eval)
    pr_list: List[_ProcessCon] = [
        _ProcessCon(
            ctx=ctx,
            target=rewowr_process_wrapper,
            args=(
                sync_out, "OptimTrainStart", _process_run_fun,
                (index_id, args[3], yield_queue, PrepareInterfaceArgs(
                    trainer_args=args[0], trainer=args[2]
                ), sync_out)
            )
        )
        for index_id, args in enumerate(train_arguments)
    ]
    res_dict: Dict[int, _ResQueueEl] = {}
    yield from _get_from_queue(
        sync_out, pr_list, yield_queue, res_dict,
        process_cnt if 0 < process_cnt < len(pr_list) else len(pr_list)
    )
    if len(res_dict) != len(pr_list):
        raise KnownSimpleAnnError(f"Expected {len(pr_list)} results, got {len(res_dict)}")
    for new_id in range(len(pr_list)):
        result_el = res_dict.get(new_id, None)
        if result_el is None:
            raise KnownSimpleAnnError(f"Missing id {new_id}")
        tr_fit[0].append((result_el.fit_0, _create_hyper_params(train_arguments[new_id][1])))
        tr_fit[1].append(result_el.fit_1)
        re_copy_model(rick.loads(result_el.state_dict),
                      train_arguments[new_id][0].module.get_net_com)


@final
@dataclass
class _SDataCont:
    loss: List[float] = field(default_factory=lambda: [])
    truth: List[float] = field(default_factory=lambda: [])
    best_truth: List[float] = field(default_factory=lambda: [])
    best_loss: List[float] = field(default_factory=lambda: [])
    best_change: List[float] = field(default_factory=lambda: [])
    param_deviation: List[float] = field(default_factory=lambda: [])
    run_cnt: int = 0


@final
@dataclass
class _SConstCont:
    best_fit: Tuple[float, Dict[str, HyperOptimReturnElem]]
    best_truth: float
    file_id: ANNTreeIdType


@final
@dataclass
class _TrainNetStatsCon:
    hyper_truth: TrainReturnFiller
    hyper_loss: TrainReturnFiller
    best_changes: TrainReturnFiller
    best_truth: TrainReturnFiller
    best_loss: TrainReturnFiller
    param_deviation: TrainReturnFiller


@final
class _SimpleTrainDataP:
    def __init__(self, hyper_cont: HyperOptimInterfaceArgs, id_file: ANNTreeIdType, /) -> None:
        super().__init__()
        self._train_net_stat_con = _TrainNetStatsCon(
            create_train_net_stats_function(), create_train_net_stats_function(),
            create_train_net_stats_function(), create_train_net_stats_function(),
            create_train_net_stats_function(), create_train_net_stats_function()
        )
        self._data_cont = _SDataCont()
        self._cost_cont = _SConstCont(
            file_id=deepcopy(id_file),
            best_fit=(float('inf'), _create_hyper_params(hyper_cont)),
            best_truth=-3.0
        )
        self._cost_cont.file_id.add_modifier(TrainSubStrSuf.FITNESS.value)

    @property
    def bets_fit_h_param(self) -> Tuple[float, Dict[str, HyperOptimReturnElem]]:
        return self._cost_cont.best_fit

    def update_fitness(self, tr_fit: _TrFitAl, hyper_args: List[HyperOptimInterfaceArgs], /) \
            -> None:
        loss = []
        truth = []
        best_truth = []
        best_loss = []
        best_change = []
        param_deviation = 100. * np.mean(_create_deviation_tuple(tr_fit, hyper_args))
        param_div_l = []

        for param_id, param in enumerate(tr_fit[0]):
            if math.isinf(param[0]):
                loss.append(-1e-12)
            else:
                loss.append(param[0])
            truth.append(tr_fit[1][param_id])
            if param[0] <= self.bets_fit_h_param[0]:
                self._cost_cont.best_fit = deepcopy(param)
                self._cost_cont.best_truth = tr_fit[1][param_id]
                best_change.append(1.0)
            else:
                best_change.append(0.0)
            if math.isinf(self.bets_fit_h_param[0]):
                best_loss.append(-1e-12)
            else:
                best_loss.append(self.bets_fit_h_param[0])
            best_truth.append(self._cost_cont.best_truth)
            param_div_l.append(param_deviation)
        self._data_cont = _SDataCont(
            loss=loss,
            truth=truth,
            best_truth=best_truth,
            best_loss=best_loss,
            best_change=best_change,
            param_deviation=param_div_l,
            run_cnt=len(tr_fit[1])
        )

    def plot(self, last: bool, dump: bool, run_id: int, /) -> Iterable[TrainNNStatsElementType]:
        if last and run_id == 0 and self._data_cont.run_cnt > 1:
            yield from self._plot_in(
                False, dump, run_id, (0, int(self._data_cont.run_cnt / 2))
            )
            yield from self._plot_in(
                True, dump, run_id, (int(self._data_cont.run_cnt / 2), self._data_cont.run_cnt)
            )
        else:
            yield from self._plot_in(last, dump, run_id, (0, self._data_cont.run_cnt))

    def _plot_in(self, last: bool, dump: bool, run_id: int,
                 range_id: Tuple[int, int], /) -> Iterable[TrainNNStatsElementType]:
        yield self._train_net_stat_con.hyper_truth(
            TrainNNStatsElemInfo(
                id_file=self._cost_cont.file_id, name_series="Truth",
                type_series='Leg', name_sub_series="Truth", type_sub_series='Sub',
                x_label="hyper run",
                y_label='Truth/Loss/Change/Dev',
                title="hyper optimization", subtitle=""
            ),
            [run_id + param_id for param_id in range(range_id[0], range_id[1])],
            self._data_cont.truth[range_id[0]:range_id[1]],
            TrainNNStatsElementFiller(
                last=last, plot_data=True, dump=dump, write_data=True, hyper_param=""
            )
        )
        yield self._train_net_stat_con.hyper_loss(
            TrainNNStatsElemInfo(
                id_file=self._cost_cont.file_id, name_series="Loss",
                type_series='Leg', name_sub_series="Loss", type_sub_series='Sub',
                x_label="hyper run",
                y_label='Truth/Loss/Change/Dev',
                title="hyper optimization", subtitle=""
            ),
            [run_id + param_id for param_id in range(range_id[0], range_id[1])],
            self._data_cont.loss[range_id[0]:range_id[1]],
            TrainNNStatsElementFiller(
                last=last, dump=dump, hyper_param="", write_data=True, plot_data=True
            )
        )
        yield self._train_net_stat_con.best_truth(
            TrainNNStatsElemInfo(
                id_file=self._cost_cont.file_id, name_series="TruthBest",
                type_series='Leg', name_sub_series="Truth", type_sub_series='Sub',
                x_label="hyper run",
                y_label='Truth/Loss/Change/Dev',
                title="hyper optimization", subtitle=""
            ),
            [run_id + param_id for param_id in range(range_id[0], range_id[1])],
            self._data_cont.best_truth[range_id[0]:range_id[1]],
            TrainNNStatsElementFiller(
                last=last, dump=dump, hyper_param="", write_data=True, plot_data=True
            )
        )
        yield self._train_net_stat_con.best_loss(
            TrainNNStatsElemInfo(
                id_file=self._cost_cont.file_id, name_series="LossBest",
                type_series='Leg', name_sub_series="Loss", type_sub_series='Sub',
                x_label="hyper run",
                y_label='Truth/Loss/Change/Dev',
                title="hyper optimization", subtitle=""
            ),
            [run_id + param_id for param_id in range(range_id[0], range_id[1])],
            self._data_cont.best_loss[range_id[0]:range_id[1]],
            TrainNNStatsElementFiller(
                last=last, dump=dump, hyper_param="", write_data=True, plot_data=True
            )
        )
        yield self._train_net_stat_con.best_changes(
            TrainNNStatsElemInfo(
                id_file=self._cost_cont.file_id, name_series="BestChange",
                type_series='Leg', name_sub_series="Best", type_sub_series='Sub',
                x_label="hyper run",
                y_label='Truth/Loss/Change/Dev',
                title="hyper optimization", subtitle=""
            ),
            [run_id + param_id for param_id in range(range_id[0], range_id[1])],
            self._data_cont.best_change[range_id[0]:range_id[1]],
            TrainNNStatsElementFiller(
                last=last, dump=dump, hyper_param="", write_data=True, plot_data=True
            )
        )
        yield self._train_net_stat_con.param_deviation(
            TrainNNStatsElemInfo(
                id_file=self._cost_cont.file_id, name_series="ParamDeviation",
                type_series='Leg', name_sub_series="Dev", type_sub_series='Sub',
                x_label="hyper run",
                y_label='Truth/Loss/Change/Dev',
                title="hyper optimization", subtitle=""
            ),
            [run_id + param_id for param_id in range(range_id[0], range_id[1])],
            self._data_cont.param_deviation[range_id[0]:range_id[1]],
            TrainNNStatsElementFiller(
                last=last, dump=dump, hyper_param="", write_data=True, plot_data=True
            )
        )


@final
@dataclass
class _RunningConst:
    fit_plotter: _SimpleTrainDataP
    hyper_cont_buffer: HyperOptimInterfaceArgs
    init_time: float
    run_id: int = 0
    running: bool = True
    smm_con: Optional[SmmConManger] = None

    @property
    def run_time_min(self) -> int:
        return int((time.time() - self.init_time) / 60)

    @property
    def smm(self) -> Optional[SharedMemoryManager]:
        if self.smm_con is None:
            return None
        return self.smm_con.smm

    def shutdown(self) -> None:
        if self.smm_con is not None:
            self.smm_con.smm_shutdown()
            self.smm_con.smm.join()
        self.smm_con = None

    def start(self) -> None:
        if self.smm_con is not None:
            self.smm_con.smm_start()

    def init(self) -> None:
        if self.smm_con is None:
            self.smm_con = SmmConManger()


_TypeBuffer = TypeVar('_TypeBuffer')


class SimpleNetCon(NetModelInterface[_TypeBuffer], abc.ABC):

    def __init__(self) -> None:
        super().__init__()
        self.__current_net: Union[Tuple[float, Dict, Dict], CurrentNetData] = \
            self._create_current_net()
        self.__buffered_best_net: Optional[CurrentNetData] = deepcopy(self.__current_net)
        self.__init_blank_net: Optional[CurrentNetData] = deepcopy(self.__current_net)

    @property
    @abc.abstractmethod
    def lego_init_cont(self) -> LegoContInit:
        raise NotImplementedError("Abstract method!")

    @abc.abstractmethod
    def _create_current_net(self) -> CurrentNetData:
        raise NotImplementedError("Abstract method!")

    @abc.abstractmethod
    def _create_current_loaded_net(self, extra_args: InitContainer, /) -> CurrentNetData:
        raise NotImplementedError("Abstract method!")

    @final
    @property
    def current_net(self) -> Union[CurrentNetData, Tuple[float, Dict, Dict]]:
        return self.__current_net

    @final
    @property
    def buffered_best_net(self) -> CurrentNetData:
        if self.__buffered_best_net is None:
            raise KnownSimpleAnnError("The net was not appropriately loaded!")
        return self.__buffered_best_net

    @final
    @property
    def init_blank_net(self) -> CurrentNetData:
        if self.__init_blank_net is None:
            raise KnownSimpleAnnError("The net was not appropriately loaded!")
        return self.__init_blank_net

    @abc.abstractmethod
    def remove_before_save(self) -> _TypeBuffer:
        raise NotImplementedError("Abstract method!")

    @abc.abstractmethod
    def reload_after_save(self, data: _TypeBuffer, /) -> None:
        raise NotImplementedError("Abstract method!")

    # ----------------------------------------------------------------------------------------------

    @final
    def redraw_current_net(self) -> None:
        if not isinstance(self.current_net, CurrentNetData):
            raise KnownSimpleAnnError(f"SimpleNetCon is not in {CurrentNetData.__name__} mode")
        self.__current_net = self._create_current_net()

    @final
    def merge_net_model(self, model: NetModelInterface, /) -> None:
        if not isinstance(model, SimpleNetCon):
            raise KnownSimpleAnnError(
                f"Expected {SimpleNetCon.__name__} got {type(model).__name__}"
            )
        self.__current_net = deepcopy(model.current_net)

    @final
    def re_copy_current_net(self) -> None:
        if not isinstance(self.current_net, CurrentNetData):
            raise KnownSimpleAnnError(f"SimpleNetCon is not in {CurrentNetData.__name__} mode")
        self.__buffered_best_net = deepcopy(self.current_net)
        self.__init_blank_net = deepcopy(self.current_net)

    @final
    def re_init_current_net(self, new_net: CurrentNetData, /) -> None:
        if not isinstance(self.current_net, CurrentNetData):
            raise KnownSimpleAnnError(f"SimpleNetCon is not in {CurrentNetData.__name__} mode")
        self.__current_net = deepcopy(new_net)
        self.__buffered_best_net = deepcopy(new_net)
        self.__init_blank_net = deepcopy(new_net)

    @final
    def update_current_net(self, fitness: float, /) -> None:
        if not isinstance(self.__current_net, CurrentNetData):
            raise KnownSimpleAnnError("The net was not appropriately loaded!")
        old_fitness = self.buffered_best_net.fitness
        self.__current_net.fitness = fitness
        if fitness <= old_fitness:
            self.__buffered_best_net = deepcopy(self.__current_net)

    @final
    def reset_current_net(self) -> None:
        if not isinstance(self.__current_net, CurrentNetData):
            raise KnownSimpleAnnError("The net was not appropriately loaded!")
        self.__current_net = deepcopy(self.init_blank_net)

    @final
    def set_best_net(self) -> None:
        if not isinstance(self.__current_net, CurrentNetData):
            raise KnownSimpleAnnError("The net was not appropriately loaded!")
        self.__current_net = deepcopy(self.buffered_best_net)

    @final
    @property
    def get_net_com(self) -> nn.Module:
        if not isinstance(self.__current_net, CurrentNetData):
            raise KnownSimpleAnnError("The net was not appropriately loaded!")
        return self.__current_net.com

    @final
    @property
    def get_net_lego(self) -> nn.Module:
        if not isinstance(self.__current_net, CurrentNetData):
            raise KnownSimpleAnnError("The net was not appropriately loaded!")
        return self.__current_net.lego

    @final
    def save(self) -> Tuple[
        bytes, Tuple[CurrentNetData, CurrentNetData, CurrentNetData], _TypeBuffer
    ]:
        cr_net = self.current_net
        if not isinstance(cr_net, CurrentNetData):
            raise KnownSimpleAnnError("The net was not appropriately loaded!")
        buf_net = self.buffered_best_net
        self.__current_net = (buf_net.fitness, buf_net.com.state_dict(), buf_net.lego.state_dict())
        init_net = self.init_blank_net
        self.__buffered_best_net = None
        self.__init_blank_net = None
        rem_buf = self.remove_before_save()
        erg = (
            rick.dumps(self, protocol=rick.HIGHEST_PROTOCOL),
            (cr_net, buf_net, init_net), rem_buf
        )
        return erg

    @final
    def save_complete(self, saved_net: Tuple[CurrentNetData, ...],
                      saved_buf: _TypeBuffer, /) -> None:
        if isinstance(self.__current_net, CurrentNetData):
            raise KnownSimpleAnnError("The net was not appropriately saved!")
        if len(saved_net) != 3:
            raise KnownSimpleAnnError(f"Expected saved_net tuple length 3 got {len(saved_net)}!")
        for elem in saved_net:
            if not isinstance(elem, CurrentNetData):
                raise KnownSimpleAnnError(f"Expected CurrentNetData got {type(elem).__name__}!")
        self.__current_net = saved_net[0]
        self.__buffered_best_net = saved_net[1]
        self.__init_blank_net = saved_net[2]
        self.reload_after_save(saved_buf)

    @final
    def load_tuple_dict_stats(self, data: Tuple[float, Dict, Dict],
                              extra_args: InitContainer, /) -> None:
        self.__current_net = self._create_current_loaded_net(extra_args)
        self.__current_net.fitness = data[0]
        self.__current_net.com.load_state_dict(data[1])
        self.__current_net.com.eval()
        self.__current_net.lego.load_state_dict(data[2])
        self.__current_net.lego.eval()
        self.__buffered_best_net = deepcopy(self.__current_net)
        self.__init_blank_net = deepcopy(self.__current_net)

    @classmethod
    @final
    def load(cls, data: bytes, extra_args: InitContainer, /) -> 'SimpleNetCon':
        if not isinstance(extra_args, InitContainer):
            raise KnownSimpleAnnError(
                f"Expected args to be {InitContainer.__name__} got {type(extra_args).__name__}!"
            )
        loaded_net = rick.loads(data)
        if not isinstance(loaded_net, SimpleNetCon):
            raise KnownSimpleAnnError(
                f"Expected bytes to be {SimpleNetCon.__name__} got {type(loaded_net).__name__}!"
            )
        loaded_tuple = loaded_net.current_net
        if not isinstance(loaded_tuple, tuple):
            raise KnownSimpleAnnError(
                f"Expected tuple got {type(loaded_tuple).__name__}!"
            )
        if len(loaded_tuple) != 3:
            raise KnownSimpleAnnError(
                f"Expected tuple to have 3 elements got {len(loaded_tuple)}!"
            )
        if not (isinstance(loaded_tuple[0], float)
                and isinstance(loaded_tuple[1], dict)
                and isinstance(loaded_tuple[2], dict)):
            raise KnownSimpleAnnError("Received wrong typed tuple!")
        casted_tuple = (
            float(loaded_tuple[0]),
            {**loaded_tuple[1]},
            {**loaded_tuple[2]}
        )
        loaded_net.load_tuple_dict_stats(casted_tuple, extra_args)
        return loaded_net


@final
@dataclass
class _SimpleANNCon:
    test_data: Optional[Tuple[Dataset, ...]] = None
    train_data: Optional[Tuple[Dataset, ...]] = None
    eval_data: Optional[Tuple[Dataset, ...]] = None
    stop_op_fp: Optional[Path] = None
    is_trainable: Tuple[bool, bool] = (True, False)


def _unlink_if_exists(file_p: Path, /) -> None:
    if file_p.exists() and file_p.is_file():
        file_p.unlink()


@final
class DataSetTypes(Enum):
    TRAIN = 'TrainData'
    TEST = 'TestData'
    EVAL = 'EvalData'


def _move_data_to_shared_mem(data_t: Optional[Tuple[Dataset, ...]],
                             smm: SharedMemoryManager, /) -> None:
    if data_t is not None:
        for data in data_t:
            if isinstance(data, DataSetSharedMemoryA):
                data.move_data_to_shared_memory(smm)


class SimpleAnnNet(
    NodeANNDataElemInterface[nn.Module, CurrentNetData, _TypeBuffer, InitContainer],
    abc.ABC
):

    def __init__(self, args: InitNetArgs, /) -> None:
        super().__init__()
        self.__arguments_con = args
        self.__data_container = _SimpleANNCon()
        self.__savable: Optional[
            NetSavable[nn.Module, CurrentNetData, _TypeBuffer, InitContainer]
        ] = None
        self.__net_module: Optional[SimpleNetCon] = None
        self.__data_name = "NotSet"

    @final
    def get_node_name(self) -> str:
        return self.__data_name

    @final
    def set_node_name(self, name: str) -> None:
        self.__data_name = name

    @final
    def _move_data_sets_to_shared_memory(self, smm: Optional[SharedMemoryManager], /) -> None:
        if smm is not None:
            _move_data_to_shared_mem(self.__data_container.train_data, smm)
            _move_data_to_shared_mem(self.__data_container.eval_data, smm)

    @abc.abstractmethod
    def re_read_data(self, data_type: DataSetTypes, /) -> Optional[Tuple[Dataset, ...]]:
        raise NotImplementedError("Abstract method!")

    @abc.abstractmethod
    def check_net_state(self) -> NetGeneralState:
        raise NotImplementedError("Abstract method!")

    @abc.abstractmethod
    def check_init_state(self) -> InitState:
        raise NotImplementedError("Abstract method!")

    @abc.abstractmethod
    def get_truth_fun_id(self) -> str:
        raise NotImplementedError("Abstract method!")

    @final
    def stop_file_it_min(self, it_cnt: int, runt_time_min: int, /) -> bool:
        return (
            it_cnt < self.arguments_con.hyper_optim_wr.stop_iterations
            or not self.arguments_con.hyper_optim_wr.stop_iterations
        ) and (
            self.stop_file is None
            or (self.stop_file.exists() and self.stop_file.is_file())
        ) and (
            runt_time_min < self.arguments_con.hyper_optim_wr.stop_time_min
            or not self.arguments_con.hyper_optim_wr.stop_time_min
        )

    @final
    @property
    def stop_file(self) -> Optional[Path]:
        return self.__data_container.stop_op_fp

    @final
    def stop_file_set(self, file_p: Optional[Path], /) -> None:
        if file_p is not None and file_p.exists() and file_p.is_file():
            self.__data_container.stop_op_fp = file_p

    @final
    @property
    def arguments_con(self) -> InitNetArgs:
        return self.__arguments_con

    @final
    def is_trainable(self) -> bool:
        return self.retrain and not self.random_net

    @final
    @property
    def retrain(self) -> bool:
        return self.__data_container.is_trainable[0]

    @final
    def retrain_set(self, retrain: bool, /) -> None:
        self.__data_container.is_trainable = (retrain, self.__data_container.is_trainable[1])

    @final
    @property
    def random_net(self) -> bool:
        return self.__data_container.is_trainable[1]

    @final
    def random_net_set(self, random_net: bool, /) -> None:
        self.__data_container.is_trainable = (self.__data_container.is_trainable[0], random_net)

    @final
    @property
    def test_data(self) -> Tuple[Dataset, ...]:
        if self.__data_container.test_data is None:
            return ()
        temp_data = self.re_read_data(DataSetTypes.TEST)
        if temp_data is not None:
            self.test_data_set(temp_data)
        return self.__data_container.test_data

    @final
    def test_data_set(self, data: Tuple[Dataset, ...], /) -> None:
        if not (isinstance(data, tuple) and data):
            raise KnownSimpleAnnError("The given test data set was empty")
        self.__data_container.test_data = data

    @final
    @property
    def train_data(self) -> Tuple[Dataset, ...]:
        if self.__data_container.train_data is None:
            return ()
        temp_data = self.re_read_data(DataSetTypes.TRAIN)
        if temp_data is not None:
            self.train_data_set(temp_data)
        return self.__data_container.train_data

    @final
    def train_data_set(self, data: Tuple[Dataset, ...], /) -> None:
        if not (isinstance(data, tuple) and data):
            raise KnownSimpleAnnError("The given train data set was empty")
        self.__data_container.train_data = data

    @final
    @property
    def eval_data(self) -> Tuple[Dataset, ...]:
        if self.__data_container.eval_data is None:
            return ()
        temp_data = self.re_read_data(DataSetTypes.EVAL)
        if temp_data is not None:
            self.eval_data_set(temp_data)
        return self.__data_container.eval_data

    @final
    def eval_data_set(self, data: Tuple[Dataset, ...], /) -> None:
        if not (isinstance(data, tuple) and data):
            raise KnownSimpleAnnError("The given eval data set was empty")
        self.__data_container.eval_data = data

    @final
    @property
    def savable(self) -> \
            Optional[NetSavable[nn.Module, CurrentNetData, _TypeBuffer, InitContainer]]:
        return self.__savable

    @final
    def savable_set(self, savable: NetSavable[
        nn.Module, CurrentNetData, _TypeBuffer, InitContainer
    ], /) -> None:
        self.__savable = savable

    @final
    def get_savable_data(self) -> NetSavable[nn.Module, CurrentNetData, _TypeBuffer, InitContainer]:
        if self.__savable is None:
            raise KnownSimpleAnnError("Net was not initialised!")
        return self.__savable

    @final
    @property
    def net_module(self) -> Optional[SimpleNetCon]:
        return self.__net_module

    @final
    def net_module_set(self, module: SimpleNetCon, /) -> None:
        if self.__net_module is not None:
            raise KnownSimpleAnnError("Net was already initialised!")
        self.__net_module = module

    @final
    def get_savable_net(self) -> SimpleNetCon:
        if self.__net_module is None:
            raise KnownSimpleAnnError("Net was not initialised!")
        return self.__net_module

    @final
    def _update_hyper_run(self, hyper_cont: HyperOptimInterfaceArgs,
                          new_params: Dict[str, HyperOptimReturnElem], /) -> None:
        self.get_savable_net().reset_current_net()
        self._update_hyper(hyper_cont, new_params)

    @final
    def _update_hyper(self, hyper_cont: HyperOptimInterfaceArgs,
                      new_params: Dict[str, HyperOptimReturnElem], /) -> None:
        update_hyper_params(self.get_savable_net(), self.arguments_con, new_params)
        update_hyper_container(self.arguments_con, hyper_cont)

    @final
    def _create_train_interface(self, id_file: ANNTreeIdType,
                                copy: bool, id_mod: str, /) -> TrainerInterfaceArgs:
        if self.arguments_con.net_state.get_kwargs().redraw:
            self.get_savable_net().redraw_current_net()
        if copy:
            buf = self.get_savable_net().remove_before_save()
            new_mod = deepcopy(self.get_savable_net())
            self.get_savable_net().reload_after_save(buf)
        else:
            new_mod = self.get_savable_net()
        new_train_args = TrainerInterfaceArgs(
            module=new_mod,
            input_train=self.train_data,
            input_eval=self.eval_data,
            id_file=deepcopy(id_file),
            dump=self.arguments_con.net_state.get_kwargs().dump,
            cuda=self.arguments_con.net_state.get_kwargs().cuda,
            optimizer=deepcopy(self.arguments_con.optimizer_wr)
            if copy else self.arguments_con.optimizer_wr,
            scheduler=deepcopy(self.arguments_con.scheduler_wr)
            if copy else self.arguments_con.scheduler_wr,
            criterion=deepcopy(self.arguments_con.criterion_wr)
            if copy else self.arguments_con.criterion_wr,
            truth_fun_id=self.get_truth_fun_id(),
            hyper_str=create_hyper_param_str(self.get_node_name(), self.arguments_con)
        )
        if id_mod:
            new_train_args.id_file.add_modifier(id_mod)
        return new_train_args

    @final
    def _create_stop_file(self, id_file: ANNTreeIdType, /) -> Optional[Path]:
        if self.arguments_con.hyper_optim_wr is not None \
                and self.arguments_con.hyper_optim_wr.stop_file is not None \
                and self.arguments_con.hyper_optim_wr.stop_file.exists() \
                and self.arguments_con.hyper_optim_wr.stop_file.is_dir():
            merged_str = \
                f"{id_file.id_merged_str}_{datetime.now().strftime('%d_%m_%Y__%H_%M_%S')}.lock"
            stop_file = self.arguments_con.hyper_optim_wr.stop_file.joinpath(merged_str)
            stop_file.touch()
            atexit.register(_unlink_if_exists, stop_file)
            return stop_file
        return None

    def _get_new_params(self, generator_optim: HGenTA, fixed_params: _TrFitParam,
                        run_cont: _RunningConst, /) -> List[Dict[str, HyperOptimReturnElem]]:
        run_cnt = 0
        l_new_params: List[Dict[str, HyperOptimReturnElem]] = []
        while run_cnt < 10 and not l_new_params:
            run_cnt += 1
            try:
                l_new_params = generator_optim.send(fixed_params)
            except StopIteration:
                run_cont.running = False
                run_cnt = 10
            else:
                run_cont.running = self.stop_file_it_min(run_cont.run_id, run_cont.run_time_min)
        if not l_new_params:
            run_cont.running = False
        return l_new_params

    def _train_single(self, sync_out: SyncStdoutInterface, run_cont: _RunningConst,
                      hyper_cont: HyperOptimInterfaceArgs,
                      id_file: ANNTreeIdType, /) -> Iterable[TrainNNStatsElementType]:
        if self.arguments_con.hyper_optim_wr is None:
            raise KnownSimpleAnnError("Hyper-optimiser is not defined!")
        generator_optim = self.arguments_con.hyper_optim_wr.hyper.hyper_optim(
            sync_out, hyper_cont
        )
        try:
            l_new_params: List[Dict[str, HyperOptimReturnElem]] = next(generator_optim)
        except StopIteration:
            raise KnownSimpleAnnError("Generator could not be started!")
        while run_cont.running:
            tr_fit: _TrFitAl = ([], [])
            trainer_args = []
            for param_id, new_param in enumerate(l_new_params):
                run_cont.hyper_cont_buffer = deepcopy(hyper_cont)
                self.arguments_con.prepare_wr.init_prepare()
                self._update_hyper_run(run_cont.hyper_cont_buffer, new_param)
                yield from self.arguments_con.prepare_wr.prepare.run_train(
                    sync_out, PrepareInterfaceArgs(
                        trainer=deepcopy(self.arguments_con.trainer_wr.trainer),
                        trainer_args=self._create_train_interface(
                            id_file, False, str(run_cont.run_id + param_id)
                        )
                    )
                )
                re_copy_model(
                    self.arguments_con.prepare_wr.prepare.p_state_dict,
                    self.get_savable_net().get_net_com
                )
                tr_fit_res = self.arguments_con.prepare_wr.prepare.fitness
                tr_fit[0].append((tr_fit_res[0], _create_hyper_params(run_cont.hyper_cont_buffer)))
                tr_fit[1].append(tr_fit_res[1])
                trainer_args.append(run_cont.hyper_cont_buffer)
                self.get_savable_net().update_current_net(tr_fit_res[0])
            run_cont.fit_plotter.update_fitness(tr_fit, trainer_args)
            self._update_hyper(hyper_cont, run_cont.fit_plotter.bets_fit_h_param[1])
            l_new_params = self._get_new_params(generator_optim, tr_fit[0], run_cont)
            yield from run_cont.fit_plotter.plot(
                not run_cont.running,
                self.arguments_con.net_state.get_kwargs().dump,
                run_cont.run_id
            )
            run_cont.run_id += len(tr_fit[0])

    def _train_parallel(self, sync_out: SyncStdoutInterface, run_cont: _RunningConst,
                        hyper_cont: HyperOptimInterfaceArgs,
                        id_file: ANNTreeIdType, /) -> Iterable[TrainNNStatsElementType]:
        if self.arguments_con.hyper_optim_wr is None:
            raise KnownSimpleAnnError("Hyper-optimiser is not defined!")
        run_cont.init()
        run_cont.start()
        self._move_data_sets_to_shared_memory(run_cont.smm)
        generator_optim = self.arguments_con.hyper_optim_wr.hyper.hyper_optim(
            sync_out, hyper_cont
        )
        try:
            l_new_params: List[Dict[str, HyperOptimReturnElem]] = next(generator_optim)
        except StopIteration:
            raise KnownSimpleAnnError("Generator could not be started!")
        while run_cont.running:
            tr_fit: _TrFitAl = ([], [])
            trainer_args: _TrainArgs = []
            for param_id, new_param in enumerate(l_new_params):
                run_cont.hyper_cont_buffer = deepcopy(hyper_cont)
                self._update_hyper_run(run_cont.hyper_cont_buffer, new_param)
                trainer_args.append((
                    self._create_train_interface(
                        id_file, True, str(run_cont.run_id + param_id)
                    ), run_cont.hyper_cont_buffer,
                    deepcopy(self.arguments_con.trainer_wr.trainer),
                    deepcopy(self.arguments_con.prepare_wr.prepare)
                ))

            yield from _optimise_in_parallel(
                sync_out, self.arguments_con.net_state.get_kwargs().process,
                trainer_args, tr_fit
            )
            for erg_index, erg_tuple in enumerate(tr_fit[0]):
                self.get_savable_net().merge_net_model(trainer_args[erg_index][0].module)
                self.get_savable_net().update_current_net(erg_tuple[0])

            run_cont.fit_plotter.update_fitness(tr_fit, [tr_ar[1] for tr_ar in trainer_args])
            self._update_hyper(hyper_cont, run_cont.fit_plotter.bets_fit_h_param[1])
            l_new_params = self._get_new_params(generator_optim, tr_fit[0], run_cont)
            yield from run_cont.fit_plotter.plot(
                not run_cont.running,
                self.arguments_con.net_state.get_kwargs().dump,
                run_cont.run_id
            )
            run_cont.run_id += len(tr_fit[0])
        run_cont.shutdown()

    def train_net(self, id_file: ANNTreeIdType, sync_out: SyncStdoutInterface, /) -> \
            Iterable[TrainNNStatsElementType]:
        if self.is_trainable():
            hyper_cont = create_hyper_arguments(self.arguments_con)
            if self.arguments_con.hyper_optim_wr is not None:
                self.stop_file_set(self._create_stop_file(id_file))
                run_cont = _RunningConst(
                    fit_plotter=_SimpleTrainDataP(hyper_cont, deepcopy(id_file)),
                    hyper_cont_buffer=deepcopy(hyper_cont),
                    init_time=time.time()
                )
                if self.check_net_state().get_kwargs().process > 1:
                    yield from self._train_parallel(sync_out, run_cont, hyper_cont, id_file)
                else:
                    yield from self._train_single(sync_out, run_cont, hyper_cont, id_file)
            else:
                update_hyper_params(
                    self.get_savable_net(), self.arguments_con, _create_hyper_params(hyper_cont)
                )
                yield from self.arguments_con.prepare_wr.prepare.run_train(
                    sync_out, PrepareInterfaceArgs(
                        trainer=self.arguments_con.trainer_wr.trainer,
                        trainer_args=self._create_train_interface(id_file, False, "")
                    )
                )
                re_copy_model(
                    self.arguments_con.prepare_wr.prepare.p_state_dict,
                    self.get_savable_net().get_net_com
                )
                self.get_savable_net().update_current_net(
                    self.arguments_con.prepare_wr.prepare.fitness[0]
                )
            self.get_savable_net().set_best_net()

    def test_net(self, id_file: ANNTreeIdType, sync_out: SyncStdoutInterface, /) \
            -> Tuple[TestNNStatsElementType, ...]:
        module_net = self.get_savable_net()
        return self.arguments_con.tester_wr.tester.test(
            sync_out,
            TesterInterfaceArgs(
                module=module_net,
                input_test=self.test_data,
                id_file=deepcopy(id_file),
                cuda=self.arguments_con.net_state.get_kwargs().cuda,
                truth_fun_id=self.get_truth_fun_id()
            )
        )

    @final
    def finalize(self) -> None:
        self.__data_container.eval_data = None
        self.__data_container.train_data = None
        self.__data_container.test_data = None
        self.__data_container.stop_op_fp = None
