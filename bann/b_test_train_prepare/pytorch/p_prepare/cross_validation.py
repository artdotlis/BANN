# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import multiprocessing
import queue
from copy import deepcopy
from dataclasses import dataclass, field
from statistics import variance, mean
from threading import Thread, Lock
from typing import Tuple, Optional, List, Iterable, Union, final, Dict

import pickle as rick

import numpy as np  # type: ignore
from torch.utils.data import ConcatDataset, Dataset

from bann.b_data_functions.pytorch.p_gen_fun import re_copy_model
from bann.b_container.errors.custom_erors import KnownPrepareError
from bann.b_container.functions.dict_str_repr import dict_string_repr
from bann.b_container.states.framework.general.prepare.cross_validate import CrossValTState
from bann.b_container.states.framework.interface.prepare_state import PrepareState
from bann.b_frameworks.pytorch.net_model_interface import NetModelInterface
from bann.b_test_train_prepare.pytorch.functions.dataset_splitter import split_data_set, \
    test_k_folds_data_sets
from bann.b_test_train_prepare.pytorch.p_train.net_trainer_1thread import add_mod_str
from bann.b_test_train_prepare.pytorch.prepare_interface import PrepareInterface, \
    PrepareInterfaceArgs
from bann.b_test_train_prepare.pytorch.trainer_interface import TrainerInterface, \
    TrainerInterfaceArgs

from pan.public.constants.train_net_stats_constants import TrainNNStatsElementType
from rewowr.public.functions.decorator_functions import rewowr_process_wrapper

from rewowr.public.functions.syncout_dep_functions import logger_print_to_console
from rewowr.public.functions.worker_ctx import get_worker_ctx
from rewowr.public.interfaces.logger_interface import SyncStdoutInterface


def _print_to_logger(sync_out: SyncStdoutInterface, trainer_stats: CrossValTState, /) -> None:
    output_string = f"The arguments given to CrossValidation:\n"
    output_string += f"The CrossValTState:"
    output_string += f"\n\t{dict_string_repr(trainer_stats.get_kwargs().__dict__)}\n"
    logger_print_to_console(sync_out, output_string)


@final
@dataclass
class _RunCont:
    best_module: NetModelInterface
    best_fit: float = float('inf')
    saved_last: List[TrainNNStatsElementType] = field(default_factory=lambda: [])
    fit_list: List[Tuple[float, float]] = field(default_factory=lambda: [])
    last_h_param: str = ""


@final
@dataclass
class _PTCon:
    yield_queue: multiprocessing.Queue
    worker_run: bool
    worker_lock: Lock


@final
@dataclass
class _PCon:
    ctx: multiprocessing.context.SpawnContext
    p_list: List[multiprocessing.context.SpawnProcess] = field(default_factory=lambda: [])
    p_finished_id: List[int] = field(default_factory=lambda: [])
    p_module_list: List[NetModelInterface] = field(default_factory=lambda: [])
    p_pointer_finished: int = 0
    p_pointer_started: int = 0


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
    index: int


def _thread_run_worker(status_con: _PTCon,
                       yield_res_l: List[Union[_YieldQueueElWr, _ResQueueEl]], /) -> None:
    while status_con.worker_run:
        try:
            erg_buf = status_con.yield_queue.get(True, 2)
        except queue.Empty:
            pass
        else:
            if isinstance(erg_buf, (_YieldQueueElWr, _ResQueueEl)):
                with status_con.worker_lock:
                    yield_res_l.append(erg_buf)
            else:
                raise KnownPrepareError(
                    f"Expected {_ResQueueEl.__name__}  or {_YieldQueueElWr.__name__} "
                    + f"got {type(erg_buf).__name__}"
                )


def _process_run_fun(index_id: int, trainer: TrainerInterface,
                     yield_queue: multiprocessing.Queue,
                     sync_out: SyncStdoutInterface, args: TrainerInterfaceArgs, /) -> None:
    for erg in trainer.train(sync_out, args):
        erg.info.title = f"{erg.info.title} (CrossValidation)"
        erg.info.id_file.add_modifier(add_mod_str(erg.info.name_series))
        yield_queue.put(_YieldQueueElWr(data=erg, index=index_id))
    yield_queue.put(_ResQueueEl(
        fit_0=trainer.fitness[0], fit_1=trainer.fitness[1], in_id=index_id,
        state_dict=rick.dumps(trainer.tr_state_dict)
    ))


@final
class RunSingleThreadTP2:
    def __init__(self, sync_out: SyncStdoutInterface, running_con: _RunCont,
                 train_state: CrossValTState, /) -> None:
        super().__init__()
        self.__sync_out = sync_out
        self.__running_con = running_con
        self.__train_state = train_state
        ctx = get_worker_ctx()
        if not isinstance(ctx, multiprocessing.context.SpawnContext):
            raise KnownPrepareError(f"Expected SpawnContext got {type(ctx).__name__}")
        self.__p_con: _PCon = _PCon(ctx)
        self.__yield_res_l: List[Union[_YieldQueueElWr, _ResQueueEl]] = []
        self.__ptc = _PTCon(ctx.Queue(), True, Lock())
        self.__worker_t = Thread(
            target=_thread_run_worker, args=(self.__ptc, self.__yield_res_l)
        )
        self.__worker_t.start()

    def join(self) -> Iterable[TrainNNStatsElementType]:
        if self.__p_con.p_pointer_started != len(self.__p_con.p_list):
            KnownPrepareError("Not all processes were started!")
        running = True
        while running:
            running = False
            for process_e in self.__p_con.p_list:
                process_e.join(2)
                if process_e.is_alive():
                    running = True
            yield from self.__analyse_out()
        if self.__p_con.p_pointer_started != self.__p_con.p_pointer_finished:
            KnownPrepareError("Not all processes were finished!")
        self.__ptc.worker_run = False
        self.__worker_t.join()

    def __analyse_out(self) -> Iterable[TrainNNStatsElementType]:
        with self.__ptc.worker_lock:
            copy_out = list(self.__yield_res_l)
            self.__yield_res_l.clear()
        for res_elem in copy_out:
            if isinstance(res_elem, _YieldQueueElWr):
                if not self.__running_con.last_h_param and res_elem.data.hyper_param:
                    self.__running_con.last_h_param = res_elem.data.info.name_series
                if self.__running_con.last_h_param != res_elem.data.info.name_series:
                    res_elem.data.hyper_param = ""
                res_elem.data.info.name_series = \
                    f"{res_elem.data.info.name_series}_id_{res_elem.index}"
                if res_elem.data.last:
                    self.__running_con.saved_last.append(res_elem.data)
                else:
                    yield res_elem.data
            else:
                if res_elem.in_id in self.__p_con.p_finished_id:
                    raise KnownPrepareError("Duplicated process id found!")
                self.__p_con.p_finished_id.append(res_elem.in_id)
                self.__p_con.p_pointer_finished += 1
                if res_elem.fit_0 <= self.__running_con.best_fit:
                    re_copy_model(
                        rick.loads(res_elem.state_dict),
                        self.__p_con.p_module_list[res_elem.in_id].get_net_com
                    )
                    self.__running_con.best_module = self.__p_con.p_module_list[res_elem.in_id]
                    self.__running_con.best_fit = res_elem.fit_0
                self.__running_con.fit_list.append((res_elem.fit_0, res_elem.fit_1))

    def send_to_thread(self, trainer: TrainerInterface, args_copy: TrainerInterfaceArgs,
                       run_id: int, k_fold: int, /) -> Iterable[TrainNNStatsElementType]:
        trainer_1_thread = deepcopy(trainer)
        ctx = get_worker_ctx()
        if run_id * self.__train_state.get_kwargs().k_folds + k_fold \
                != len(self.__p_con.p_list):
            raise KnownPrepareError("Mismatching process id found!")
        if k_fold or run_id:
            trainer_1_thread.deactivate_log()
        self.__p_con.p_list.append(ctx.Process(
            target=rewowr_process_wrapper,
            args=(
                self.__sync_out, 'CrossValTrainer', _process_run_fun,
                (run_id * self.__train_state.get_kwargs().k_folds + k_fold,
                 trainer_1_thread, self.__ptc.yield_queue, self.__sync_out, args_copy))
        ))
        self.__p_con.p_module_list.append(args_copy.module)
        to_start = True
        while to_start:
            if self.__sync_out.error_occurred():
                raise KnownPrepareError("Found error on sync queue!")
            elif self.__train_state.get_kwargs().cross_p \
                    > self.__p_con.p_pointer_started - self.__p_con.p_pointer_finished:
                self.__p_con.p_list[self.__p_con.p_pointer_started].start()
                self.__p_con.p_pointer_started += 1
                to_start = False
            else:
                yield from self.__analyse_out()


def _run_trainer_p_1(sync_out: SyncStdoutInterface, args_copy: TrainerInterfaceArgs,
                     running_cot: _RunCont, trainer: TrainerInterface,
                     run_k: Tuple[int, int], /) -> Iterable[TrainNNStatsElementType]:
    run, k_fold = run_k
    trainer_1_thread = deepcopy(trainer)
    if k_fold or run:
        trainer_1_thread.deactivate_log()
    for erg in trainer_1_thread.train(sync_out, args_copy):
        if not running_cot.last_h_param and erg.hyper_param:
            running_cot.last_h_param = erg.info.name_series
        if running_cot.last_h_param != erg.info.name_series:
            erg.hyper_param = ""
        erg.info.title = f"{erg.info.title} (CrossValidation)"
        erg.info.id_file.add_modifier(add_mod_str(erg.info.name_series))
        erg.info.name_series = f"{erg.info.name_series}_f_{k_fold}_r_{run}"
        if erg.last:
            running_cot.saved_last.append(erg)
        else:
            yield erg
    if trainer_1_thread.fitness[0] <= running_cot.best_fit:
        re_copy_model(trainer_1_thread.tr_state_dict, args_copy.module.get_net_com)
        running_cot.best_module = args_copy.module
        running_cot.best_fit = trainer_1_thread.fitness[0]

    running_cot.fit_list.append(trainer_1_thread.fitness)


class CrossValidate(PrepareInterface):

    def __init__(self) -> None:
        super().__init__()
        self.__fitness: Optional[Tuple[float, float]] = None
        self.__pr_state: Optional[CrossValTState] = None
        self.__state_dict: Optional[Dict] = None

    @property
    def p_state_dict(self) -> Dict:
        if self.__state_dict is None:
            raise KnownPrepareError("Prepare Trainig-run was not finished")
        return self.__state_dict

    @property
    def fitness(self) -> Tuple[float, float]:
        if self.__fitness is None:
            raise KnownPrepareError("The trainer was not started!")
        return self.__fitness

    def _set_fitness(self, eval_loss_list: List[float], eval_truth_list: List[float], /) -> None:
        if len(eval_truth_list) > 1:
            eval_truth_local = np.mean(eval_truth_list)
        else:
            eval_truth_local = eval_loss_list[0]
        valid_loss = float('inf') not in eval_loss_list
        eval_loss_local = eval_loss_list[0] if len(eval_loss_list) < 2 else min(eval_loss_list)
        if self.pr_state.get_kwargs().bias_var_f > 0:
            if valid_loss and len(eval_loss_list) > 1:
                eval_loss_local = min(eval_loss_list)
                eval_loss_local += (
                    self.pr_state.get_kwargs().bias_var_f * variance(eval_loss_list)
                    + self.pr_state.get_kwargs().bias_var_f * abs(
                        (max(eval_loss_list) - min(eval_loss_list)) * 0.5
                        - mean(eval_loss_list)
                    )
                )
            else:
                eval_loss_local = float('inf')
        self.__fitness = (eval_loss_local, eval_truth_local)

    @staticmethod
    def _create_args_copy(args: TrainerInterfaceArgs, train_data: Tuple[Dataset, ...],
                          eval_data: Tuple[Dataset, ...], /) \
            -> TrainerInterfaceArgs:
        new_train_args = TrainerInterfaceArgs(
            module=deepcopy(args.module),
            input_train=train_data,
            input_eval=eval_data,
            id_file=deepcopy(args.id_file),
            dump=args.dump,
            cuda=args.cuda,
            optimizer=deepcopy(args.optimizer),
            scheduler=deepcopy(args.scheduler),
            criterion=deepcopy(args.criterion),
            truth_fun_id=args.truth_fun_id,
            hyper_str=args.hyper_str
        )
        return new_train_args

    def run_train(self, sync_out: SyncStdoutInterface,
                  args: PrepareInterfaceArgs, /) -> Iterable[TrainNNStatsElementType]:
        self.__state_dict = None
        if not args.trainer_args.input_train:
            raise KnownPrepareError("Received empty dataset!")
        if self.pr_state.get_kwargs().k_folds < 2:
            self.pr_state.get_kwargs().k_folds = 2
        if self.pr_state.get_kwargs().n_repeats < 1:
            self.pr_state.get_kwargs().n_repeats = 1

        running_cot = _RunCont(best_module=args.trainer_args.module)
        merged_data_set: Tuple[Dataset, ...] = tuple(
            ConcatDataset([data_set, args.trainer_args.input_eval[data_index]])
            if data_index < len(args.trainer_args.input_eval) and self.pr_state.get_kwargs().eval_on
            else ConcatDataset([data_set])
            for data_index, data_set in enumerate(args.trainer_args.input_train)
        )
        test_k_folds_data_sets(self.pr_state.get_kwargs().k_folds, merged_data_set)
        _print_to_logger(sync_out, self.pr_state)
        trainer_fit_con = None
        if self.pr_state.get_kwargs().cross_p > 1:
            trainer_fit_con = RunSingleThreadTP2(sync_out, running_cot, self.pr_state)
        for run in range(self.pr_state.get_kwargs().n_repeats):
            merged_subsets = [
                split_data_set(self.pr_state.get_kwargs().k_folds, data_set)
                for data_set in merged_data_set
            ]

            for k_fold in range(self.pr_state.get_kwargs().k_folds):
                args_copy = self._create_args_copy(
                    args.trainer_args, tuple(
                        ConcatDataset([*data_sets[0: k_fold], *data_sets[k_fold + 1:]])
                        for data_sets in merged_subsets
                    ),
                    tuple(data_sets[k_fold] for data_sets in merged_subsets)
                )
                if trainer_fit_con is not None:
                    yield from trainer_fit_con.send_to_thread(args.trainer, args_copy, run, k_fold)
                else:
                    yield from _run_trainer_p_1(
                        sync_out, args_copy, running_cot, args.trainer, (run, k_fold)
                    )
        if trainer_fit_con is not None:
            yield from trainer_fit_con.join()
        yield from running_cot.saved_last
        self._set_fitness(
            [tup[0] for tup in running_cot.fit_list],
            [tup[1] for tup in running_cot.fit_list]
        )
        self.__state_dict = running_cot.best_module.get_net_com.state_dict()

    @property
    def pr_state(self) -> CrossValTState:
        if self.__pr_state is None or not isinstance(self.__pr_state, CrossValTState):
            raise KnownPrepareError("Train state was not set properly!")
        return self.__pr_state

    def set_prepare_state(self, state: PrepareState, /) -> None:
        if not isinstance(state, CrossValTState):
            raise KnownPrepareError(
                f"Expected type {CrossValTState.__name__} got {type(state).__name__}"
            )
        self.__pr_state = state
