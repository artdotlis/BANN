# -*- coding: utf-8 -*-
"""Was inspired by https://github.com/pytorch/examples/tree/master/mnist_hogwild

LICENSE (BSD 3-Clause License): see extra_licenses/LICENSE_P_EXAMPLES

.. moduleauthor:: Artur Lissin
"""
import math
import multiprocessing

import queue
from copy import deepcopy
from dataclasses import dataclass
from typing import Iterable, Optional, Callable, List, Dict, Union, Tuple, Final, final

from torch import nn
import torch
import torch.multiprocessing as torch_mp
from torch.utils.data import DataLoader, Dataset

from bann.b_container.constants.file_names import TrainSubStrSuf
from bann.b_container.states.framework.pytorch.optim_param import OptimAlgWr
from bann.b_test_train_prepare.pytorch.p_train.functions.p_train_gen_fun import PQueueTupleErg, \
    tr_create_dict_id_queue, p_train_epoch_gen, PTrainEpochFun, PTrainEpochFunReturn, \
    p_train_epoch_fun, t_print_to_logger, SimpleSGDOptim
from bann.b_container.states.framework.interface.train_state import TrainState
from bann.b_test_train_prepare.pytorch.trainer_interface import TrainerInterfaceArgs, \
    TrainerInterface
from bann.b_container.states.framework.pytorch.p_train.p_train_hogwild import HogwildTState
from bann.b_test_train_prepare.errors.custom_errors import KnownTrainerError
from bann.b_pan_integration.framwork_key_lib import FrameworkKeyLib
from bann.b_container.functions.dict_str_repr import dict_string_repr
from pan.public.constants.train_net_stats_constants import TrainNNStatsElementType, \
    create_train_net_stats_function, TrainNNStatsElemInfo, TrainReturnFiller, \
    TrainNNStatsElementFiller

from rewowr.public.functions.decorator_functions import rewowr_process_wrapper
from rewowr.public.functions.syncout_dep_functions import logger_print_to_console
from rewowr.public.functions.worker_ctx import get_worker_ctx
from rewowr.public.interfaces.logger_interface import SyncStdoutInterface

_FRAMEWORK: Final[str] = FrameworkKeyLib.PYTORCH.value


@final
@dataclass
class _TrainWrArgs:
    model: nn.Module
    args: TrainerInterfaceArgs
    state: HogwildTState
    cuda: bool
    device: torch.device
    queue: torch_mp.Queue


_TRAIN_LOSS_STR: Final[str] = 'Train_Loss'
_EVAL_LOSS_STR: Final[str] = 'Eval_Loss'
_EVAL_TR_V_STR: Final[str] = 'Eval_Truth'
_TRAIN_TR_V_STR: Final[str] = 'Train_Truth'


_TrfT: Final = Callable[
    [Tuple[DataLoader, ...], Tuple[DataLoader, ...], int, int], PTrainEpochFunReturn
]


def _split_data(split: bool, pr_index: int, pr_cnt: int,
                data: Tuple[Dataset, ...], m_batch: Tuple[int, ...], /) \
        -> Iterable[Tuple[Dataset, int]]:
    batch_size = len(m_batch)
    for data_i, data_e in enumerate(data):
        if not split or pr_cnt <= 1:
            yield data_e, m_batch[data_i % batch_size]
        elif batch_size < pr_cnt and pr_index % batch_size == data_i:
            yield data_e, m_batch[data_i % batch_size]
        elif data_i % pr_cnt == pr_index:
            yield data_e, m_batch[data_i % batch_size]


def _train_fun(pr_index: int, stop_at_loss: float,
               container: _TrainWrArgs, train_epoch: _TrfT, /) -> None:
    train_loader = tuple(DataLoader(
        tr_in[0], batch_size=tr_in[1],
        shuffle=container.state.get_kwargs().shuffle,
        pin_memory=container.cuda,
        num_workers=container.state.get_kwargs().num_workers,
        drop_last=container.state.get_kwargs().drop_last
    ) for tr_in in _split_data(
        container.state.get_kwargs().split_data, pr_index,
        container.state.get_kwargs().tr_worker,
        container.args.input_train,
        container.state.get_kwargs().batch_size
    ))
    test_loader = tuple(DataLoader(
        te_in[0], batch_size=te_in[1],
        shuffle=False,
        pin_memory=container.cuda,
        num_workers=container.state.get_kwargs().num_workers,
        drop_last=False
    ) for te_in in _split_data(
        False, pr_index, container.state.get_kwargs().tr_worker,
        container.args.input_eval, container.state.get_kwargs().batch_size
    ))
    if container.state.get_kwargs().num_workers > 0:
        for train_test in [test_loader, train_loader]:
            for tr_te_el in train_test:
                tr_te_el.multiprocessing_context = get_worker_ctx()
    max_epoch = int(
        container.state.get_kwargs().epoch_size / container.state.get_kwargs().tr_worker
    )
    max_epoch = max_epoch if max_epoch > 4 else 4
    report_size = container.state.get_kwargs().report_size
    report_size = int(max_epoch / 2) if report_size >= int(max_epoch / 2) else report_size
    epoch = 1
    running_loss: float = float('inf')
    report_cnt = 0
    while epoch < max_epoch + 1:
        int_max = int(epoch + report_size)
        int_max = max_epoch + 1 if int_max > max_epoch + 1 else int_max
        erg_loss: List[PTrainEpochFunReturn] = list(
            p_train_epoch_gen(train_loader, test_loader, epoch, int_max, train_epoch)
        )
        if erg_loss and not math.isnan(erg_loss[-1].running_loss):
            running_loss = erg_loss[-1].running_loss
        last_run = int_max >= max_epoch + 1 or (
            0 < running_loss < stop_at_loss and report_cnt >= 2
        )
        container.queue.put(PQueueTupleErg(
            _TRAIN_LOSS_STR, epoch, [
                elem.test_train.test_loss
                if not (math.isnan(elem.test_train.test_loss)
                        or math.isinf(elem.test_train.test_loss))
                else -1.0
                for elem in erg_loss
            ], last_run
        ))
        container.queue.put(PQueueTupleErg(
            _EVAL_LOSS_STR, epoch, [
                elem.test_eval.test_loss
                if not (math.isnan(elem.test_eval.test_loss)
                        or math.isinf(elem.test_eval.test_loss))
                else -1.0
                for elem in erg_loss
            ], last_run
        ))
        container.queue.put(PQueueTupleErg(
            _TRAIN_TR_V_STR, epoch, [elem.test_train.truth_v for elem in erg_loss], last_run
        ))
        container.queue.put(PQueueTupleErg(
            _EVAL_TR_V_STR, epoch, [elem.test_eval.truth_v for elem in erg_loss], last_run
        ))
        epoch = int_max
        if 0 < running_loss < stop_at_loss and report_cnt >= 2:
            epoch = max_epoch + 1
        report_cnt += 1

    container.queue.put(tuple())


def _train_wrapper(pr_index: int, container: _TrainWrArgs,
                   sync_out: SyncStdoutInterface, /) -> None:
    if container.state.get_kwargs().torch_thread:
        torch.set_num_threads(container.state.get_kwargs().torch_thread)
        logger_print_to_console(sync_out, f"Torch threads set to {torch.get_num_threads()}\n")

    if container.args.criterion is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = container.args.criterion.criterion
    if container.args.optimizer is None:
        optimizer: Union[OptimAlgWr, SimpleSGDOptim] = SimpleSGDOptim(container.model)
    else:
        optimizer = container.args.optimizer

    def train_epoch(data_loader_train: Tuple[DataLoader, ...],
                    data_loader_test: Tuple[DataLoader, ...],
                    epoch: int, batch_cnt: int) -> PTrainEpochFunReturn:
        return p_train_epoch_fun(PTrainEpochFun(
            data_loader_train=data_loader_train, data_loader_test=data_loader_test,
            epoch=epoch, max_batch_cnt=batch_cnt,
            model=container.model, device=container.device, optimizer=optimizer,
            criterion=criterion, scheduler_wrapper=container.args.scheduler,
            truth_fun_id=container.args.truth_fun_id,
            train_ll=container.state.get_kwargs().train_ll,
            test_ll=container.state.get_kwargs().eval_ll,
            shuffle=container.state.get_kwargs().shuffle
        ))

    _train_fun(pr_index, container.state.get_kwargs().end_criterion, container, train_epoch)


def _check_consistency(data: List[PQueueTupleErg], /) -> PQueueTupleErg:
    epoch_number = data[0].epoch
    series_name = data[0].series
    last_val = data[0].last
    y_cords_len = len(data[0].y_cords)
    y_cords_all = []

    for data_elem in data:
        if last_val != data_elem.last:
            raise KnownTrainerError("The last parameter was not set properly!")
        if len(data_elem.y_cords) != y_cords_len:
            raise KnownTrainerError("The y_cord parameter have different length!")
        y_cords_all.append(data_elem.y_cords)

    return PQueueTupleErg(
        epoch=epoch_number,
        series=series_name,
        last=last_val,
        y_cords=[
            sum(y_cord[y_cords_id] for y_cord in y_cords_all) / len(y_cords_all)
            if min(y_cord[y_cords_id] for y_cord in y_cords_all) >= 0
            else min(y_cord[y_cords_id] for y_cord in y_cords_all)
            for y_cords_id in range(y_cords_len)
        ]
    )


@final
@dataclass
class _ExtraData:
    use_cuda: bool
    model: nn.Module
    tr_size: int


def _print_to_logger(sync_out: SyncStdoutInterface, trainer_stats: HogwildTState,
                     args_trainer: TrainerInterfaceArgs, extra_args: _ExtraData, /) -> None:
    output_string = f"The arguments given to HogwildTrainer:\n"
    output_string += f"Training on data with the size {extra_args.tr_size}\n"
    output_string += f"The HogwildTState:"
    output_string += f"\n\t{dict_string_repr(trainer_stats.get_kwargs().__dict__)}\n"
    output_string += t_print_to_logger(args_trainer, extra_args.use_cuda, extra_args.model)
    logger_print_to_console(sync_out, output_string)


def _get_spawn_context() -> multiprocessing.context.SpawnContext:
    context = torch_mp.get_context('spawn')
    if not isinstance(context, multiprocessing.context.SpawnContext):
        raise KnownTrainerError(f"Expected SpawnContext got {type(context).__name__}")
    return context


@final
@dataclass
class _HogwildTrainerCon:
    finished_num: int
    erg_dict: Dict[str, List[PQueueTupleErg]]
    eval_truth: float
    eval_loss: float
    train_loss: float


@final
@dataclass
class _HogwildTrainerGenCon:
    use_cuda: bool
    torch_device: torch.device
    model: nn.Module
    ctx: multiprocessing.context.SpawnContext


@final
class HogwildTrainer(TrainerInterface):

    def __init__(self) -> None:
        super().__init__()
        self.__fitness: Optional[Tuple[float, float]] = None
        self.__train_state: Optional[HogwildTState] = None
        self.__log = True
        self.__state_dict: Optional[Dict] = None

    @property
    def tr_state_dict(self) -> Dict:
        if self.__state_dict is None:
            raise KnownTrainerError("Training was not finished")
        return self.__state_dict

    def deactivate_log(self) -> None:
        self.__log = False

    @property
    def train_state(self) -> HogwildTState:
        if self.__train_state is None or not isinstance(self.__train_state, HogwildTState):
            raise KnownTrainerError("Train state was not set properly!")
        return self.__train_state

    def set_train_state(self, state: TrainState, /) -> None:
        if not isinstance(state, HogwildTState):
            raise KnownTrainerError(
                f"Expected type {HogwildTState.__name__} got {type(state).__name__}"
            )
        self.__train_state = state

    @property
    def fitness(self) -> Tuple[float, float]:
        if self.__fitness is None:
            raise KnownTrainerError("The Trainer was not started!")
        return self.__fitness

    @staticmethod
    def _get_from_queue(results_queue: torch_mp.Queue, sync_out: SyncStdoutInterface,
                        finished_num: int, /) -> Union[Tuple, PQueueTupleErg]:
        while True:
            if sync_out.error_occurred():
                raise KnownTrainerError("Found error on sync queue!")
            try:
                erg = results_queue.get(True, 10)
            except queue.Empty:
                if finished_num == 1:
                    return tuple()
            else:
                if not isinstance(erg, (tuple, PQueueTupleErg)):
                    raise KnownTrainerError(
                        f"Received strange answer to queue {type(erg).__name__}"
                    )
                return erg

    def _set_fitness(self, eval_loss: float, train_loss: float, eval_truth: float, /) -> None:
        eval_truth_local = -1.0
        eval_loss_local = float('inf')
        if not math.isnan(eval_truth) and 0 <= eval_truth < float('inf'):
            eval_truth_local = eval_truth
        if not math.isnan(eval_loss) and 0 < eval_loss < float('inf'):
            if not math.isnan(train_loss) and 0 < train_loss < float('inf'):
                eval_loss_local = eval_loss + abs(eval_loss - train_loss) \
                                  * self.train_state.get_kwargs().over_fit
            elif self.train_state.get_kwargs().over_fit == 0:
                eval_loss_local = eval_loss

        self.__fitness = (eval_loss_local, eval_truth_local)

    @staticmethod
    def _change_container(cont: _HogwildTrainerCon, fixed_erg: PQueueTupleErg, /) -> None:
        if _EVAL_TR_V_STR == fixed_erg.series:
            cont.eval_truth = fixed_erg.y_cords[-1]
        if _EVAL_LOSS_STR == fixed_erg.series:
            cont.eval_loss = fixed_erg.y_cords[-1]
        if _TRAIN_LOSS_STR == fixed_erg.series:
            cont.train_loss = fixed_erg.y_cords[-1]

    def train(self, sync_out: SyncStdoutInterface,
              args: TrainerInterfaceArgs, /) -> Iterable[TrainNNStatsElementType]:
        self.__state_dict = None
        if not (args.input_train and args.input_eval):
            raise KnownTrainerError("Received empty dataset!")
        gen_cont = _HogwildTrainerGenCon(
            use_cuda=args.cuda and torch.cuda.is_available(),
            torch_device=torch.device("cpu"),
            model=args.module.get_net_com,
            ctx=_get_spawn_context()
        )
        gen_cont.torch_device = torch.device("cuda" if gen_cont.use_cuda else "cpu")
        gen_cont.model = args.module.get_net_com.to(gen_cont.torch_device)
        gen_cont.model.share_memory()
        if self.__log:
            _print_to_logger(sync_out, self.train_state, args, _ExtraData(
                use_cuda=gen_cont.use_cuda, model=gen_cont.model,
                tr_size=sum(len(tr_in) for tr_in in args.input_train)
            ))

        results_queue: torch.multiprocessing.Queue = gen_cont.ctx.Queue()
        processes = []
        for pr_i in range(self.train_state.get_kwargs().tr_worker):
            pr_el = gen_cont.ctx.Process(
                target=rewowr_process_wrapper, args=(
                    sync_out, "HogwildTrainerProcess", _train_wrapper,
                    (pr_i, _TrainWrArgs(
                        model=gen_cont.model, args=args, state=self.train_state,
                        cuda=gen_cont.use_cuda, device=gen_cont.torch_device,
                        queue=results_queue
                    ), sync_out)
                )
            )
            processes.append(pr_el)
            pr_el.start()
        cont = _HogwildTrainerCon(
            finished_num=self.train_state.get_kwargs().tr_worker + 1, erg_dict={},
            eval_truth=-1., eval_loss=-1., train_loss=-1.
        )
        net_stats_creator_dict: Dict[str, TrainReturnFiller] = {}
        while cont.finished_num > 0:
            erg = self._get_from_queue(results_queue, sync_out, cont.finished_num)
            list_erg: Optional[List[PQueueTupleErg]] = None
            if not erg:
                cont.finished_num -= 1 if cont.finished_num >= 1 else 0
            elif isinstance(erg, PQueueTupleErg):
                list_erg = cont.erg_dict.setdefault(tr_create_dict_id_queue(erg), [])
                list_erg.append(erg)
            else:
                raise KnownTrainerError("This should never happen!")

            if isinstance(erg, PQueueTupleErg) and list_erg is not None \
                    and len(list_erg) == self.train_state.get_kwargs().tr_worker:
                fixed_erg = _check_consistency(list_erg)
                if self.train_state.get_kwargs().plot_data \
                        or self.train_state.get_kwargs().write_data:
                    net_stats_creator = net_stats_creator_dict.setdefault(
                        fixed_erg.series, create_train_net_stats_function()
                    )
                    id_ne_file = deepcopy(args.id_file)
                    if fixed_erg.series not in [_EVAL_LOSS_STR, _TRAIN_LOSS_STR]:
                        id_ne_file.add_modifier(TrainSubStrSuf.TRUTH.value)
                    yield net_stats_creator(
                        TrainNNStatsElemInfo(
                            id_file=id_ne_file, name_series=fixed_erg.series,
                            type_series='Loss Leg'
                            if fixed_erg.series in [_EVAL_LOSS_STR, _TRAIN_LOSS_STR]
                            else 'Truth Leg',
                            name_sub_series="Loss"
                            if fixed_erg.series in [_EVAL_LOSS_STR, _TRAIN_LOSS_STR]
                            else 'Truth',
                            type_sub_series='Sub',
                            x_label="epoch",
                            y_label='Loss' if fixed_erg.series in [_EVAL_LOSS_STR, _TRAIN_LOSS_STR]
                            else 'Truth',
                            title="HogwildTrainer", subtitle=""
                        ),
                        [x_elem + fixed_erg.epoch for x_elem in range(len(fixed_erg.y_cords))],
                        fixed_erg.y_cords,
                        TrainNNStatsElementFiller(
                            last=fixed_erg.last, plot_data=self.train_state.get_kwargs().plot_data,
                            dump=args.dump, write_data=self.train_state.get_kwargs().write_data,
                            hyper_param=args.hyper_str
                            if fixed_erg.series in [_EVAL_LOSS_STR, _TRAIN_LOSS_STR] else ""
                        )
                    )
                self._change_container(cont, fixed_erg)
                del cont.erg_dict[tr_create_dict_id_queue(erg)]
            if results_queue.empty() and cont.finished_num == 1:
                cont.finished_num -= 1

        for pr_el in processes:
            pr_el.join()
        if cont.erg_dict:
            raise KnownTrainerError(
                f"The dictionary was not completely send\n{repr(cont.erg_dict)}!"
            )
        self.__state_dict = deepcopy(gen_cont.model.state_dict())
        self._set_fitness(cont.eval_loss, cont.train_loss, cont.eval_truth)
