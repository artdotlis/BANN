# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable, List, Callable, Dict, Union, Final, final

import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn

from bann.b_container.constants.file_names import TrainSubStrSuf
from bann.b_test_train_prepare.pytorch.p_train.functions.p_train_gen_fun import \
    PTrainEpochFunReturn, PTrainEpochFun, p_train_epoch_fun, p_train_epoch_gen, PQueueTupleErg, \
    t_print_to_logger, SimpleSGDOptim
from bann.b_container.functions.dict_str_repr import dict_string_repr
from bann.b_container.states.framework.interface.train_state import TrainState
from bann.b_test_train_prepare.errors.custom_errors import KnownTrainerError
from bann.b_container.states.framework.pytorch.p_train.p_train_single_thread import \
    SingleThreadTState
from bann.b_container.states.framework.pytorch.criterion_param import CriterionAlgWr
from bann.b_container.states.framework.pytorch.lr_scheduler_param import LrSchAlgWr
from bann.b_container.states.framework.pytorch.optim_param import OptimAlgWr
from bann.b_test_train_prepare.pytorch.trainer_interface import TrainerInterface, \
    TrainerInterfaceArgs
from rewowr.public.functions.syncout_dep_functions import logger_print_to_console
from rewowr.public.functions.worker_ctx import get_worker_ctx
from rewowr.public.interfaces.logger_interface import SyncStdoutInterface
from pan.public.constants.train_net_stats_constants import TrainNNStatsElementType, \
    TrainReturnFiller, create_train_net_stats_function, TrainNNStatsElemInfo, \
    TrainNNStatsElementFiller

_TRAIN_LOSS_STR: Final[str] = 'Train_Loss'
_EVAL_LOSS_STR: Final[str] = 'Eval_Loss'
_EVAL_TR_V_STR: Final[str] = 'Eval_Truth'
_TRAIN_TR_V_STR: Final[str] = 'Train_Truth'


def add_mod_str(data_to_spl: str, /) -> str:
    if data_to_spl in [_EVAL_LOSS_STR, _EVAL_TR_V_STR]:
        return "eval"
    if data_to_spl in [_TRAIN_LOSS_STR, _TRAIN_TR_V_STR]:
        return "train"
    return ""


@final
@dataclass
class _TrainArgsCon:
    batch_size: Tuple[int, ...]
    shuffle: bool
    input_train: Tuple[Dataset, ...]
    input_eval: Tuple[Dataset, ...]
    cuda: bool
    drop_last: bool
    num_workers: int
    epoch_size: int
    report_size: int


_TrfT: Final = Callable[
    [Tuple[DataLoader, ...], Tuple[DataLoader, ...], int, int], PTrainEpochFunReturn
]


def _train_fun(stop_at_loss: float, args_cont: _TrainArgsCon, train_epoch: _TrfT, /) \
        -> Iterable[Tuple[PQueueTupleErg, ...]]:
    train_loader = tuple(DataLoader(
        tr_l[1],
        batch_size=args_cont.batch_size[tr_l[0] % len(args_cont.batch_size)],
        shuffle=args_cont.shuffle,
        pin_memory=args_cont.cuda,
        num_workers=args_cont.num_workers,
        drop_last=args_cont.drop_last
    ) for tr_l in enumerate(args_cont.input_train))
    test_loader = tuple(DataLoader(
        te_l[1],
        batch_size=args_cont.batch_size[te_l[0] % len(args_cont.batch_size)],
        shuffle=False,
        pin_memory=args_cont.cuda,
        num_workers=args_cont.num_workers,
        drop_last=False
    ) for te_l in enumerate(args_cont.input_eval))
    if args_cont.num_workers > 0:
        for train_test in [test_loader, train_loader]:
            for tr_te_el in train_test:
                tr_te_el.multiprocessing_context = get_worker_ctx()
    max_epoch = args_cont.epoch_size if args_cont.epoch_size > 4 else 4
    report_size = args_cont.report_size
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
        yield (
            PQueueTupleErg(
                _TRAIN_LOSS_STR, epoch, [
                    elem.test_train.test_loss
                    if not (math.isnan(elem.test_train.test_loss)
                            or math.isinf(elem.test_train.test_loss))
                    else -1.0
                    for elem in erg_loss
                ], last_run
            ),
            PQueueTupleErg(
                _EVAL_LOSS_STR, epoch, [
                    elem.test_eval.test_loss
                    if not (math.isnan(elem.test_eval.test_loss)
                            or math.isinf(elem.test_eval.test_loss))
                    else -1.0
                    for elem in erg_loss
                ], last_run
            ),
            PQueueTupleErg(
                _TRAIN_TR_V_STR, epoch, [elem.test_train.truth_v for elem in erg_loss], last_run
            ),
            PQueueTupleErg(
                _EVAL_TR_V_STR, epoch, [elem.test_eval.truth_v for elem in erg_loss], last_run
            )
        )
        epoch = int_max
        if 0 < running_loss < stop_at_loss and report_cnt >= 2:
            epoch = max_epoch + 1
        report_cnt += 1


@final
@dataclass
class _TrainWrArgs:
    optimizer: Optional[OptimAlgWr]
    scheduler: Optional[LrSchAlgWr]
    criterion: Optional[CriterionAlgWr]
    model: nn.Module
    device: torch.device
    truth_fun_id: str
    train_ll: Tuple[str, Optional[int]]
    test_ll: Tuple[str, Optional[int]]
    end_criterion: float
    batch_size: Tuple[int, ...]
    shuffle: bool
    input_train: Tuple[Dataset, ...]
    input_eval: Tuple[Dataset, ...]
    cuda: bool
    drop_last: bool
    num_workers: int
    epoch_size: int
    report_size: int


def _train_wrapper(wr_container: _TrainWrArgs, /) -> Iterable[Tuple[PQueueTupleErg, ...]]:
    if wr_container.criterion is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = wr_container.criterion.criterion
    if wr_container.optimizer is None:
        optimizer: Union[OptimAlgWr, SimpleSGDOptim] = SimpleSGDOptim(wr_container.model)
    else:
        optimizer = wr_container.optimizer

    def train_epoch(data_loader_train: Tuple[DataLoader, ...],
                    data_loader_test: Tuple[DataLoader, ...],
                    epoch: int, batch_cnt: int) -> PTrainEpochFunReturn:
        return p_train_epoch_fun(PTrainEpochFun(
            data_loader_train=data_loader_train, data_loader_test=data_loader_test,
            epoch=epoch, max_batch_cnt=batch_cnt,
            model=wr_container.model, device=wr_container.device,
            optimizer=optimizer,
            criterion=criterion, scheduler_wrapper=wr_container.scheduler,
            truth_fun_id=wr_container.truth_fun_id,
            train_ll=wr_container.train_ll,
            test_ll=wr_container.test_ll,
            shuffle=wr_container.shuffle
        ))

    yield from _train_fun(wr_container.end_criterion, _TrainArgsCon(
        batch_size=wr_container.batch_size, shuffle=wr_container.shuffle,
        input_train=wr_container.input_train, input_eval=wr_container.input_eval,
        cuda=wr_container.cuda, drop_last=wr_container.drop_last,
        num_workers=wr_container.num_workers, epoch_size=wr_container.epoch_size,
        report_size=wr_container.report_size
    ), train_epoch)


@final
@dataclass
class _ExtraData:
    use_cuda: bool
    model: nn.Module
    tr_size: int


def _print_to_logger(sync_out: SyncStdoutInterface,
                     trainer_stats: SingleThreadTState,
                     args_trainer: TrainerInterfaceArgs,
                     extra_args: _ExtraData, /) -> None:
    output_string = f"The arguments given to SingleThreadTrainer:\n"
    output_string += f"Training on data with the size {extra_args.tr_size}\n"
    output_string += f"The SingleThreadTState:"
    output_string += f"\n\t{dict_string_repr(trainer_stats.get_kwargs().__dict__)}\n"
    output_string += t_print_to_logger(args_trainer, extra_args.use_cuda, extra_args.model)
    logger_print_to_console(sync_out, output_string)


@final
class SingThreadTrainer(TrainerInterface):

    def __init__(self) -> None:
        super().__init__()
        self.__fitness: Optional[Tuple[float, float]] = None
        self.__train_state: Optional[SingleThreadTState] = None
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
    def train_state(self) -> SingleThreadTState:
        if self.__train_state is None or not isinstance(self.__train_state, SingleThreadTState):
            raise KnownTrainerError("Train state was not set properly!")
        return self.__train_state

    def set_train_state(self, state: TrainState, /) -> None:
        if not isinstance(state, SingleThreadTState):
            raise KnownTrainerError(
                f"Expected type {SingleThreadTState.__name__} got {type(state).__name__}"
            )
        self.__train_state = state

    @property
    def fitness(self) -> Tuple[float, float]:
        if self.__fitness is None:
            raise KnownTrainerError("The Trainer was not started!")
        return self.__fitness

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

    def train(self, sync_out: SyncStdoutInterface,
              args: TrainerInterfaceArgs, /) -> Iterable[TrainNNStatsElementType]:
        self.__state_dict = None
        if not (args.input_train and args.input_eval):
            raise KnownTrainerError("Received empty dataset!")
        if self.train_state.get_kwargs().torch_thread:
            torch.set_num_threads(self.train_state.get_kwargs().torch_thread)
            logger_print_to_console(sync_out, f"Torch threads set to {torch.get_num_threads()}\n")
        use_cuda = args.cuda and torch.cuda.is_available()
        torch_device = torch.device("cuda" if use_cuda else "cpu")
        model = args.module.get_net_com.to(torch_device)
        if self.__log:
            _print_to_logger(sync_out, self.train_state, args, _ExtraData(
                use_cuda=use_cuda, model=model,
                tr_size=sum(len(tr_in) for tr_in in args.input_train)
            ))

        net_stats_creator_dict: Dict[str, TrainReturnFiller] = {}
        eval_loss_truth = [-1.0, -1.0, -1.0]
        for tuple_erg in _train_wrapper(
                _TrainWrArgs(
                    optimizer=args.optimizer, scheduler=args.scheduler,
                    criterion=args.criterion, device=torch_device,
                    truth_fun_id=args.truth_fun_id, train_ll=self.train_state.get_kwargs().train_ll,
                    test_ll=self.train_state.get_kwargs().eval_ll,
                    end_criterion=self.train_state.get_kwargs().end_criterion,
                    batch_size=self.train_state.get_kwargs().batch_size,
                    shuffle=self.train_state.get_kwargs().shuffle,
                    input_train=args.input_train, input_eval=args.input_eval, cuda=use_cuda,
                    drop_last=self.train_state.get_kwargs().drop_last,
                    num_workers=self.train_state.get_kwargs().num_workers,
                    epoch_size=self.train_state.get_kwargs().epoch_size,
                    report_size=self.train_state.get_kwargs().report_size,
                    model=model
                )
        ):
            for res in tuple_erg:
                if self.train_state.get_kwargs().plot_data \
                        or self.train_state.get_kwargs().write_data:
                    net_stats_creator = net_stats_creator_dict.setdefault(
                        res.series, create_train_net_stats_function()
                    )
                    id_ne_file = deepcopy(args.id_file)
                    if res.series not in [_EVAL_LOSS_STR, _TRAIN_LOSS_STR]:
                        id_ne_file.add_modifier(TrainSubStrSuf.TRUTH.value)
                    yield net_stats_creator(
                        TrainNNStatsElemInfo(
                            id_file=id_ne_file, name_series=res.series,
                            type_series='Loss Leg'
                            if res.series in [_EVAL_LOSS_STR, _TRAIN_LOSS_STR]
                            else 'Truth Leg',
                            name_sub_series='Loss'
                            if res.series in [_EVAL_LOSS_STR, _TRAIN_LOSS_STR]
                            else 'Truth',
                            type_sub_series='Sub',
                            x_label="epoch",
                            y_label='Loss' if res.series in [_EVAL_LOSS_STR, _TRAIN_LOSS_STR]
                            else 'Truth',
                            title="SingleThreadTrainer", subtitle=""
                        ),
                        [x_elem + res.epoch for x_elem in range(len(res.y_cords))],
                        res.y_cords,
                        TrainNNStatsElementFiller(
                            last=res.last, plot_data=self.train_state.get_kwargs().plot_data,
                            dump=args.dump, write_data=self.train_state.get_kwargs().write_data,
                            hyper_param=args.hyper_str
                            if res.series in [_EVAL_LOSS_STR, _TRAIN_LOSS_STR] else ""
                        )
                    )
                if _EVAL_TR_V_STR == res.series:
                    eval_loss_truth[1] = res.y_cords[-1]
                if _EVAL_LOSS_STR == res.series:
                    eval_loss_truth[0] = res.y_cords[-1]
                if _TRAIN_LOSS_STR == res.series:
                    eval_loss_truth[2] = res.y_cords[-1]
        self.__state_dict = deepcopy(model.state_dict())
        self._set_fitness(eval_loss_truth[0], eval_loss_truth[2], eval_loss_truth[1])
