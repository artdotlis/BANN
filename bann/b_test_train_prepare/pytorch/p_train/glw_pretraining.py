# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import math
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable, Callable, Dict, List, Pattern, Union, Final, final

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from bann.b_container.constants.file_names import TrainSubStrSuf
from bann.b_data_functions.pytorch.p_gen_fun import re_copy_model
from bann.b_frameworks.pytorch.interfaces.glw_pretraining_interface import \
    GLWPNetInterface
from bann.b_container.functions.dict_str_repr import dict_string_repr
from bann.b_container.states.framework.interface.train_state import TrainState
from bann.b_test_train_prepare.pytorch.trainer_interface import TrainerInterfaceArgs, \
    TrainerInterface
from bann.b_container.states.framework.pytorch.p_train.p_train_glw import GLWPTState
from bann.b_test_train_prepare.errors.custom_errors import KnownTrainerError
from bann.b_test_train_prepare.pytorch.p_train.functions.p_train_gen_fun import \
    t_print_to_logger, PQueueTupleErg, PTrainEpochFunReturn, \
    p_pre_train_epoch_gen, PPreTrainEpochFun, p_train_epoch_fun, SimpleSGDOptim
from bann.b_container.states.framework.pytorch.lr_scheduler_param import LrSchAlgWr
from bann.b_container.states.framework.pytorch.criterion_param import CriterionAlgWr
from bann.b_container.states.framework.pytorch.optim_param import OptimAlgWr

from pan.public.constants.train_net_stats_constants import TrainNNStatsElementType, \
    TrainReturnFiller, TrainNNStatsElementFiller, TrainNNStatsElemInfo, \
    create_train_net_stats_function
from rewowr.public.functions.syncout_dep_functions import logger_print_to_console
from rewowr.public.functions.worker_ctx import get_worker_ctx
from rewowr.public.interfaces.logger_interface import SyncStdoutInterface

_TRAIN_LOSS_STR: Final[str] = 'Train_Loss'
_EVAL_LOSS_STR: Final[str] = 'Eval_Loss'
_EVAL_TR_V_STR: Final[str] = 'Eval_Truth'
_TRAIN_TR_V_STR: Final[str] = 'Train_Truth'
_STR_LOSS_NUM: Final[Pattern[str]] = re.compile(r'(Eval_Loss|Train_Loss)_(\d+)')
_STR_TR_V_NUM: Final[Pattern[str]] = re.compile(r'(Eval_Truth)_(\d+)')


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
    model: GLWPNetInterface
    device: torch.device


@final
@dataclass
class _TrainWrArgs:
    optimizer: Optional[OptimAlgWr]
    scheduler: Optional[LrSchAlgWr]
    criterion: Optional[CriterionAlgWr]
    model: GLWPNetInterface
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


@final
@dataclass
class _ExtraData:
    use_cuda: bool
    model: nn.Module
    tr_size: int


def _print_to_logger(sync_out: SyncStdoutInterface, trainer_stats: GLWPTState,
                     args_trainer: TrainerInterfaceArgs, extra_args: _ExtraData, /) -> None:
    output_string = f"The arguments given to GLW-Pretrainer:\n"
    output_string += f"Training on data with the size {extra_args.tr_size}\n"
    output_string += f"The GLW-Pretrainer:"
    output_string += f"\n\t{dict_string_repr(trainer_stats.get_kwargs().__dict__)}\n"
    output_string += t_print_to_logger(args_trainer, extra_args.use_cuda, extra_args.model)
    logger_print_to_console(sync_out, output_string)


_TrsT: Final = Callable[
    [Tuple[DataLoader, ...], Tuple[DataLoader, ...], nn.Module, Tuple[int, int, int]],
    PTrainEpochFunReturn
]


def _train_stack_fun(epoch_rep_layer: Tuple[int, int, int],
                     tt_loader: Tuple[Tuple[DataLoader, ...], Tuple[DataLoader, ...]],
                     stop_at_loss: float, layer_model: nn.Module, train_epoch: _TrsT, /) \
        -> Iterable[Tuple[PQueueTupleErg, ...]]:
    max_epoch, report_size, layer_cnt = epoch_rep_layer
    epoch = 1
    running_loss: float = float('inf')
    report_cnt = 0
    while epoch < max_epoch + 1:
        int_max = int(epoch + report_size)
        int_max = max_epoch + 1 if int_max > max_epoch + 1 else int_max
        erg_loss: List[PTrainEpochFunReturn] = list(
            p_pre_train_epoch_gen(tt_loader, (epoch, int_max, layer_cnt), layer_model, train_epoch)
        )
        if erg_loss and not math.isnan(erg_loss[-1].running_loss):
            running_loss = erg_loss[-1].running_loss
        last_run = int_max >= max_epoch + 1 or (
            0 < running_loss < stop_at_loss and report_cnt >= 2
        )
        yield (
            PQueueTupleErg(
                f"{_TRAIN_LOSS_STR}_{layer_cnt}", epoch, [
                    elem.test_train.test_loss
                    if not (math.isnan(elem.test_train.test_loss)
                            or math.isinf(elem.test_train.test_loss))
                    else -1.0
                    for elem in erg_loss
                ], last_run
            ),
            PQueueTupleErg(
                f"{_EVAL_LOSS_STR}_{layer_cnt}", epoch, [
                    elem.test_eval.test_loss
                    if not (math.isnan(elem.test_eval.test_loss)
                            or math.isinf(elem.test_eval.test_loss))
                    else -1.0
                    for elem in erg_loss
                ], last_run
            ),
            PQueueTupleErg(
                f"{_TRAIN_TR_V_STR}_{layer_cnt}", epoch,
                [elem.test_train.truth_v for elem in erg_loss], last_run
            ),
            PQueueTupleErg(
                f"{_EVAL_TR_V_STR}_{layer_cnt}", epoch,
                [elem.test_eval.truth_v for elem in erg_loss], last_run
            )
        )
        epoch = int_max
        if 0 < running_loss < stop_at_loss and report_cnt >= 2:
            epoch = max_epoch + 1
        report_cnt += 1


def _train_fun(stop_at_loss: float, args_cont: _TrainArgsCon, train_epoch: _TrsT, /) \
        -> Iterable[Tuple[PQueueTupleErg, ...]]:
    train_loader = tuple(DataLoader(
        tr_in[1],
        batch_size=args_cont.batch_size[tr_in[0] % len(args_cont.batch_size)],
        shuffle=args_cont.shuffle,
        pin_memory=args_cont.cuda,
        num_workers=args_cont.num_workers,
        drop_last=args_cont.drop_last
    ) for tr_in in enumerate(args_cont.input_train))
    test_loader = tuple(DataLoader(
        te_in[1],
        batch_size=args_cont.batch_size[te_in[0] % len(args_cont.batch_size)],
        shuffle=False,
        pin_memory=args_cont.cuda,
        num_workers=args_cont.num_workers,
        drop_last=False
    ) for te_in in enumerate(args_cont.input_eval))
    if args_cont.num_workers > 0:
        for train_test in [test_loader, train_loader]:
            for tr_te_el in train_test:
                tr_te_el.multiprocessing_context = get_worker_ctx()
    max_epoch = args_cont.epoch_size if args_cont.epoch_size > 4 else 4
    report_size = args_cont.report_size
    report_size = int(max_epoch / 2) if report_size >= int(max_epoch / 2) else report_size
    layer_model: nn.Module
    for layer_cnt, layer_model in enumerate(args_cont.model.get_stack()):
        model = layer_model.to(args_cont.device)
        yield from _train_stack_fun(
            (max_epoch, report_size, layer_cnt), (train_loader, test_loader),
            stop_at_loss, model, train_epoch
        )
        re_copy_model(model.state_dict(), layer_model)


def _train_wrapper(wr_container: _TrainWrArgs, /) -> Iterable[Tuple[PQueueTupleErg, ...]]:
    if wr_container.criterion is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = wr_container.criterion.criterion
    if wr_container.optimizer is None:
        optimizer: Union[OptimAlgWr, SimpleSGDOptim] \
            = SimpleSGDOptim(wr_container.model.get_stack_first)
    else:
        optimizer = wr_container.optimizer

    def train_epoch(data_loader_train: Tuple[DataLoader, ...],
                    data_loader_test: Tuple[DataLoader, ...],
                    model: nn.Module, epoch_layer_batch: Tuple[int, int, int]) \
            -> PTrainEpochFunReturn:
        return p_train_epoch_fun(PPreTrainEpochFun(
            data_loader_train=data_loader_train, data_loader_test=data_loader_test,
            epoch=epoch_layer_batch[0], max_batch_cnt=epoch_layer_batch[2],
            model=model, device=wr_container.device,
            optimizer=optimizer,
            criterion=criterion, scheduler_wrapper=wr_container.scheduler,
            truth_fun_id=wr_container.truth_fun_id,
            train_ll=wr_container.train_ll,
            test_ll=wr_container.test_ll,
            complete_model=wr_container.model,
            layer_cnt=epoch_layer_batch[1],
            shuffle=wr_container.shuffle
        ))

    yield from _train_fun(wr_container.end_criterion, _TrainArgsCon(
        batch_size=wr_container.batch_size, shuffle=wr_container.shuffle,
        input_train=wr_container.input_train, input_eval=wr_container.input_eval,
        cuda=wr_container.cuda, drop_last=wr_container.drop_last,
        num_workers=wr_container.num_workers, epoch_size=wr_container.epoch_size,
        report_size=wr_container.report_size, device=wr_container.device,
        model=wr_container.model
    ), train_epoch)


@final
class GLWPreTrainer(TrainerInterface):
    def __init__(self) -> None:
        super().__init__()
        self.__fitness: Optional[Tuple[float, float]] = None
        self.__train_state: Optional[GLWPTState] = None
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
    def train_state(self) -> GLWPTState:
        if self.__train_state is None or not isinstance(self.__train_state, GLWPTState):
            raise KnownTrainerError("Train state was not set properly!")
        return self.__train_state

    def set_train_state(self, state: TrainState, /) -> None:
        if not isinstance(state, GLWPTState):
            raise KnownTrainerError(
                f"Expected type {GLWPTState.__name__} got {type(state).__name__}"
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
        model = args.module.get_net_com
        if not isinstance(model, GLWPNetInterface):
            raise KnownTrainerError(
                f"Expected {GLWPNetInterface.__name__} got {type(model).__name__}"
            )
        if self.__log:
            _print_to_logger(sync_out, self.train_state, args, _ExtraData(
                use_cuda=use_cuda, model=args.module.get_net_com,
                tr_size=sum(len(tr_in) for tr_in in args.input_train)
            ))

        net_stats_creator_dict: Dict[str, TrainReturnFiller] = {}
        result_buffer = []
        eval_truth_loss = [-1.0, -1.0, -1.0]
        for tuple_erg in _train_wrapper(
                _TrainWrArgs(
                    optimizer=args.optimizer, scheduler=args.scheduler,
                    criterion=args.criterion, device=torch.device("cuda" if use_cuda else "cpu"),
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
                found_str = _STR_LOSS_NUM.search(res.series)
                if self.train_state.get_kwargs().plot_data \
                        or self.train_state.get_kwargs().write_data:
                    net_stats_creator = net_stats_creator_dict.setdefault(
                        res.series, create_train_net_stats_function()
                    )
                    id_ne_file = deepcopy(args.id_file)
                    if found_str is None:
                        id_ne_file.add_modifier(TrainSubStrSuf.TRUTH.value)
                    result = net_stats_creator(
                        TrainNNStatsElemInfo(
                            id_file=id_ne_file, name_series=res.series,
                            type_series='Loss Leg' if found_str is not None else 'Truth Leg',
                            name_sub_series='Loss' if found_str is not None else 'Truth',
                            type_sub_series='Sub',
                            x_label="epoch",
                            y_label='Loss' if found_str is not None else 'Truth',
                            title="GLW-Pretrainer", subtitle=""
                        ),
                        [x_elem + res.epoch for x_elem in range(len(res.y_cords))],
                        res.y_cords,
                        TrainNNStatsElementFiller(
                            last=res.last, plot_data=self.train_state.get_kwargs().plot_data,
                            dump=args.dump, write_data=self.train_state.get_kwargs().write_data,
                            hyper_param=args.hyper_str if found_str is not None else ""
                        )
                    )
                    if res.last:
                        result_buffer.append(result)
                    else:
                        yield result
                if found_str is not None and _EVAL_LOSS_STR == found_str.group(1):
                    eval_truth_loss[1] = res.y_cords[-1]
                if found_str is not None and _TRAIN_LOSS_STR == found_str.group(1):
                    eval_truth_loss[2] = res.y_cords[-1]
                found_str = _STR_TR_V_NUM.search(res.series)
                if found_str is not None and _EVAL_TR_V_STR == found_str.group(1):
                    eval_truth_loss[0] = res.y_cords[-1]

        self._set_fitness(eval_truth_loss[1], eval_truth_loss[2], eval_truth_loss[0])
        if not isinstance(model, nn.Module):
            raise KnownTrainerError(
                f"Expected {nn.Module.__name__} got {type(model).__name__}"
            )
        self.__state_dict = deepcopy(model.state_dict())
        yield from result_buffer
