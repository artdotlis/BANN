# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import math
import random
from dataclasses import dataclass, field
from typing import Optional, Callable, Iterable, List, Tuple, Union, Final, final

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from bann.b_container.states.framework.interface.pytorch.optim_per_parameter import \
    PerParameterAbc
from bann.b_frameworks.pytorch.interfaces.activation_function_interface import AfcDataTC
from bann.b_frameworks.pytorch.interfaces.glw_pretraining_interface import \
    GLWPNetInterface
from bann.b_container.functions.dict_str_repr import dict_string_repr
from bann.b_test_train_prepare.pytorch.trainer_interface import TrainerInterfaceArgs
from bann.b_frameworks.pytorch.interfaces.criterion_less import CriterionLess
from bann.b_container.states.framework.pytorch.criterion_param import CriterionAlias
from bann.b_container.states.framework.pytorch.lr_scheduler_param import LrSchAlgWr
from bann.b_container.states.framework.pytorch.optim_param import OptimizerAlias, OptimAlgWr
from bann.b_pan_integration.framwork_key_lib import FrameworkKeyLib
from bann.b_test_train_prepare.errors.custom_errors import KnownTrainerError
from bann.b_frameworks.pytorch.act_fun_lib import get_framework_act_lib
from bann.b_frameworks.pytorch.truth_fun_lib import get_framework_truth_lib


_FRAMEWORK: Final[str] = FrameworkKeyLib.PYTORCH.value


@final
class SimpleSGDOptim:
    def __init__(self, model: nn.Module, /) -> None:
        super().__init__()
        self.__optim = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    @property
    def optim(self) -> OptimizerAlias:
        return self.__optim

    def update_only_model(self, model: nn.Module, /) -> None:
        self.__optim = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


@final
@dataclass
class PTestEpochFunReturn:
    test_loss: float
    truth_v: float


@final
@dataclass
class PTrainEpochFunReturn:
    running_loss: float
    test_train: PTestEpochFunReturn
    test_eval: PTestEpochFunReturn


@final
@dataclass
class PTestEpochFun:
    model: nn.Module
    data_loader: Tuple[DataLoader, ...]
    criterion: CriterionAlias
    device: torch.device
    truth_fun_id: str
    last_layer: Tuple[str, Optional[int]]
    last_layer_test: Tuple[str, Optional[int]]
    complete_model: Optional[GLWPNetInterface]
    layer_cnt: int


@final
@dataclass
class PTrainEpochFun:
    data_loader_train: Tuple[DataLoader, ...]
    data_loader_test: Tuple[DataLoader, ...]
    epoch: int
    max_batch_cnt: int
    model: nn.Module
    device: torch.device
    criterion: CriterionAlias
    optimizer: Union[OptimAlgWr, SimpleSGDOptim]
    scheduler_wrapper: Optional[LrSchAlgWr]
    truth_fun_id: str
    train_ll: Tuple[str, Optional[int]]
    test_ll: Tuple[str, Optional[int]]
    shuffle: bool


@final
@dataclass
class PPreTrainEpochFun:
    data_loader_train: Tuple[DataLoader, ...]
    data_loader_test: Tuple[DataLoader, ...]
    epoch: int
    max_batch_cnt: int
    model: nn.Module
    device: torch.device
    criterion: CriterionAlias
    optimizer: Union[OptimAlgWr, SimpleSGDOptim]
    scheduler_wrapper: Optional[LrSchAlgWr]
    truth_fun_id: str
    train_ll: Tuple[str, Optional[int]]
    test_ll: Tuple[str, Optional[int]]
    complete_model: GLWPNetInterface
    layer_cnt: int
    shuffle: bool


def p_test_epoch_fun(args: PTestEpochFun, /) -> PTestEpochFunReturn:
    args.model.eval()
    test_loss: float = 0.0
    test_cnt: int = 0
    correct: float = 0.0
    te_fun = get_framework_truth_lib(_FRAMEWORK).truth_fun(args.truth_fun_id)
    last_layer_tr = get_framework_act_lib(_FRAMEWORK).act_b(args.last_layer[0])
    last_layer_te = get_framework_act_lib(_FRAMEWORK).act_b(args.last_layer_test[0])
    with torch.no_grad():
        for d_l_e in args.data_loader:
            for data, target in d_l_e:
                if isinstance(data, torch.Tensor):
                    dev_data: Tuple[torch.Tensor, ...] = (data.to(args.device), )
                elif isinstance(data, (tuple, list, set)):
                    dev_data = tuple(data_el.to(args.device) for data_el in data)
                else:
                    raise KnownTrainerError(
                        f"Found unknown data type {type(data).__name__} for input data"
                    )
                if isinstance(target, torch.Tensor):
                    target = target.to(args.device)
                else:
                    raise KnownTrainerError(
                        f"Found unknown data type {type(target).__name__} for target"
                    )
                if args.complete_model is not None:
                    dev_data = args.complete_model.prepare_input(args.layer_cnt, dev_data)
                    target = args.complete_model.prepare_target(args.layer_cnt, target)
                    output_loss = last_layer_tr.act(AfcDataTC(
                        data=args.complete_model.prepare_output(
                            args.layer_cnt, args.model(*dev_data)
                        ), dim=args.last_layer[1]
                    ))
                    output_test = last_layer_te.act(AfcDataTC(
                        data=args.complete_model.prepare_output(
                            args.layer_cnt, args.model(*dev_data)
                        ), dim=args.last_layer_test[1]
                    ))
                else:
                    output_loss = last_layer_tr.act(AfcDataTC(
                        data=args.model(*dev_data), dim=args.last_layer[1]
                    ))
                    output_test = last_layer_te.act(AfcDataTC(
                        data=args.model(*dev_data), dim=args.last_layer_test[1]
                    ))
                if isinstance(args.model, CriterionLess):
                    loss = args.model.criterion(output_loss, target).item()
                else:
                    loss = args.criterion(output_loss, target).item()
                test_loss += (loss * target.size(0))
                test_cnt += target.size(0)
                correct += te_fun.calc_truth(
                    te_fun.cr_truth_container(output_test, target, args.device)
                )
    test_loss /= float(test_cnt)
    if test_loss <= 0:
        print(f"Strange test loss detected: {test_loss}, setting loss to inf")
        test_loss = float('inf')
    truth_v = 1. * correct / float(test_cnt)
    return PTestEpochFunReturn(
        test_loss=test_loss,
        truth_v=truth_v
    )


@final
@dataclass
class _BatchInT:
    input: Tuple[torch.Tensor, ...]
    target: torch.Tensor


@final
@dataclass
class _RunningConst:
    running_loss: float = 0.0
    running_cnt: int = 0
    batch_size: int = 0
    step_ac: List[_BatchInT] = field(default_factory=lambda: [])


def p_prepare_train_e(args: Union[PTrainEpochFun, PPreTrainEpochFun], /) -> None:
    if not args.epoch:
        raise KnownTrainerError(f"Epoch should start at 1 not {args.epoch}")
    if args.epoch == 1:
        args.optimizer.update_only_model(args.model)
        if args.scheduler_wrapper is not None:
            args.scheduler_wrapper.update_only_optim(args.optimizer.optim)
    len_data = len(args.data_loader_train)
    if args.shuffle and len_data > 1:
        random.seed()
        args.data_loader_train = tuple(random.sample(args.data_loader_train, len_data))
    args.optimizer.update_epoch(args.epoch, args.model)
    if args.scheduler_wrapper is not None:
        args.scheduler_wrapper.update_lr_optim(args.optimizer)
    args.model.train()


def p_shuffle_data_loader(data_loader_t: Tuple[DataLoader, ...], /) -> Iterable[Tuple[Tuple, bool]]:
    all_data_l = len(data_loader_t)
    all_iter = {
        l_i: iter(data_l) for l_i, data_l in enumerate(data_loader_t)
    }
    while all_iter:
        finished_l = []
        for index_i in range(all_data_l):
            if index_i in all_iter:
                try:
                    yield next(all_iter[index_i]), False
                except StopIteration:
                    finished_l.append(index_i)
        for index_i in finished_l:
            del all_iter[index_i]
    yield (), True


def p_train_epoch_fun(args: Union[PTrainEpochFun, PPreTrainEpochFun], /) -> PTrainEpochFunReturn:
    p_prepare_train_e(args)
    running_const = _RunningConst()
    last_layer = get_framework_act_lib(_FRAMEWORK).act_b(args.train_ll[0])
    inputs: Tuple[torch.Tensor, ...]
    targets: torch.Tensor
    for data_l, last_r in p_shuffle_data_loader(args.data_loader_train):
        if not last_r:
            if isinstance(data_l[0], torch.Tensor):
                inputs = (data_l[0].to(args.device),)
            elif isinstance(data_l[0], (tuple, list, set)):
                inputs = tuple(data_el.to(args.device) for data_el in data_l[0])
            else:
                raise KnownTrainerError(
                    f"Found unknown data type {type(data_l[0]).__name__} for input data"
                )
            if isinstance(data_l[1], torch.Tensor):
                targets = data_l[1].to(args.device)
            else:
                raise KnownTrainerError(
                    f"Found unknown data type {type(data_l[1]).__name__} for target"
                )

            if isinstance(args, PPreTrainEpochFun):
                inputs = args.complete_model.prepare_input(args.layer_cnt, inputs)
                targets = args.complete_model.prepare_target(args.layer_cnt, targets)

            running_const.batch_size += targets.size(0)
            running_const.step_ac.append(_BatchInT(input=inputs, target=targets))

        def closure_train() -> float:
            args.optimizer.optim.zero_grad()
            loss: Optional[torch.Tensor] = None
            for in_tar in running_const.step_ac:
                # forward + backward + optimize
                if isinstance(args, PPreTrainEpochFun):
                    outputs = last_layer.act(AfcDataTC(
                        data=args.complete_model.prepare_output(
                            args.layer_cnt, args.model(*in_tar.input)
                        ), dim=args.train_ll[1]
                    ))
                else:
                    outputs = last_layer.act(AfcDataTC(
                        data=args.model(*in_tar.input), dim=args.train_ll[1]
                    ))
                if isinstance(args.model, CriterionLess):
                    loss_puf = args.model.criterion(outputs, in_tar.target)
                else:
                    loss_puf = args.criterion(outputs, in_tar.target)
                loss_puf.backward()
                if loss is None:
                    loss = loss_puf
                else:
                    loss += loss_puf
            if loss is None:
                raise KnownTrainerError("No data received!")
            return float((loss / len(running_const.step_ac)).item())

        if running_const.batch_size >= args.max_batch_cnt or (
                last_r and running_const.batch_size > 0
        ):
            erg: Optional[float] = float('inf')
            try:
                erg = args.optimizer.optim.step(closure_train)
            except RuntimeError as r_er:
                print(f"WARNING! RuntimeError during optim step occurred {str(r_er)}")
            if not isinstance(erg, float):
                raise KnownTrainerError(f"Optimiser did not return a loss value ({erg})!")
            running_const.running_loss += (erg * running_const.batch_size)
            running_const.running_cnt += running_const.batch_size

            if args.scheduler_wrapper is not None:
                args.scheduler_wrapper.step_wrapper(
                    running_const.running_loss / float(running_const.running_cnt), True
                )
            running_const.step_ac = []
            running_const.batch_size = 0

    running_const.running_loss /= float(running_const.running_cnt)
    if running_const.running_loss <= 0:
        print(f"Strange training loss detected: {running_const.running_loss}, setting loss to inf")
        running_const.running_loss = float('inf')
    if args.scheduler_wrapper is not None:
        args.scheduler_wrapper.step_wrapper(running_const.running_loss, False)
    return PTrainEpochFunReturn(
        running_loss=running_const.running_loss,
        test_train=p_test_epoch_fun(PTestEpochFun(
            model=args.model, data_loader=args.data_loader_train,
            criterion=args.criterion, device=args.device,
            truth_fun_id=args.truth_fun_id,
            last_layer=args.train_ll,
            last_layer_test=args.test_ll,
            complete_model=args.complete_model if isinstance(args, PPreTrainEpochFun) else None,
            layer_cnt=args.layer_cnt if isinstance(args, PPreTrainEpochFun) else 0
        )),
        test_eval=p_test_epoch_fun(PTestEpochFun(
            model=args.model, data_loader=args.data_loader_test,
            criterion=args.criterion, device=args.device,
            truth_fun_id=args.truth_fun_id,
            last_layer=args.train_ll,
            last_layer_test=args.test_ll,
            complete_model=args.complete_model if isinstance(args, PPreTrainEpochFun) else None,
            layer_cnt=args.layer_cnt if isinstance(args, PPreTrainEpochFun) else 0
        ))
    )


_TrEftN: Final = Callable[
    [Tuple[DataLoader, ...], Tuple[DataLoader, ...], int, int], PTrainEpochFunReturn
]


def p_train_epoch_gen(train_loader: Tuple[DataLoader, ...], test_loader: Tuple[DataLoader, ...],
                      start_v: int, stop_v: int, train_epoch: _TrEftN, /) \
        -> Iterable[PTrainEpochFunReturn]:
    last_value_loss = 1.0
    batch_cnt: int = 0
    for data_l in train_loader:
        batch_cnt += len(data_l)
    for index in range(start_v, stop_v):
        if not math.isnan(last_value_loss):
            erg_value = train_epoch(train_loader, test_loader, index, batch_cnt)
            last_value_loss = erg_value.running_loss
            yield erg_value
        else:
            yield PTrainEpochFunReturn(
                running_loss=-2.0,
                test_train=PTestEpochFunReturn(test_loss=-2.0, truth_v=-2.0),
                test_eval=PTestEpochFunReturn(test_loss=-2.0, truth_v=-2.0)
            )


_TrEftP: Final = Callable[
    [Tuple[DataLoader, ...], Tuple[DataLoader, ...], nn.Module, Tuple[int, int, int]],
    PTrainEpochFunReturn
]


def p_pre_train_epoch_gen(tt_loader: Tuple[Tuple[DataLoader, ...], Tuple[DataLoader, ...]],
                          start_stop_layer: Tuple[int, int, int], model: nn.Module,
                          train_epoch: _TrEftP, /) -> Iterable[PTrainEpochFunReturn]:
    train_loader, test_loader = tt_loader
    start_v, stop_v, layer = start_stop_layer
    last_value_loss = 1.0
    batch_cnt: int = 0
    for data_l in train_loader:
        batch_cnt += len(data_l)
    for index in range(start_v, stop_v):
        if not math.isnan(last_value_loss):
            erg_value = train_epoch(train_loader, test_loader, model, (index, layer, batch_cnt))
            last_value_loss = erg_value.running_loss
            yield erg_value
        else:
            yield PTrainEpochFunReturn(
                running_loss=-2.0,
                test_train=PTestEpochFunReturn(test_loss=-2.0, truth_v=-2.0),
                test_eval=PTestEpochFunReturn(test_loss=-2.0, truth_v=-2.0)
            )


@final
@dataclass
class PQueueTupleErg:
    series: str
    epoch: int
    y_cords: List[float]
    last: bool


def tr_create_dict_id_queue(data: PQueueTupleErg, /) -> str:
    return f"{data.series}_{data.epoch}"


def t_print_to_logger(args_trainer: TrainerInterfaceArgs, use_cuda: bool,
                      model: nn.Module, /) -> str:
    output_string = "The optim parameters"
    if args_trainer.optimizer is None:
        output_string += ":\n\tno optimizer selected, a default was chosen!\n"
    else:
        output_string += \
            f" ({','.join(name for name in args_trainer.optimizer.optim_type_name(False))}):\n"
        output_string += ''.join(
            f"\t--- {opt_i}: ---\n\t{dict_string_repr(opt_s.get_kwargs().__dict__)}"
            + f"\n\t---- {opt_i} ----\n"
            for opt_i, opt_s in enumerate(args_trainer.optimizer.optim_state, 1)
        )
        for opt_i, opt_s in enumerate(args_trainer.optimizer.optim_state, 1):
            if isinstance(opt_s, PerParameterAbc):
                for layer_n, layer_v in opt_s.layer_params(False).items():
                    output_string += f"\t{layer_n}:\n\t{dict_string_repr(layer_v)}\n"
    if isinstance(model, CriterionLess):
        output_string += f"The model has it's own criterion:\n\t{model.criterion_str()}\n"
    else:
        output_string += f"The criterion parameters "
        output_string += f"({args_trainer.criterion.criterion_type_name}):\n"
        if args_trainer.criterion is None:
            output_string += "\tno criterion selected, a default was chosen!\n"
        else:
            puf = dict_string_repr(args_trainer.criterion.criterion_state.get_kwargs().__dict__)
            output_string += f"\t{puf}\n"

    output_string += f"The scheduler parameters"
    if args_trainer.scheduler is None:
        output_string += ":\n\tno scheduler selected, none was chosen!\n"
    else:
        output_string += \
            f" ({','.join(name for name in args_trainer.scheduler.lr_sch_type_name)}):\n"
        output_string += ''.join(
            f"\t--- {lr_i}: ---\n\t{dict_string_repr(lr_sch.get_kwargs().__dict__)}"
            + f"\n\t---- {lr_i} ----\n"
            for lr_i, lr_sch in enumerate(args_trainer.scheduler.lr_sch_state, 1)
        )

    output_string += f"The selected device:\t{'GPU' if args_trainer.cuda else 'CPU'}\n"
    output_string += f"The selection of the device:\t"
    output_string += f"{'failed' if args_trainer.cuda and not use_cuda else 'was successful'}\n"
    return output_string
