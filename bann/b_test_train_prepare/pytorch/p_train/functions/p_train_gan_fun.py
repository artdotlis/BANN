# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import math
from dataclasses import dataclass, field
from typing import final, Tuple, Iterable, Callable, Final, Optional, List

import torch
from torch.utils.data import DataLoader

from bann.b_frameworks.pytorch.truth_fun_lib import get_framework_truth_lib
from bann.b_container.states.framework.pytorch.criterion_param import CriterionAlias
from bann.b_frameworks.pytorch.interfaces.activation_function_interface import AfcDataTC, \
    AFClassInterface
from bann.b_frameworks.pytorch.interfaces.gan_interface import GanInterface
from bann.b_test_train_prepare.errors.custom_errors import KnownTrainerError
from bann.b_frameworks.pytorch.act_fun_lib import get_framework_act_lib
from bann.b_pan_integration.framwork_key_lib import FrameworkKeyLib
from bann.b_test_train_prepare.pytorch.p_train.functions.p_train_gen_fun import PTrainEpochFun, \
    p_prepare_train_e, p_shuffle_data_loader

_FRAMEWORK: Final[str] = FrameworkKeyLib.PYTORCH.value


@final
@dataclass
class PGanPTestEpochFunReturn:
    test_loss_d: float
    truth_v_d: float
    test_loss_g: float
    truth_v_g: float


@final
@dataclass
class PGanTrainEpochFunReturn:
    running_loss_d: float
    running_loss_g: float
    test_train: PGanPTestEpochFunReturn
    test_eval: PGanPTestEpochFunReturn


_TrEftN: Final = Callable[
    [Tuple[DataLoader, ...], Tuple[DataLoader, ...], int, int], PGanTrainEpochFunReturn
]


def p_train_epoch_gan(train_loader: Tuple[DataLoader, ...], test_loader: Tuple[DataLoader, ...],
                      start_v: int, stop_v: int, train_epoch: _TrEftN, /) \
        -> Iterable[PGanTrainEpochFunReturn]:
    last_value_loss_d = 1.0
    last_value_loss_g = 1.0
    batch_cnt: int = 0
    for data_l in train_loader:
        batch_cnt += len(data_l)
    for index in range(start_v, stop_v):
        if not (math.isnan(last_value_loss_d) or math.isnan(last_value_loss_g)):
            erg_value = train_epoch(train_loader, test_loader, index, batch_cnt)
            last_value_loss_d = erg_value.running_loss_d
            last_value_loss_g = erg_value.running_loss_g
            yield erg_value
        else:
            yield PGanTrainEpochFunReturn(
                running_loss_d=-2.0,
                running_loss_g=-2.0,
                test_train=PGanPTestEpochFunReturn(
                    test_loss_d=-2.0, truth_v_d=-2.0, test_loss_g=-2.0, truth_v_g=-2.0
                ),
                test_eval=PGanPTestEpochFunReturn(
                    test_loss_d=-2.0, truth_v_d=-2.0, test_loss_g=-2.0, truth_v_g=-2.0
                )
            )


@final
@dataclass
class _TestRunningCont:
    test_loss_d: float = 0.0
    test_loss_g: float = 0.0
    correct_d: float = 0.0
    correct_g: float = 0.0
    test_cnt_d: int = 0
    test_cnt_g: int = 0


@final
@dataclass
class PGanTestEpochFun:
    model: GanInterface
    data_loader: Tuple[DataLoader, ...]
    criterion: CriterionAlias
    device: torch.device
    truth_fun_id: str
    last_layer: Tuple[str, Optional[int]]
    last_layer_test: Tuple[str, Optional[int]]


def p_gan_test_epoch_fun(args: PGanTestEpochFun, /) -> PGanPTestEpochFunReturn:
    args.model.generator.eval()
    args.model.discriminator.eval()
    r_con = _TestRunningCont()
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
                dis_out_t = args.model.create_input_target(dev_data, args.device, False)
                gen_out_t = args.model.create_input_target(dev_data, args.device, True)
                if isinstance(target, torch.Tensor):
                    target = args.model.fix_target_d(target, args.device)
                else:
                    raise KnownTrainerError(
                        f"Found unknown data type {type(target).__name__} for target"
                    )
                dev_data = tuple(
                    torch.cat([data_e, dis_out_t.input[d_i]], dim=0)
                    for d_i, data_e in enumerate(dev_data)
                )
                target = torch.cat([target, dis_out_t.target], dim=0)
                fw_data = args.model.forward_gan(dev_data, args.device)
                output_l_t = (last_layer_tr.act(AfcDataTC(
                    data=fw_data, dim=args.last_layer[1]
                )), last_layer_te.act(AfcDataTC(
                    data=fw_data, dim=args.last_layer_test[1]
                )))
                fw_data = args.model.forward_gan(gen_out_t.input, args.device)
                out_gan_l_t = (last_layer_tr.act(AfcDataTC(
                    data=fw_data, dim=args.last_layer[1]
                )), last_layer_te.act(AfcDataTC(
                    data=fw_data, dim=args.last_layer_test[1]
                )))
                loss_d = args.model.criterion(
                    output_l_t[0], target, args.device, args.criterion
                ).item()
                loss_g = args.model.criterion(
                    out_gan_l_t[0], gen_out_t.target, args.device, args.criterion
                ).item()
                r_con.test_loss_d += (loss_d * target.size(0))
                r_con.test_loss_g += (loss_g * gen_out_t.target.size(0))
                r_con.test_cnt_d += target.size(0)
                r_con.test_cnt_g += gen_out_t.target.size(0)
                r_con.correct_d += args.model.truth(output_l_t[1], target, args.device, te_fun)
                r_con.correct_g += args.model.truth(
                    out_gan_l_t[1], gen_out_t.target, args.device, te_fun
                )
    r_con.test_loss_d /= float(r_con.test_cnt_d)
    r_con.test_loss_g /= float(r_con.test_cnt_g)
    if r_con.test_loss_d < 0:
        print(f"Strange test_d loss detected: {r_con.test_loss_d}, setting loss to inf")
        r_con.test_loss_d = float('inf')
    if r_con.test_loss_g < 0:
        print(f"Strange test_d loss detected: {r_con.test_loss_g}, setting loss to inf")
        r_con.test_loss_g = float('inf')
    return PGanPTestEpochFunReturn(
        test_loss_d=r_con.test_loss_d, test_loss_g=r_con.test_loss_g,
        truth_v_d=1. * r_con.correct_d / float(r_con.test_cnt_d),
        truth_v_g=1. * r_con.correct_g / float(r_con.test_cnt_g)
    )


@final
@dataclass
class _BatchInT:
    input: Tuple[torch.Tensor, ...]
    target: torch.Tensor


@final
@dataclass
class _RunningConst:
    running_loss_d: float = 0.0
    running_loss_g: float = 0.0
    running_cnt_d: int = 0
    running_cnt_g: int = 0
    batch_size: int = 0
    batch_size_d: int = 0
    batch_size_g: int = 0
    step_ac: List[_BatchInT] = field(default_factory=lambda: [])


def _calc_loss_backwards(g_model: GanInterface, device: torch.device,
                         last_layer: Tuple[AFClassInterface, int], criterion: CriterionAlias,
                         in_t: Tuple[Tuple[torch.Tensor, ...], torch.Tensor]) -> torch.Tensor:
    outputs = last_layer[0].act(AfcDataTC(
        data=g_model.forward_gan(in_t[0], device), dim=last_layer[1]
    ))
    loss_puf = g_model.criterion(outputs, in_t[1], device, criterion)
    loss_puf.backward()
    return loss_puf


def p_gan_train_epoch_fun(args: PTrainEpochFun, /) -> PGanTrainEpochFunReturn:
    if not isinstance(args.model, GanInterface):
        raise KnownTrainerError(
            f"Expected {GanInterface.__name__} got {type(args.model).__name__}"
        )
    gan_net: GanInterface = args.model
    p_prepare_train_e(args)
    r_con = _RunningConst()
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
                targets = gan_net.fix_target_d(data_l[1], args.device)
            else:
                raise KnownTrainerError(
                    f"Found unknown data type {type(data_l[1]).__name__} for target"
                )

            r_con.batch_size += targets.size(0)
            r_con.step_ac.append(_BatchInT(input=inputs, target=targets))

        def closure_train_d() -> float:
            args.optimizer.optim.zero_grad()
            loss: Optional[torch.Tensor] = None
            r_con.batch_size_d = 0
            for in_tar in r_con.step_ac:
                # forward + backward + optimize
                with torch.no_grad():
                    gen_out_t = gan_net.create_input_target(in_tar.input, args.device, False)
                r_con.batch_size_d += in_tar.target.size(0) + gen_out_t.target.size(0)
                loss_puf = _calc_loss_backwards(
                    gan_net, args.device, (last_layer, args.train_ll[1]), args.criterion,
                    (in_tar.input, in_tar.target)
                )
                loss_puf += _calc_loss_backwards(
                    gan_net, args.device, (last_layer, args.train_ll[1]), args.criterion,
                    (gen_out_t.input, gen_out_t.target)
                )
                gan_net.generator.zero_grad()
                if loss is None:
                    loss = loss_puf
                else:
                    loss += loss_puf
            if loss is None:
                raise KnownTrainerError("No data received!")
            return float((loss / len(r_con.step_ac)).item())

        def closure_train_g() -> float:
            args.optimizer.optim.zero_grad()
            loss: Optional[torch.Tensor] = None
            r_con.batch_size_g = 0
            for in_tar in r_con.step_ac:
                # forward + backward + optimize
                gen_out_t = gan_net.create_input_target(in_tar.input, args.device, True)
                r_con.batch_size_g += gen_out_t.target.size(0)
                loss_puf = _calc_loss_backwards(
                    gan_net, args.device, (last_layer, args.train_ll[1]), args.criterion,
                    (gen_out_t.input, gen_out_t.target)
                )
                gan_net.discriminator.zero_grad()
                if loss is None:
                    loss = loss_puf
                else:
                    loss += loss_puf
            if loss is None:
                raise KnownTrainerError("No data received!")
            return float((loss / len(r_con.step_ac)).item())

        if r_con.batch_size >= args.max_batch_cnt or (
                last_r and r_con.batch_size > 0
        ):
            erg: Optional[float] = float('inf')
            try:
                erg = args.optimizer.optim.step(closure_train_d)
            except RuntimeError as r_er:
                print(f"WARNING! RuntimeError during optim_d step occurred {str(r_er)}")
            if not isinstance(erg, float):
                raise KnownTrainerError(f"Optimiser did not return a loss value ({erg})!")
            r_con.running_loss_d += (erg * r_con.batch_size_d)
            r_con.running_cnt_d += r_con.batch_size_d
            try:
                erg = args.optimizer.optim.step(closure_train_g)
            except RuntimeError as r_er:
                print(f"WARNING! RuntimeError during optim_g step occurred {str(r_er)}")
            if not isinstance(erg, float):
                raise KnownTrainerError(f"Optimiser did not return a loss value ({erg})!")
            r_con.running_loss_g += (erg * r_con.batch_size_g)
            r_con.running_cnt_g += r_con.batch_size_g
            if args.scheduler_wrapper is not None:
                args.scheduler_wrapper.step_wrapper(
                    r_con.running_loss_g / float(r_con.running_cnt_g)
                    + r_con.running_loss_d / float(r_con.running_cnt_d), True
                )
            r_con.step_ac = []
            r_con.batch_size = 0
    r_con.running_loss_d /= float(r_con.running_cnt_d)
    if r_con.running_loss_d < 0:
        print(f"Strange training_d loss detected: {r_con.running_loss_d}, setting loss to inf")
        r_con.running_loss_d = float('inf')
    r_con.running_loss_g /= float(r_con.running_cnt_g)
    if r_con.running_loss_g < 0:
        print(f"Strange training_g loss detected: {r_con.running_loss_g}, setting loss to inf")
        r_con.running_loss_g = float('inf')
    if args.scheduler_wrapper is not None:
        args.scheduler_wrapper.step_wrapper(r_con.running_loss_g + r_con.running_loss_d, False)
    return PGanTrainEpochFunReturn(
        running_loss_g=r_con.running_loss_g,
        running_loss_d=r_con.running_loss_d,
        test_train=p_gan_test_epoch_fun(PGanTestEpochFun(
            model=gan_net, data_loader=args.data_loader_train,
            criterion=args.criterion, device=args.device,
            truth_fun_id=args.truth_fun_id,
            last_layer=args.train_ll,
            last_layer_test=args.test_ll
        )),
        test_eval=p_gan_test_epoch_fun(PGanTestEpochFun(
            model=gan_net, data_loader=args.data_loader_test,
            criterion=args.criterion, device=args.device,
            truth_fun_id=args.truth_fun_id,
            last_layer=args.train_ll,
            last_layer_test=args.test_ll
        ))
    )
