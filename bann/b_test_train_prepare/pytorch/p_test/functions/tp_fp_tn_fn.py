# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from copy import deepcopy
from typing import Tuple, Dict

import torch
from torch.nn import functional as nn_fun

from bann.b_test_train_prepare.container.test.rttff_c import RtTfF
from bann.b_test_train_prepare.pytorch.p_test.functions.one_class import create_bin_reg_tensor
from bann.b_test_train_prepare.errors.custom_errors import KnownTesterError
from bann.b_test_train_prepare.pytorch.p_test.functions.steps import C_STEP_SIZE, C_STEP_SIZE_F, \
    calc_step_f
from bann.b_test_train_prepare.pytorch.p_test.libs.container import RegOutTarDevC


def one_class_ttff(output: torch.Tensor, target: torch.Tensor, device: torch.device,
                   class_num: int, step_cnt: int, /) -> Tuple[Dict[int, RtTfF], ...]:
    target_out = (target, output)
    if target_out[0].device != device:
        target_out[0].to(device)
    if target_out[1].device != device:
        target_out[1].to(device)
    step_f = calc_step_f(step_cnt)
    classes_list = tuple(
        {num * step_f: RtTfF() for num in range(int(C_STEP_SIZE / step_f) + 1)}
        for _ in range(class_num)
    )
    soft_maxed_data = nn_fun.softmax(target_out[1], dim=1)
    for num in range(int(C_STEP_SIZE / step_f) + 1):
        threshold = num * step_f
        for class_id in range(class_num):
            tr_pr = soft_maxed_data[:, class_id].ge(threshold / C_STEP_SIZE_F)
            fa_pr = soft_maxed_data[:, class_id].lt(threshold / C_STEP_SIZE_F)
            tar_pr = target_out[0].eq(class_id)
            classes_list[class_id][threshold].r_tp += \
                int(torch.masked_select(tar_pr, tr_pr).sum().item())
            classes_list[class_id][threshold].r_fp += \
                int(torch.masked_select(tar_pr, tr_pr).eq(0).sum().item())
            classes_list[class_id][threshold].r_tn += \
                int(torch.masked_select(tar_pr, fa_pr).eq(0).sum().item())
            classes_list[class_id][threshold].r_fn += \
                int(torch.masked_select(tar_pr, fa_pr).sum().item())
    return classes_list


def _scale_tensor(min_a: float, max_a: float, output: torch.Tensor, /) \
        -> Tuple[torch.Tensor, float, float]:
    copy_t = deepcopy(output)
    min_v = copy_t.min().item()
    if min_v > max_a:
        min_v = min_a
    max_v = copy_t.max().item()
    if max_v < max_a:
        max_v = max_a
    return (copy_t - min_v) / (max_v - min_v), min_v, max_v


def _fix_output(lower_limit: float, output: torch.Tensor,
                device: torch.device, /) -> torch.Tensor:
    c_out = deepcopy(output)
    if c_out.device != device:
        c_out.to(device)
    if lower_limit == 0:
        return c_out
    max_v = c_out.max().item()
    min_v = c_out.min().item()
    if max_v > 1:
        print(f"WARNING: MAX VALUE {max_v} EXCEEDS ONE!")
    if min_v < 0:
        print(f"WARNING: NEGATIVE MIN VALUE DETECTED! {min_v}")
    c_out = torch.ones(c_out.size(), device=device) - c_out
    c_out = torch.where(c_out > 1, c_out - 1, c_out)
    c_out = torch.where(c_out < 0, torch.ones(c_out.size(), device=device) - c_out, c_out)
    return c_out


def _reg_class_ttff(tod: RegOutTarDevC, bin_cut: Tuple[float, ...],
                    class_num: int, step_cnt: int, /) -> Tuple[Dict[int, RtTfF], ...]:
    step_f = calc_step_f(step_cnt)
    classes_list = tuple(
        {num * step_f: RtTfF() for num in range(int(C_STEP_SIZE / step_f) + 1)}
        for _ in range(class_num)
    )
    for class_id in range(class_num):
        fixed_out = _fix_output(bin_cut[class_id], tod.output, tod.device)
        for num in range(int(C_STEP_SIZE / step_f) + 1):
            threshold = num * step_f
            tr_pr = fixed_out.lt(threshold / C_STEP_SIZE_F)
            fa_pr = fixed_out.ge(threshold / C_STEP_SIZE_F)
            tar_pr = tod.target.eq(class_id)
            classes_list[class_id][threshold].r_tp += \
                int(torch.masked_select(tar_pr, tr_pr).sum().item())
            classes_list[class_id][threshold].r_fp += \
                int(torch.masked_select(tar_pr, tr_pr).eq(0).sum().item())
            classes_list[class_id][threshold].r_tn += \
                int(torch.masked_select(tar_pr, fa_pr).eq(0).sum().item())
            classes_list[class_id][threshold].r_fn += \
                int(torch.masked_select(tar_pr, fa_pr).sum().item())
    return classes_list


def regression_ttff(out_tar: RegOutTarDevC, class_num: int, step_cnt: int,
                    bin_cut: Tuple[float, ...], /) -> Tuple[Dict[int, RtTfF], ...]:
    if out_tar.output.dim() > 2 or (out_tar.output.dim() == 2 and out_tar.output.size(1) > 1) \
            or out_tar.output.dim() < 1:
        raise KnownTesterError(f"Output tensor has a wrong format {out_tar.output.size()}")
    if class_num != 2:
        print("Warning: optimised for only two classes!")
    output_n: torch.Tensor = deepcopy(out_tar.output.view(-1))
    if not bin_cut or len(bin_cut) != class_num + 1:
        raise KnownTesterError("Received wrong cutoff number for class definition!")
    target_n: torch.Tensor = create_bin_reg_tensor(
        deepcopy(out_tar.target.view(-1)), bin_cut, class_num
    )
    new_cut = list(bin_cut)
    if target_n.device != out_tar.device:
        target_n.to(out_tar.device)
    if output_n.device != out_tar.device:
        output_n.to(out_tar.device)
    new_out, new_min, new_max = _scale_tensor(bin_cut[0], bin_cut[-1], output_n)
    new_cut[0] = new_min
    new_cut[-1] = new_max
    return _reg_class_ttff(RegOutTarDevC(
        output=new_out, target=target_n, device=out_tar.device
    ), tuple((val_f - new_min) / (new_max - new_min) for val_f in new_cut), class_num, step_cnt)
