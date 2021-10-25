# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from copy import deepcopy
from typing import Tuple, List

import torch

from bann.b_test_train_prepare.errors.custom_errors import KnownTesterError


def one_class_stats(output: torch.Tensor, target: torch.Tensor, device: torch.device,
                    class_num: int, /) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    target_l = target
    if target_l.device != device:
        target_l.to(device)
    output_l = output
    if output_l.device != device:
        output_l.to(device)
    classes_list = [0 for _ in range(class_num)]
    ges_list = [0 for _ in range(class_num)]
    prediction = output_l.max(1)[1].eq(target_l).float()
    for index, tar in enumerate(target_l.tolist()):
        classes_list[tar] += prediction[index].item()
        ges_list[tar] += 1
    return tuple(classes_list), tuple(ges_list)


def create_bin_reg_tensor(target_n: torch.Tensor, bin_cut: Tuple[float, ...],
                          class_num: int, /) -> torch.Tensor:
    cut_offs = (float('-inf'), *bin_cut[1:-1], float('inf'))
    for t_id in range(target_n.size(0)):
        cnt_cl: int = 0
        found = False
        while cnt_cl < class_num:
            if cut_offs[cnt_cl] <= target_n[t_id] < cut_offs[cnt_cl + 1]:
                target_n[t_id] = cnt_cl * 1.
                cnt_cl = class_num
                found = True
            else:
                cnt_cl += 1
        if not found:
            raise KnownTesterError(
                f"Value {target_n[t_id]} could not be parsed, with cut-offs {cut_offs}"
            )
    return target_n


def one_reg_stats(output: torch.Tensor, target: torch.Tensor, device: torch.device, class_num: int,
                  bin_cut: Tuple[float, ...], /) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    if output.dim() > 2 or (output.dim() == 2 and output.size(1) > 1) or output.dim() < 1:
        raise KnownTesterError(f"Output tensor has a wrong format {output.size()}")
    output_n: torch.Tensor = deepcopy(output.view(-1))
    target_n: torch.Tensor = deepcopy(target.view(-1))
    if not bin_cut or len(bin_cut) != class_num + 1:
        raise KnownTesterError("Received wrong cutoff number for class definition!")
    target_l = create_bin_reg_tensor(target_n, bin_cut, class_num)
    if target_l.device != device:
        target_l.to(device)
    output_l = create_bin_reg_tensor(output_n, bin_cut, class_num)
    if output_l.device != device:
        output_l.to(device)
    classes_list = [0 for _ in range(class_num)]
    ges_list = [0 for _ in range(class_num)]
    prediction = output_l.eq(target_l).int()
    for index, tar in enumerate(target_l.int().tolist()):
        classes_list[tar] += prediction[index].item()
        ges_list[tar] += 1
    return tuple(classes_list), tuple(ges_list)


def one_class_prediction(output: torch.Tensor, device: torch.device, /) -> torch.Tensor:
    output_l = output
    if output_l.device != device:
        output_l.to(device)
    prediction = output_l.max(1)[1].float()
    return prediction


def merge_one_class_stats(data: List[Tuple[Tuple[float, ...], Tuple[float, ...]]],
                          class_num: int, /) -> str:
    if not data:
        return ''
    calc_sum = (
        [0.0 for _ in range(class_num)],
        [0.0 for _ in range(class_num)]
    )
    for elem_t in data:
        for t_ind in range(class_num):
            calc_sum[0][t_ind] += elem_t[0][t_ind]
            calc_sum[1][t_ind] += elem_t[1][t_ind]
    erg_str = "\"OneClassStats\": {\n"
    erg_str += f"\"Class\": [ {','.join(str(index) for index in range(class_num))} ],\n"
    erg_str += "\"Predicted\": [ "
    erg_str += ','.join(str(calc_sum[0][index]) for index in range(class_num)) + " ],\n"
    erg_str += "\"All\": [ "
    erg_str += ','.join(str(calc_sum[1][index]) for index in range(class_num)) + " ],\n"
    erg_str += "\"Percentage\": [ "
    res_pr = ','.join(
        str(1.0 * calc_sum[0][index] / calc_sum[1][index]) for index in range(class_num)
    )
    erg_str += f"{res_pr}" + " ]\n}"
    return erg_str
