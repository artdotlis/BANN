# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
# Spearman-Korrelationskoeffizient
from dataclasses import dataclass
from typing import Tuple, final, List, Iterable

import torch


@final
@dataclass
class TargetOutput:
    target: torch.Tensor
    output: torch.Tensor


def _compare_rank(sorted_id: torch.Tensor, data: torch.Tensor, /) -> Iterable[float]:
    t_pos_r = 1
    while t_pos_r <= len(sorted_id):
        current_v = data[sorted_id[t_pos_r - 1]]
        next_pos = t_pos_r
        running_sum_r: float = t_pos_r
        cmp_cnt: int = 0
        comp: bool = next_pos < len(data) and bool(
            torch.all(current_v == data[sorted_id[next_pos]]).item()
        )
        while comp:
            cmp_cnt += 1
            running_sum_r += t_pos_r + cmp_cnt
            next_pos += 1
            comp = next_pos < len(data) and bool(
                torch.all(current_v == data[sorted_id[next_pos]]).item()
            )
        rank_calculated: float = 1.0 * running_sum_r / (cmp_cnt + 1)
        for _ in range(cmp_cnt + 1):
            yield rank_calculated
        t_pos_r += cmp_cnt + 1


def _create_ranks(sorted_id: torch.Tensor, data: torch.Tensor, /) -> torch.Tensor:
    reverse_dict_id = {data_i: id_i for id_i, data_i in enumerate(sorted_id)}
    results = tuple(float(rank_v) for rank_v in _compare_rank(sorted_id, data))
    return torch.tensor(tuple(
        results[reverse_dict_id[data_id]]
        for data_id in sorted(reverse_dict_id)
    ), dtype=torch.float)


def merge_srcc(data: List[Tuple[TargetOutput, ...]], /) -> Tuple[str, str]:
    targets = torch.cat(tuple(data_el[0].target for data_el in data), 0).view(-1)
    t_sorted_id = targets.argsort(dim=0)
    t_ranks = _create_ranks(t_sorted_id, targets)
    mean_t_rank = t_ranks.mean()
    outputs = torch.cat(tuple(data_el[0].output for data_el in data), 0).view(-1)
    o_sorted_id = outputs.argsort(dim=0)
    o_ranks = _create_ranks(o_sorted_id, outputs)
    mean_o_ranks = o_ranks.mean()
    results = ((t_ranks - mean_t_rank) * (o_ranks - mean_o_ranks)).sum()
    div = ((t_ranks - mean_t_rank).pow(2).sum() * (o_ranks - mean_o_ranks).pow(2).sum()).sqrt()
    if not div.item():
        return '\"SRCC\":', '\"division by zero\"'
    return '\"SRCC\":', str(1. * results.item() / div.item())
