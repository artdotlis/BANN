# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import Dict, List, Tuple

import numpy  # type: ignore


def _calc_auc_right(y_l: List[float], x_l_ex: List[float], /) -> float:
    res: float = sum(
        (x_l_ex[index + 1] - x_elem) * y_l[index]
        for index, x_elem in enumerate(x_l_ex[:-1])
    )
    return res


def _calc_auc_left(y_l: List[float], x_l_ex: List[float], /) -> float:
    res: float = sum(
        (x_elem - x_l_ex[index - 1]) * y_l[index]
        for index, x_elem in enumerate(x_l_ex[1:], 1)
    )
    return res


def _fix_roc(x_l: List[float], x_l_ex: List[float], y_l: List[float], /) -> None:
    if x_l[0] > 0.0:
        x_l.insert(0, 0.0)
        x_l_ex.insert(0, 0.0)
        y_l.insert(0, 0.0)

    if x_l[-1] < 1.0:
        x_l.append(1.0)
        x_l_ex.append(1.0)
        y_l.append(1.0)


def _fix_prc(x_l: List[float], x_l_ex: List[float], y_l: List[float], /) -> None:
    if x_l[0] > 0.0:
        x_l.insert(0, 0.0)
        x_l_ex.insert(0, 0.0)
        y_l.insert(0, 1.0)

    if x_l[-1] < 1.0:
        x_l.append(1.0)
        x_l_ex.append(1.0)
        y_l.append(0.0)


def calc_auc(x_data_d: Dict[str, List[float]], rev: bool, /) -> Tuple[float, float]:
    x_l = sorted([float(elem) for elem in x_data_d.keys()])
    x_l_ex = [x_v for x_v in x_l for _ in x_data_d[str(x_v)]]
    y_l = [y_v for x_v in x_l for y_v in sorted(x_data_d[str(x_v)], reverse=rev)]
    if rev:
        _fix_prc(x_l, x_l_ex, y_l)
        step_auc = _calc_auc_left(y_l, x_l_ex)
    else:
        _fix_roc(x_l, x_l_ex, y_l)
        step_auc = _calc_auc_right(y_l, x_l_ex)
    trapz_auc: float = numpy.trapz(y=y_l, x=x_l_ex)
    return trapz_auc, step_auc
