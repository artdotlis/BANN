# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from copy import deepcopy
from typing import Tuple, List, Iterable, final

from bann.b_hyper_optim.optimiser.fun_const.hyper_const import SimplyTrainOArgs


@final
class SimplyTrainCont:
    def __init__(self, arg_con: SimplyTrainOArgs, params: Tuple[float, ...], /) -> None:
        super().__init__()
        self._arg_con = arg_con
        self._hyper_param_flat: Tuple[float, ...] = deepcopy(params)

    def simply_copy_params(self) -> Iterable[List[Tuple[float, ...]]]:
        run_cnt = 0
        while run_cnt < self._arg_con.repeats:
            erg = [deepcopy(self._hyper_param_flat) for _ in range(self._arg_con.package)]
            yield erg
            run_cnt += 1
