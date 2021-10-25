# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from copy import deepcopy
from typing import Tuple, Dict, List, Iterable, final

from bann.b_hyper_optim.errors.custom_erors import KnownDPError


@final
class DPMatrix:
    def __init__(self, memory: int, repeats: int, /) -> None:
        super().__init__()
        self.__max_mem = memory
        if memory < 0:
            self.__max_mem = 0
        self.__repeats = repeats
        if repeats < 1:
            self.__repeats = 1
        self.__memory: Dict[Tuple[float, ...], Tuple[float, int, Tuple[float, ...]]] = {}
        self.__last_saved_ids: List[bool] = []
        self.__last_saved: List[Tuple[float, ...]] = []

    def _create_keys_to_rem(self) -> Iterable[Tuple[float, ...]]:
        cnt = 0
        for key_i in self.__memory.keys():
            if cnt < len(self.__memory) - self.__max_mem / 2:
                yield key_i
            cnt += 1

    def _remove_half(self) -> None:
        if len(self.__memory) >= self.__max_mem:
            for key_d in list(self._create_keys_to_rem()):
                del self.__memory[key_d]

    def cr_hyper_params(self, hyper_params: List[Tuple[float, ...]], /) -> List[Tuple[float, ...]]:
        if not self.__max_mem:
            return hyper_params
        self.__last_saved_ids = [
            hyp_p in self.__memory and self.__memory[hyp_p][1] >= self.__repeats
            for hyp_p in hyper_params
        ]
        self.__last_saved = deepcopy(hyper_params)
        return [
            hyp_p for hyp_i, hyp_p in enumerate(hyper_params) if not self.__last_saved_ids[hyp_i]
        ]

    def _update_value(self, dict_key: Tuple[float, ...], fit: float,
                      new_val: Tuple[float, ...], /) -> None:
        values = self.__memory.get(dict_key, (float('inf'), 0))
        new_cnt = values[1] + 1
        new_fit = values[0]
        if new_fit > fit:
            new_fit = fit
        self.__memory[dict_key] = (new_fit, new_cnt, tuple() if dict_key == new_val else new_val)

    def _update_dict(self, new_fit_val: List[Tuple[float, Tuple[float, ...]]], /) -> None:
        cnt = 0
        for l_i, l_id in enumerate(self.__last_saved_ids):
            if not l_id:
                dict_key = self.__last_saved[l_i]
                self._update_value(dict_key, new_fit_val[cnt][0], new_fit_val[cnt][1])
                if self.__memory[dict_key][2] and dict_key != self.__memory[dict_key][2]:
                    self._update_value(
                        self.__memory[dict_key][2], new_fit_val[cnt][0], self.__memory[dict_key][2]
                    )
                cnt += 1

    def up_hyper_params(self, new_fit_val: List[Tuple[float, Tuple[float, ...]]], /) \
            -> List[Tuple[float, Tuple[float, ...]]]:
        if not self.__max_mem:
            return new_fit_val
        if self.__last_saved_ids:
            self._update_dict(new_fit_val)
            new_list = [
                (
                    self.__memory[hyp_p][0],
                    self.__memory[hyp_p][2] if self.__memory[hyp_p][2] else deepcopy(hyp_p)
                )
                for hyp_p in self.__last_saved
            ]
            self._remove_half()
            self.__last_saved_ids = []
            self.__last_saved = []
            return new_list
        raise KnownDPError("saved id list should never be empty!")
