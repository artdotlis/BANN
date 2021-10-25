# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import math
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple, Optional, final, Iterable
from dataclasses import dataclass

import numpy as np  # type: ignore
import seaborn as sns  # type: ignore
import pandas as pd  # type: ignore
from pandas.core.frame import DataFrame as DataFrameType  # type: ignore
import matplotlib  # type: ignore
from matplotlib import pyplot  # type: ignore

from bann.b_pan_integration.errors.custom_erors import NetPlotterError

from pan.public.constants.train_net_stats_constants import TrainNNPlotterStatDictSeriesType
from pan.public.interfaces.net_plotter_interface import NetXYPlotter


matplotlib.use('agg')
SeabornColorPalette = List[Tuple[int, int, int]]


@final
@dataclass
class SnsPlottableData:
    x_label: str
    y_label: str
    hue: str
    row: str
    title: str
    sub_title: str
    row_num: int
    series_num: int
    data_frame: DataFrameType
    to_log_x: bool = False
    to_log_y: Optional[List[bool]] = None


@final
@dataclass
class _ThreeM:
    y_min: float
    y_max: float
    median: float


def _squash_y_value(y_value: float, median: float, abs_dif: float, /) -> float:
    upper_l = median + abs_dif
    lower_l = median - abs_dif
    if lower_l <= y_value <= upper_l:
        return y_value
    if y_value > upper_l:
        return upper_l + abs_dif / 10
    if y_value < lower_l:
        return lower_l - abs_dif / 10
    raise NetPlotterError("Should never happen")


def _iterable_gen_data(row_name: str, data_copy: SnsPlottableData,
                       data_row_dict: Dict[str, List[Tuple[int, int]]], /) -> Iterable[float]:
    for start_row, row_i in data_row_dict[row_name]:
        for y_val in data_copy.data_frame[data_copy.y_label][start_row:row_i]:
            yield y_val


def _calc_min_max_median(row_name: str, data_copy: SnsPlottableData,
                         data_row_dict: Dict[str, List[Tuple[int, int]]], /) -> _ThreeM:
    y_min = min(_iterable_gen_data(row_name, data_copy, data_row_dict))
    y_max = max(_iterable_gen_data(row_name, data_copy, data_row_dict))
    median = float(np.median(list(_iterable_gen_data(row_name, data_copy, data_row_dict))))
    return _ThreeM(y_min=y_min, y_max=y_max, median=median)


def _modify_plottable_data_fun(data: SnsPlottableData, /) -> SnsPlottableData:
    data_copy = deepcopy(data)
    x_min = min(data_copy.data_frame[data_copy.x_label])
    x_space = abs(x_min - max(data_copy.data_frame[data_copy.x_label]))

    if x_space >= 200:
        data_copy.to_log_x = True
        data_copy.data_frame[data_copy.x_label] = [
            math.log10(x_value - x_min + 1) for x_value in data.data_frame[data.x_label]
        ]
    start_row = 0
    row_name: Optional[str] = None
    data_copy.to_log_y = []
    data_row_len = len(data_copy.data_frame[data_copy.row])
    data_row_dict: Dict[str, List[Tuple[int, int]]] = {}
    data_row_sorted: List[str] = []
    for row_i in range(data_row_len + 1):
        if row_name is None and row_i < data_row_len:
            row_name = data_copy.data_frame[data_copy.row][row_i]
        if row_name is not None and (
                row_i >= data_row_len
                or row_name != data_copy.data_frame[data_copy.row][row_i]
        ):
            if row_name not in data_row_dict:
                data_row_sorted.append(row_name)
            data_row_dict.setdefault(row_name, []).append((start_row, row_i))
            start_row = row_i
            if row_i < data_row_len:
                row_name = data_copy.data_frame[data_copy.row][row_i]
    for row_name in data_row_sorted:
        m_three = _calc_min_max_median(row_name, data_copy, data_row_dict)
        data_copy.to_log_y.append(m_three.median <= 1 and m_three.y_min > 0)
        for start_row, row_i in data_row_dict[row_name]:
            if m_three.median <= 1 and m_three.y_min > 0:
                data_copy.data_frame.loc[start_row:row_i-1, data_copy.y_label] = [
                    math.log10(y_value)
                    for y_value in data.data_frame[data.y_label][start_row:row_i]
                ]
            else:
                abs_min = min(
                    abs(m_three.median - m_three.y_min), abs(m_three.median - m_three.y_max)
                )
                data_copy.data_frame.loc[start_row:row_i-1, data_copy.y_label] = [
                    _squash_y_value(y_value, m_three.median, abs_min)
                    if abs_min > 0 else y_value
                    for y_value in data.data_frame[data.y_label][start_row:row_i]
                ]

    return data_copy


def _extend_xy_values_col(x_to_extend: List[float], y_to_extend: List[float],
                          series_sub_name_col: List[str],
                          data: TrainNNPlotterStatDictSeriesType, to_sort: bool, /) -> None:
    if to_sort:
        sorted_key_list = list(data.data_elem_dict.keys())
        sorted_key_list.sort()
    else:
        sorted_key_list = list(range(data.last_index + 1))
    x_to_extend.extend(
        x_elem
        for pos in sorted_key_list
        for x_elem in data.data_elem_dict[pos].x_cords
    )
    y_to_extend.extend(
        y_elem
        for pos in sorted_key_list
        for y_elem in data.data_elem_dict[pos].y_cords
    )
    series_sub_name_col.extend(
        data.data_elem_dict[pos].info.name_sub_series
        for pos in sorted_key_list
        for _ in data.data_elem_dict[pos].x_cords
    )


def _create_plottable_data_fun(data_series: Dict[str, TrainNNPlotterStatDictSeriesType],
                               to_sort: bool, /) -> SnsPlottableData:
    random_ser = next(iter(data_series.values()))
    random_point = next(iter(random_ser.data_elem_dict.values()))
    x_values_col: List[float] = []
    y_values_col: List[float] = []
    series_name_col: List[str] = []
    series_sub_name_col: List[str] = []
    for ser_name, data in data_series.items():
        added_num = len(x_values_col)
        _extend_xy_values_col(x_values_col, y_values_col, series_sub_name_col, data, to_sort)
        series_name_col.extend(ser_name for _ in range(len(x_values_col) - added_num))
    return SnsPlottableData(
        x_label=random_point.info.x_label,
        y_label=random_point.info.y_label,
        hue=random_point.info.type_series,
        row=f"sub_{random_point.info.type_sub_series}",
        title=random_point.info.title,
        sub_title=random_point.info.subtitle,
        row_num=len({row_n: True for row_n in series_sub_name_col}),
        series_num=len(data_series),
        data_frame=pd.DataFrame({
            str(random_point.info.x_label): x_values_col,
            str(random_point.info.y_label): y_values_col,
            str(random_point.info.type_series): series_name_col,
            f"sub_{random_point.info.type_sub_series}": series_sub_name_col
        })
    )


@final
class _SeabornPlotter(NetXYPlotter[SnsPlottableData, SnsPlottableData]):
    def create_plottable_dump_data(
            self, data_series: Dict[str, TrainNNPlotterStatDictSeriesType], /
    ) -> SnsPlottableData:
        return _create_plottable_data_fun(data_series, True)

    def create_writable_dump_data(
            self, data_series: Dict[str, TrainNNPlotterStatDictSeriesType], /
    ) -> SnsPlottableData:
        return _create_plottable_data_fun(data_series, True)

    def create_plottable_data(self, data_series: Dict[str, TrainNNPlotterStatDictSeriesType], /) \
            -> SnsPlottableData:
        return _create_plottable_data_fun(data_series, False)

    def create_writable_data(self, data_series: Dict[str, TrainNNPlotterStatDictSeriesType], /) \
            -> SnsPlottableData:
        return _create_plottable_data_fun(data_series, False)

    def plot_data(self, file_name: str, data: SnsPlottableData, /) -> None:
        pyplot.ioff()
        new_data = _modify_plottable_data_fun(data)
        height = new_data.row_num * 4
        aspect = 12 / height
        palette = sns.color_palette("bright", new_data.series_num)
        grid = sns.FacetGrid(
            new_data.data_frame, row=new_data.row, margin_titles=False,
            height=height, aspect=aspect, legend_out=False,
            palette=palette, hue=new_data.hue, sharey=False
        )
        grid.map(sns.lineplot, new_data.x_label, new_data.y_label)
        if data.series_num > 1:
            grid.add_legend()
        for x_y_i, x_y_axes in enumerate(grid.axes):
            pyplot.setp(x_y_axes[0].lines, linewidth=0.75)
            if new_data.to_log_x:
                x_y_axes[0].xaxis.set_major_formatter(pyplot.FuncFormatter(
                    lambda l_va, tick_number:
                    f"${str(math.pow(10, l_va - math.floor(l_va)))[:4]}x10^{{{math.floor(l_va)}}}$"
                ))
            if new_data.to_log_y is not None and new_data.to_log_y[x_y_i]:
                x_y_axes[0].set_yticklabels(
                    x_y_axes[0].get_yticklabels(), rotation=45, horizontalalignment='right',
                    fontsize=8
                )
                x_y_axes[0].yaxis.set_major_formatter(pyplot.FuncFormatter(
                    lambda l_va, tick_number:
                    f"${str(math.pow(10, l_va - math.floor(l_va)))[:4]}x10^{{{math.floor(l_va)}}}$"
                ))
        grid.set_axis_labels(x_var=new_data.x_label, y_var=new_data.y_label)
        grid.set_titles(new_data.title + " {row_name}")
        pyplot.tight_layout()
        pyplot.savefig(f"{file_name}_plot.png")
        pyplot.close('all')

    def dump_plot(self, file_name: str, data: SnsPlottableData, /) -> None:
        file = Path(f"{file_name}_plot.png")
        if file.exists() and file.is_file():
            file.unlink()
        self.plot_data(file_name, data)

    def write_data(self, file_name: str, data: SnsPlottableData, extra_str: str, /) -> None:
        csv = data.data_frame.to_csv(
            path_or_buf=None,
            index=False,
            header=True,
            sep='\t',
            encoding='utf-8'
        )
        with Path(f"{file_name}_plot_data.csv").open("w") as wr_handler:
            wr_handler.write(csv)

        if extra_str:
            with Path(f"{file_name}_hyper_param.txt").open("w") as wr_handler:
                wr_handler.write(extra_str)

    def dump_data(self, file_name: str, data: SnsPlottableData, extra_str: str, /) -> None:
        file = Path(f"{file_name}_plot_data.csv")
        if file.exists() and file.is_file():
            file.unlink()
        self.write_data(file_name, data, extra_str)


def net_plotter() -> NetXYPlotter:
    return _SeabornPlotter()
