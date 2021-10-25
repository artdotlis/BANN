# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Optional, Tuple, List, Final, final

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from bann.b_container.constants.file_names import TestSubStrSuf
from bann.b_frameworks.pytorch.interfaces.activation_function_interface import AfcDataTC
from bann.b_test_train_prepare.pytorch.functions.dataset_splitter import test_k_folds_data_sets, \
    split_data_set
from bann.b_frameworks.pytorch.act_fun_lib import get_framework_act_lib
from bann.b_container.functions.dict_str_repr import dict_string_repr
from bann.b_pan_integration.framwork_key_lib import FrameworkKeyLib
from bann.b_test_train_prepare.pytorch.p_test.functions.deepcopy_id_file import deepcopy_id_file
from bann.b_test_train_prepare.pytorch.tester_interface import BTesterInterface
from bann.b_container.states.framework.interface.test_state import TestState
from bann.b_test_train_prepare.pytorch.tester_interface import TesterInterfaceArgs
from bann.b_container.states.framework.pytorch.p_test.p_test_general import TesterGeneral
from bann.b_test_train_prepare.errors.custom_errors import KnownTesterError
from bann.b_frameworks.pytorch.truth_fun_lib import get_framework_truth_lib

from pan.public.constants.test_net_stats_constants import TestNNStatsElementType, \
    create_test_net_stats

from rewowr.public.functions.syncout_dep_functions import logger_print_to_console
from rewowr.public.functions.worker_ctx import get_worker_ctx
from rewowr.public.interfaces.logger_interface import SyncStdoutInterface

_FRAMEWORK: Final[str] = FrameworkKeyLib.PYTORCH.value


@final
@dataclass
class _TestCont:
    use_cuda: bool
    torch_device: torch.device
    module: nn.Module
    truth_fun_id: str


@final
@dataclass
class _TestNetArgs:
    device: torch.device
    truth_fun_id: str
    last_layer: Tuple[str, Optional[int]]
    print_out: bool


def _create_output_str(in_t: Tuple[torch.Tensor, ...],
                       out_t: torch.Tensor, tar: torch.Tensor, /) -> List[str]:
    return [
        f"{';'.join(str(inp[i].tolist()) for inp in in_t)}\t{out_t[i].tolist()}\t{tar[i].tolist()}"
        for i in range(tar.size(0))
    ]


def _test_net_fun(model: nn.Module, data_loader_t: Tuple[DataLoader, ...],
                  args: _TestNetArgs, /) -> Tuple[float, List[str]]:
    model.eval()
    correct: int = 0
    test_cnt: int = 0
    print_out_l: List[str] = []
    tr_fun = get_framework_truth_lib(_FRAMEWORK).truth_fun(args.truth_fun_id)
    last_layer = get_framework_act_lib(_FRAMEWORK).act_b(args.last_layer[0])
    with torch.no_grad():
        for d_loader in data_loader_t:
            for data, target in d_loader:
                if isinstance(data, torch.Tensor):
                    dev_data: Tuple[torch.Tensor, ...] = (data.to(args.device), )
                elif isinstance(data, (tuple, list, set)):
                    dev_data = tuple(data_el.to(args.device) for data_el in data)
                else:
                    raise KnownTesterError(
                        f"Found unknown data type {type(data).__name__} for input"
                    )
                if isinstance(target, torch.Tensor):
                    target = target.to(args.device)
                else:
                    raise KnownTesterError(
                        f"Found unknown data type {type(target).__name__} for target"
                    )
                output = last_layer.act(AfcDataTC(
                    data=model(*dev_data), dim=args.last_layer[1]
                ))
                if args.print_out:
                    print_out_l.extend(_create_output_str(dev_data, output, target))
                test_cnt += target.size()[0]
                correct += tr_fun.calc_truth(
                    tr_fun.cr_truth_container(output, target, args.device)
                )

    truth_v = 1. * correct / test_cnt
    return truth_v, print_out_l


def _print_to_logger(sync_out: SyncStdoutInterface, tester_st: TesterGeneral,
                     args_te: TesterInterfaceArgs, use_cuda: bool, /) -> None:
    output_string = f"The arguments given to {SimpleTester.__name__}:\n"
    output_string += f"The {SimpleTester.__name__}:"
    output_string += f"\n\t{dict_string_repr(tester_st.get_kwargs().__dict__)}\n"

    output_string += f"The selected device:\t{'GPU' if args_te.cuda else 'CPU'}\n"
    output_string += f"The selection of the device:\t"
    output_string += f"{'failed' if args_te.cuda and not use_cuda else 'was successful'}\n"
    logger_print_to_console(sync_out, output_string)


def _create_res_str(erg_list: List[List[Tuple[float, List[str]]]], /) -> str:
    return '\n'.join(
        f"Run {index_run} k_fold {index_f} truth: {f_value[0]}"
        for index_run, run_v in enumerate(erg_list, 1)
        for index_f, f_value in enumerate(run_v, 1)
    )


@final
class SimpleTester(BTesterInterface):

    def __init__(self) -> None:
        super().__init__()
        self.__test_state: Optional[TesterGeneral] = None

    @property
    def test_state(self) -> TesterGeneral:
        if self.__test_state is None or not isinstance(self.__test_state, TesterGeneral):
            raise KnownTesterError("Test state was not set properly!")
        return self.__test_state

    def set_test_state(self, state: TestState, /) -> None:
        if not isinstance(state, TesterGeneral):
            raise KnownTesterError(
                f"Expected type {TesterGeneral.__name__} got {type(state).__name__}"
            )
        self.__test_state = state

    def _test_run(self, arg_con: _TestCont, input_data: Tuple[Dataset, ...], /) \
            -> Tuple[float, List[str]]:
        model = arg_con.module.to(arg_con.torch_device)
        test_loader = tuple(DataLoader(
            tes_e,
            batch_size=self.test_state.get_kwargs().batch_size,
            shuffle=False,
            pin_memory=arg_con.use_cuda,
            num_workers=self.test_state.get_kwargs().num_workers,
            drop_last=False
        ) for tes_e in input_data)
        if self.test_state.get_kwargs().num_workers > 0:
            for te_el in test_loader:
                te_el.multiprocessing_context = get_worker_ctx()
        if not test_loader:
            raise KnownTesterError("Empty data-loader!")
        return _test_net_fun(
            model, test_loader, _TestNetArgs(
                device=arg_con.torch_device,
                truth_fun_id=arg_con.truth_fun_id,
                last_layer=self.test_state.get_kwargs().last_layer,
                print_out=self.test_state.get_kwargs().print_results
            )
        )

    def test(self, sync_out: SyncStdoutInterface, args: TesterInterfaceArgs, /) -> \
            Tuple[TestNNStatsElementType, ...]:
        if not args.input_test:
            return ()
        use_cuda = args.cuda and torch.cuda.is_available()
        _print_to_logger(sync_out, self.test_state, args, use_cuda)
        test_cont = _TestCont(
            use_cuda=use_cuda,
            torch_device=torch.device("cuda" if use_cuda else "cpu"),
            module=args.module.get_net_com,
            truth_fun_id=args.truth_fun_id
        )
        if self.test_state.get_kwargs().n_repeats > 0:
            if self.test_state.get_kwargs().k_folds < 2:
                self.test_state.get_kwargs().k_folds = 2
            test_k_folds_data_sets(self.test_state.get_kwargs().k_folds, args.input_test)
            erg_list: List[List[Tuple[float, List[str]]]] = []
            for _ in range(self.test_state.get_kwargs().n_repeats):
                merged_subsets = [
                    split_data_set(self.test_state.get_kwargs().k_folds, data_set)
                    for data_set in args.input_test
                ]
                erg_list.append([
                    self._test_run(
                        test_cont, tuple(data_sets[k_fold] for data_sets in merged_subsets)
                    )
                    for k_fold in range(self.test_state.get_kwargs().k_folds)
                ])
            results_list = [
                create_test_net_stats(
                    deepcopy_id_file(args.id_file, TestSubStrSuf.STRUCT.value),
                    [repr(test_cont.module)], "txt"
                ), create_test_net_stats(
                    deepcopy_id_file(args.id_file, TestSubStrSuf.RESULT.value),
                    ["{", f"\"Truth\":\n{_create_res_str(erg_list)}", "}"], "json"
                )
            ]
            if self.test_state.get_kwargs().print_results:
                for run_i, run_v in enumerate(erg_list):
                    for pr_i, print_value in enumerate(run_v):
                        results_list.append(create_test_net_stats(
                            deepcopy_id_file(
                                args.id_file, f"{TestSubStrSuf.OUTPUT.value}_r_{run_i}_k_{pr_i}"
                            ),
                            list(print_value[1]), "txt"
                        ))
            return tuple(results_list)
        buf_res = self._test_run(test_cont, args.input_test)
        results_list = [
            create_test_net_stats(
                deepcopy_id_file(args.id_file, TestSubStrSuf.STRUCT.value),
                [repr(test_cont.module)], "txt"
            ), create_test_net_stats(
                deepcopy_id_file(args.id_file, TestSubStrSuf.RESULT.value),
                ["{", f"\"Truth\":\n{buf_res[0]}", "}"], "json"
            )
        ]
        if self.test_state.get_kwargs().print_results:
            results_list.append(create_test_net_stats(
                deepcopy_id_file(args.id_file, TestSubStrSuf.OUTPUT.value),
                list(buf_res[1]), "txt"
            ))
        return tuple(results_list)
