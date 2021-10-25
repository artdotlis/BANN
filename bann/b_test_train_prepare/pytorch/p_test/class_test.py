# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from dataclasses import dataclass
from typing import Optional, Tuple, Final, final, List, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from bann.b_container.constants.file_names import TestSubStrSuf
from bann.b_test_train_prepare.pytorch.p_test.functions.deepcopy_id_file import deepcopy_id_file
from bann.b_test_train_prepare.pytorch.p_test.libs.class_test_lib import one_class_run_test, \
    one_class_calc_stats
from bann.b_test_train_prepare.pytorch.p_test.libs.container import OneClassOutTar
from bann.b_frameworks.pytorch.interfaces.activation_function_interface import AfcDataTC
from bann.b_frameworks.pytorch.p_truth.p_truth_enum import PTruthId
from bann.b_test_train_prepare.pytorch.p_test.functions.one_class import one_class_stats, \
    merge_one_class_stats
from bann.b_test_train_prepare.pytorch.functions.dataset_splitter import test_k_folds_data_sets, \
    split_data_set
from bann.b_frameworks.pytorch.act_fun_lib import get_framework_act_lib
from bann.b_frameworks.pytorch.truth_fun_lib import get_framework_truth_lib
from bann.b_container.states.framework.interface.test_state import TestState
from bann.b_test_train_prepare.errors.custom_errors import KnownTesterError
from bann.b_container.functions.dict_str_repr import dict_string_repr
from bann.b_container.states.framework.pytorch.p_test.p_test_class import TesterClass
from bann.b_test_train_prepare.pytorch.tester_interface import TesterInterfaceArgs, \
    BTesterInterface
from bann.b_pan_integration.framwork_key_lib import FrameworkKeyLib

from pan.public.constants.test_net_stats_constants import TestNNStatsElementType, \
    create_test_net_stats
from rewowr.public.functions.syncout_dep_functions import logger_print_to_console
from rewowr.public.functions.worker_ctx import get_worker_ctx
from rewowr.public.interfaces.logger_interface import SyncStdoutInterface

_FRAMEWORK: Final[str] = FrameworkKeyLib.PYTORCH.value


@final
@dataclass
class _ECont:
    truth_fun_id: str
    last_layer: Tuple[str, Optional[int]]
    class_num: int
    step_cnt: int
    tests: Tuple[str, ...]


@final
@dataclass
class _TestCont:
    use_cuda: bool
    torch_device: torch.device
    module: nn.Module
    truth_fun_id: str


@final
@dataclass
class _LCounter:
    correct: int = 0
    test_cnt: int = 0


def _test_net_fun(model: nn.Module, data_loader_t: Tuple[DataLoader, ...],
                  device: torch.device, extra_con: _ECont, /) -> Tuple[str, ...]:
    model.eval()
    counter = _LCounter()
    class_spec = []
    results: Dict[str, List[Tuple]] = {}
    tr_fun = get_framework_truth_lib(_FRAMEWORK).truth_fun(extra_con.truth_fun_id)
    last_layer = get_framework_act_lib(_FRAMEWORK).act_b(extra_con.last_layer[0])
    with torch.no_grad():
        for data_loader in data_loader_t:
            for data, target in data_loader:
                if isinstance(data, torch.Tensor):
                    dev_data: Tuple[torch.Tensor, ...] = (data.to(device), )
                elif isinstance(data, (tuple, list, set)):
                    dev_data = tuple(data_el.to(device) for data_el in data)
                else:
                    raise KnownTesterError(
                        f"Found unknown data type {type(data).__name__} for input data"
                    )
                if isinstance(target, torch.Tensor):
                    target = target.to(device)
                else:
                    raise KnownTesterError(
                        f"Found unknown data type {type(target).__name__} for target"
                    )
                output = model(*dev_data)
                counter.test_cnt += target.size()[0]
                counter.correct += tr_fun.calc_truth(
                    tr_fun.cr_truth_container(
                        last_layer.act(
                            AfcDataTC(data=output, dim=extra_con.last_layer[1])
                        ), target, device
                    )
                )
                class_spec.append(one_class_stats(
                    last_layer.act(
                        AfcDataTC(data=output, dim=extra_con.last_layer[1])
                    ), target, device, extra_con.class_num
                ))
                for test_n in extra_con.tests:
                    results.setdefault(test_n, []).append(one_class_calc_stats(
                        test_n, extra_con.class_num, extra_con.step_cnt, device, OneClassOutTar(
                            output=output, target=target,
                            output_act=last_layer.act(
                                AfcDataTC(data=output, dim=extra_con.last_layer[1])
                            )
                        )
                    ))
    truth_v = f"\"Accuracy\": {1. * counter.correct / counter.test_cnt},"
    res_list_c = []
    for test_n in extra_con.tests:
        res_list_c.append(f"\"Test_{test_n}\":" + "{")
        res_list_c.extend(
            res_v
            for res_v in one_class_run_test(
                test_n, extra_con.class_num, extra_con.step_cnt, results
            )
        )
        res_list_c.append("},")
    res_list_c[-1] = "}"
    return (
        truth_v, merge_one_class_stats(class_spec, extra_con.class_num) + ",",
        *res_list_c
    )


def _print_to_logger(sync_out: SyncStdoutInterface, tester_st: TesterClass,
                     args_te: TesterInterfaceArgs, use_cuda: bool, /) -> None:
    output_string = f"The arguments given to {ClassTester.__name__}:\n"
    output_string += f"The {ClassTester.__name__}:"
    output_string += f"\n\t{dict_string_repr(tester_st.get_kwargs().__dict__)}\n"

    output_string += f"The selected device:\t{'GPU' if args_te.cuda else 'CPU'}\n"
    output_string += f"The selection of the device:\t"
    output_string += f"{'failed' if args_te.cuda and not use_cuda else 'was successful'}\n"
    logger_print_to_console(sync_out, output_string)


@final
class ClassTester(BTesterInterface):
    def __init__(self) -> None:
        super().__init__()
        self.__test_state: Optional[TesterClass] = None

    @property
    def test_state(self) -> TesterClass:
        if self.__test_state is None or not isinstance(self.__test_state, TesterClass):
            raise KnownTesterError("Test state was not set properly!")
        return self.__test_state

    def set_test_state(self, state: TestState, /) -> None:
        if not isinstance(state, TesterClass):
            raise KnownTesterError(
                f"Expected type {TesterClass.__name__} got {type(state).__name__}"
            )
        self.__test_state = state

    def _test_run(self, arg_con: _TestCont, input_data: Tuple[Dataset, ...], /) -> Tuple[str, ...]:
        model = arg_con.module.to(arg_con.torch_device)
        test_loader = tuple(DataLoader(
            in_t,
            batch_size=self.test_state.get_kwargs().batch_size,
            shuffle=False,
            pin_memory=arg_con.use_cuda,
            num_workers=self.test_state.get_kwargs().num_workers,
            drop_last=False
        ) for in_t in input_data)
        if self.test_state.get_kwargs().num_workers > 0:
            for te_el in test_loader:
                te_el.multiprocessing_context = get_worker_ctx()
        if not test_loader:
            raise KnownTesterError("Empty data-loader!")
        return _test_net_fun(model, test_loader, arg_con.torch_device, _ECont(
            truth_fun_id=arg_con.truth_fun_id,
            last_layer=self.test_state.get_kwargs().last_layer,
            class_num=self.test_state.get_kwargs().class_num,
            step_cnt=self.test_state.get_kwargs().step_cnt,
            tests=self.test_state.get_kwargs().tests
        ))

    def test(self, sync_out: SyncStdoutInterface, args: TesterInterfaceArgs, /) -> \
            Tuple[TestNNStatsElementType, ...]:
        if not args.input_test:
            return ()
        if args.truth_fun_id != PTruthId.ONECLASS.value:
            raise KnownTesterError(
                f"{ClassTester.__name__} works only with {PTruthId.ONECLASS.value}, "
                + f"but got {args.truth_fun_id}!"
            )
        use_cuda = args.cuda and torch.cuda.is_available()
        torch_device = torch.device("cuda" if use_cuda else "cpu")
        _print_to_logger(sync_out, self.test_state, args, use_cuda)
        test_cont = _TestCont(
            use_cuda=use_cuda,
            torch_device=torch_device,
            module=args.module.get_net_com,
            truth_fun_id=args.truth_fun_id
        )
        if self.test_state.get_kwargs().n_repeats > 0:
            if self.test_state.get_kwargs().k_folds < 2:
                self.test_state.get_kwargs().k_folds = 2
            test_k_folds_data_sets(self.test_state.get_kwargs().k_folds, args.input_test)
            erg_list = []
            for run_id in range(self.test_state.get_kwargs().n_repeats):
                merged_subsets = [
                    split_data_set(self.test_state.get_kwargs().k_folds, data_set)
                    for data_set in args.input_test
                ]
                for k_fold in range(self.test_state.get_kwargs().k_folds):
                    erg_list.append(f"\"Run_{run_id + 1}_k_fold_{k_fold + 1}_results\":" + "{")
                    erg_list.extend(self._test_run(
                        test_cont, tuple(data_sets[k_fold] for data_sets in merged_subsets)
                    ))
                    erg_list.append("},")
            erg_list[-1] = "}"
            return (create_test_net_stats(
                deepcopy_id_file(args.id_file, TestSubStrSuf.STRUCT.value),
                [repr(test_cont.module)], "txt"
            ), create_test_net_stats(
                deepcopy_id_file(args.id_file, TestSubStrSuf.RESULT.value),
                ["{", *erg_list, "}"], "json"
            ))

        return (create_test_net_stats(
            deepcopy_id_file(args.id_file, TestSubStrSuf.STRUCT.value),
            [repr(test_cont.module)], "txt"
        ),  create_test_net_stats(
            deepcopy_id_file(args.id_file, TestSubStrSuf.RESULT.value),
            ["{", *self._test_run(test_cont, args.input_test), "}"], "json"
        ))
