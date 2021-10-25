# -*- coding: utf-8 -*-
"""inspired by:
  - https://github.com/pytorch/examples/blob/master/mnist/main.py

LICENSE (BSD 3-Clause License): see extra_licenses/LICENSE_P_EXAMPLES

.. moduleauthor:: Artur Lissin
"""
import sys

from typing import Tuple, Optional, Iterable, final

from torch import nn
from torch.utils.data import Dataset
import torchvision  # type: ignore

from bann.b_frameworks.pytorch.pytorch_lego_const import LegoContInit
from bann.b_frameworks.pytorch.p_truth.p_truth_enum import PTruthId
from bann.b_container.functions.print_init_net_state import print_init_net_states
from bann.b_data_functions.data_loader_percent_wrapper import RedirectWriteToLoggerPercent
from bann.b_container.states.general.g_init.init_general import InitGeneralState
from bann.b_frameworks.pytorch.net_model_interface import CurrentNetData, NetModelInterface
from bann.b_frameworks.pytorch.truth_fun_lib import get_framework_truth_lib
from bann.b_container.functions.pytorch.init_framework_fun import InitNetArgs, init_wrapper
from bann.b_frameworks.pytorch.network_nodes.simple_net_abstract import SimpleAnnNet, \
    SimpleNetCon, get_simple_net_framework, DataSetTypes
from bann.b_frameworks.pytorch.net_model_interface import InitContainer

from bann_demo.pytorch.networks.b_nets.complex_demo import create_current_complex_demo_net, \
    BComplexDemoInitCon, load_current_complex_demo
from bann_demo.pytorch.connections.demo_connection import get_demo_connection
from bann_demo.pytorch.states.net_complex import NetComplexState
from bann_demo.pytorch.networks.errors.custom_erors import KnownComplexDemoError
from bann_demo.pytorch.networks.simple_net_demo import SimpleDemoUpdater

from bann_ex_con.pytorch.external_enum import ENetInterfaceNames

from pan.public.constants.net_tree_id_constants import ANNTreeIdType
from pan.public.constants.train_net_stats_constants import TrainNNStatsElementType
from pan.public.functions.load_net_function import net_interface_load_net
from pan.public.interfaces.pub_net_interface import NodeANNDataElemInterface, NetSavable, \
    NetSavableArgs
from pan.public.interfaces.net_connection import NetConnectionWr

from rewowr.public.functions.path_functions import create_dirs_rec
from rewowr.public.functions.syncout_dep_functions import logger_print_to_console
from rewowr.public.interfaces.logger_interface import SyncStdoutInterface


@final
class ComplexDemoCon(SimpleNetCon[Optional[Tuple[NodeANNDataElemInterface, ...]]]):
    def __init__(self, internal_nets: Tuple[NodeANNDataElemInterface, ...], /) -> None:
        self.__children_nets: Optional[Tuple[NodeANNDataElemInterface, ...]] = internal_nets
        super().__init__()

    @property
    def children_nets(self) -> Optional[Tuple[NodeANNDataElemInterface, ...]]:
        return self.__children_nets

    def remove_before_save(self) -> Optional[Tuple[NodeANNDataElemInterface, ...]]:
        buf = self.__children_nets
        self.__children_nets = None
        return buf

    def reload_after_save(self, data: Optional[Tuple[NodeANNDataElemInterface, ...]], /) -> None:
        self.__children_nets = data

    @property
    def lego_init_cont(self) -> LegoContInit:
        raise KnownComplexDemoError(f"{type(self).__name__} is not a lego net!")

    def _create_current_net(self) -> CurrentNetData:
        if self.__children_nets is None:
            raise KnownComplexDemoError("Children were not set!")
        created_net = create_current_complex_demo_net(self.__children_nets)
        return created_net

    def _create_current_loaded_net(self, extra_args: InitContainer, /) -> CurrentNetData:
        if not isinstance(extra_args, BComplexDemoInitCon):
            raise KnownComplexDemoError(
                f"Expected args type {BComplexDemoInitCon.__name__} got {type(extra_args).__name__}"
            )
        return load_current_complex_demo(extra_args)

    def hyper_update(self, data_up: Tuple[float, ...], /) -> None:
        if data_up:
            raise KnownComplexDemoError(f"Expected empty tuple got {len(data_up)} elements.")


@final
class ComplexDemoNet(SimpleAnnNet[Optional[Tuple[NodeANNDataElemInterface, ...]]]):

    def re_read_data(self, data_type: DataSetTypes, /) -> Optional[Tuple[Dataset, ...]]:
        return None

    def __init__(self, args: InitNetArgs, /) -> None:
        super().__init__(args)
        self.__connection_cnt = self.check_net_state().get_kwargs().children_cnt

    def get_truth_fun_id(self) -> str:
        truth_id = PTruthId.ONECLASS.value
        if not get_framework_truth_lib(
                get_simple_net_framework()
        ).truth_fun_check(truth_id):
            raise KnownComplexDemoError("Could not find the needed truth function!")
        return truth_id

    @property
    def connection_in(self) -> NetConnectionWr:
        return NetConnectionWr("", "")

    @property
    def connection_out(self) -> Tuple[NetConnectionWr, ...]:
        return tuple(
            get_demo_connection(get_simple_net_framework()) for _ in range(self.__connection_cnt)
        )

    def check_net(self, internal_nets: Tuple[NodeANNDataElemInterface, ...],
                  sync_out: SyncStdoutInterface, /) -> None:
        _ = self.init_net(internal_nets, sync_out)
        state_args = self.check_init_state()
        in_tr_path = state_args.get_kwargs().input_train
        in_e_path = state_args.get_kwargs().input_eval
        in_te_path = state_args.get_kwargs().input_test
        if not (in_tr_path is not None and in_tr_path.is_absolute()):
            raise KnownComplexDemoError("Path for train-data is missing!")

        if not (in_e_path is not None and in_e_path.is_absolute()):
            raise KnownComplexDemoError("Path for evaluation-data is missing!")

        if not (in_te_path is not None and in_te_path.is_absolute()):
            raise KnownComplexDemoError("Path for test-data is missing!")

    def _init_data_load(self, state_args: InitGeneralState, /) -> None:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])
        in_tr_path = state_args.get_kwargs().input_train
        in_e_path = state_args.get_kwargs().input_eval
        if in_tr_path is not None and in_e_path is not None:
            create_dirs_rec(in_tr_path)
            self.train_data_set((torchvision.datasets.MNIST(
                root=str(in_tr_path),
                train=True, download=True, transform=transform
            ),))
            create_dirs_rec(in_e_path)
            self.eval_data_set((torchvision.datasets.MNIST(
                root=str(in_e_path),
                train=False, download=True, transform=transform
            ),))
        in_te_path = state_args.get_kwargs().input_test
        if in_te_path is not None:
            create_dirs_rec(in_te_path)
            self.test_data_set((torchvision.datasets.MNIST(
                root=str(in_te_path),
                train=False, download=True, transform=transform
            ),))

    def _create_net(self, state_args: InitGeneralState, net_state: NetComplexState,
                    internal_nets: Tuple[NodeANNDataElemInterface, ...], /) \
            -> Tuple[bool, NetModelInterface]:
        net_load = state_args.get_kwargs().input_net
        if net_load is not None:
            if net_load.is_absolute() and net_load.exists() and net_load.is_file():
                self.retrain_set(net_state.get_kwargs().retrain)
                with net_load.open('rb') as net_handler:
                    buffer_load = net_interface_load_net(
                        type(self), net_handler.read()
                    )
                    if not isinstance(buffer_load, NetSavable):
                        raise KnownComplexDemoError(
                            f"Expected net {NetSavable.__name__} "
                            + f" got {type(buffer_load).__name__}!"
                        )
                    current_net = buffer_load.ann_container
                    if not isinstance(current_net, ComplexDemoCon):
                        raise KnownComplexDemoError(
                            f"Expected net {ComplexDemoCon.__name__} "
                            + f" got {type(current_net).__name__}!"
                        )
                    self.net_module_set(current_net)
                    self.savable_set(buffer_load)
                    return False, current_net
            raise KnownComplexDemoError("The net to load was not found!")
        if self.net_module is None:
            if not internal_nets:
                raise KnownComplexDemoError("Expected to have children got none!")
            current_net = ComplexDemoCon(internal_nets)
            self.net_module_set(current_net)
            self.random_net_set(net_state.get_kwargs().random)
            lego_init_cont = []
            for child in internal_nets:
                net_m_buf = child.get_savable_data().ann_container
                if isinstance(net_m_buf, SimpleNetCon):
                    lego_init_cont.append(net_m_buf.lego_init_cont)
            self.savable_set(
                NetSavable[
                    nn.Module, CurrentNetData,
                    Optional[Tuple[NodeANNDataElemInterface, ...]], InitContainer
                ](
                    current_net,
                    NetSavableArgs(
                        node_id=ENetInterfaceNames.COMPLEXDEMONET.value,
                        node_type=type(self),
                        ann_type=type(current_net),
                        to_save=net_state.get_kwargs().save
                    ), BComplexDemoInitCon(
                        children_ids=tuple(
                            child.get_savable_data().node_id for child in internal_nets
                        ),
                        lego_init_con=tuple(lego_init_cont)
                    )
                )
            )
            return True, current_net
        raise KnownComplexDemoError("This should never happen, no module could be created!")

    def check_init_state(self) -> InitGeneralState:
        state_args = self.arguments_con.initializer_wr.init_state
        if not isinstance(state_args, InitGeneralState):
            raise KnownComplexDemoError(
                f"Init expected state type {InitGeneralState.__name__}"
                + f" got {self.arguments_con.initializer_wr.init_state_type}"
            )
        return state_args

    def check_net_state(self) -> NetComplexState:
        net_state = self.arguments_con.net_state
        if not isinstance(net_state, NetComplexState):
            raise KnownComplexDemoError(
                f"Init expected net state type {NetComplexState.__name__}"
                + f" got {type(net_state)}"
            )
        return net_state

    def prepare_data(self, sync_out: SyncStdoutInterface, /) -> None:
        state_args = self.check_init_state()
        net_state = self.check_net_state()
        print_init_net_states(net_state, state_args, sync_out)
        # re-init module container
        module = self.get_savable_net()
        if isinstance(module, ComplexDemoCon) and module.children_nets is not None:
            lego_init_cont = []
            input_data_con = self.get_savable_data().input_data
            if not isinstance(input_data_con, BComplexDemoInitCon):
                raise KnownComplexDemoError(
                    f"Expected {BComplexDemoInitCon.__name__} got {type(input_data_con).__name__}"
                )
            for child in module.children_nets:
                net_m_buf = child.get_savable_data().ann_container
                if isinstance(net_m_buf, SimpleNetCon):
                    lego_init_cont.append(net_m_buf.lego_init_cont)
                else:
                    raise KnownComplexDemoError(
                        f"Expected {SimpleNetCon.__name__} got {type(net_m_buf).__name__}"
                    )
            input_data_con.lego_init_con = tuple(lego_init_cont)
        # init data
        new_stderr = RedirectWriteToLoggerPercent(sync_out, sys.stderr)
        try:
            sys.stderr = new_stderr
            self._init_data_load(state_args)
        except Exception as ex:
            raise ex
        finally:
            sys.stderr = sys.__stderr__
            new_stderr.close()

    def init_net(self, internal_nets: Tuple[NodeANNDataElemInterface, ...],
                 sync_out: SyncStdoutInterface, /) -> bool:
        if len(internal_nets) != self.__connection_cnt:
            raise KnownComplexDemoError(
                f"This node expected to have {self.__connection_cnt} children "
                + f"got {len(internal_nets)}!"
            )
        state_args = self.check_init_state()
        if state_args.get_kwargs().subset_size is not None:
            logger_print_to_console(sync_out, "subset size are not supported, setting to None")
            state_args.disable_subset()
        net_state = self.check_net_state()
        # create net
        tt_list, created_net = self._create_net(state_args, net_state, internal_nets)
        # wrapper init
        init_update = SimpleDemoUpdater(created_net, InitGeneralState)
        init_wrapper(created_net, init_update, self.arguments_con)
        return tt_list

    def train_net(self, id_file: ANNTreeIdType, sync_out: SyncStdoutInterface, /) -> \
            Iterable[TrainNNStatsElementType]:
        # recreate net
        module = self.get_savable_net()
        if isinstance(module, ComplexDemoCon) and module.children_nets is not None:
            module.re_init_current_net(create_current_complex_demo_net(module.children_nets))
        yield from super().train_net(id_file, sync_out)
