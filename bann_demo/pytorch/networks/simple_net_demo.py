# -*- coding: utf-8 -*-
"""inspired by:
  - https://github.com/pytorch/examples/blob/master/mnist/main.py

LICENSE (BSD 3-Clause License): see extra_licenses/LICENSE_P_EXAMPLES

.. moduleauthor:: Artur Lissin
"""
import sys

from typing import Tuple, Type, Optional, final

from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms, datasets  # type: ignore

from bann.b_frameworks.pytorch.p_truth.p_truth_enum import PTruthId
from bann.b_data_functions.pytorch.subset_dataset import create_subsets
from bann.b_frameworks.pytorch.pytorch_lego_const import LegoContInit
from bann.b_frameworks.pytorch.network_nodes.simple_net_abstract import SimpleAnnNet, \
    SimpleNetCon, get_simple_net_framework, DataSetTypes
from bann.b_frameworks.pytorch.net_model_interface import InitContainer
from bann.b_container.functions.print_init_net_state import print_init_net_states
from bann.b_data_functions.data_loader_percent_wrapper import RedirectWriteToLoggerPercent
from bann.b_container.states.general.net.net_general import NetGeneralState
from bann.b_container.states.general.g_init.init_general import InitGeneralState
from bann.b_container.states.general.interface.init_state import InitState
from bann.b_container.states.general.interface.init_state import NetInitGlobalInterface
from bann.b_container.functions.pytorch.init_framework_fun import init_wrapper, InitNetArgs
from bann.b_frameworks.pytorch.truth_fun_lib import get_framework_truth_lib
from bann.b_frameworks.pytorch.net_model_interface import NetModelInterface
from bann.b_frameworks.pytorch.net_model_interface import CurrentNetData

from bann_demo.pytorch.networks.b_nets.simple_demo import create_current_demo_net
from bann_demo.pytorch.connections.demo_connection import get_demo_connection
from bann_demo.pytorch.networks.errors.custom_erors import KnownSimpleDemoError

from bann_ex_con.pytorch.external_enum import ENetInterfaceNames

from pan.public.functions.load_net_function import net_interface_load_net
from pan.public.interfaces.net_connection import NetConnectionWr
from pan.public.interfaces.pub_net_interface import NodeANNDataElemInterface, NetSavable, \
    NetSavableArgs

from rewowr.public.functions.path_functions import create_dirs_rec
from rewowr.public.functions.syncout_dep_functions import logger_print_to_console
from rewowr.public.interfaces.logger_interface import SyncStdoutInterface


@final
class SimpleDemoUpdater(NetInitGlobalInterface):

    def __init__(self, model: NetModelInterface, init_state_type: Type[InitState], /) -> None:
        super().__init__()
        self.__model = model
        self.__init_state_type = init_state_type

    def update_init(self, data_up: InitState, /) -> None:
        if not isinstance(data_up, InitGeneralState):
            raise KnownSimpleDemoError("Received wrong type for init updates!")
        self.__model.hyper_update((*data_up.get_kwargs().drop_rate,
                                   *data_up.get_kwargs().net_sizes))

    def update_init_type(self) -> Type[InitState]:
        return self.__init_state_type


@final
class SimpleDemoCon(SimpleNetCon[None]):

    def remove_before_save(self) -> None:
        return None

    def reload_after_save(self, data: None, /) -> None:
        pass

    @property
    def lego_init_cont(self) -> LegoContInit:
        return LegoContInit()

    def _create_current_net(self) -> CurrentNetData:
        return create_current_demo_net(LegoContInit())

    def _create_current_loaded_net(self, extra_args: InitContainer, /) -> CurrentNetData:
        return self._create_current_net()

    def hyper_update(self, data_up: Tuple[float, ...], /) -> None:
        if data_up:
            raise KnownSimpleDemoError(f"Expected empty tuple got {len(data_up)} elements.")


class SimpleDemoNet(SimpleAnnNet[None]):

    def __init__(self, args: InitNetArgs, /) -> None:
        self.__train: Optional[Tuple[Dataset, ...]] = None
        super().__init__(args)

    @final
    def re_read_data(self, data_type: DataSetTypes, /) -> Optional[Tuple[Dataset, ...]]:
        if not self.check_net_state().get_kwargs().resample:
            return None
        if data_type == DataSetTypes.TRAIN and self.__train is not None:
            return create_subsets(1000, self.__train)
        return None

    @final
    def get_truth_fun_id(self) -> str:
        truth_id = PTruthId.ONECLASS.value
        if not get_framework_truth_lib(
                get_simple_net_framework()
        ).truth_fun_check(truth_id):
            raise KnownSimpleDemoError("Could not find the needed truth function!")
        return truth_id

    @property
    def connection_in(self) -> NetConnectionWr:
        return get_demo_connection(get_simple_net_framework())

    @property
    def connection_out(self) -> Tuple[NetConnectionWr, ...]:
        return tuple()

    @final
    def check_net(self, internal_nets: Tuple[NodeANNDataElemInterface, ...],
                  sync_out: SyncStdoutInterface, /) -> None:
        _ = self.init_net(internal_nets, sync_out)
        state_args = self.check_init_state()
        in_tr_path = state_args.get_kwargs().input_train
        in_e_path = state_args.get_kwargs().input_eval
        in_te_path = state_args.get_kwargs().input_test
        if not (in_tr_path is not None and in_tr_path.is_absolute()):
            raise KnownSimpleDemoError("Path for training data is missing!")

        if not (in_e_path is not None and in_e_path.is_absolute()):
            raise KnownSimpleDemoError("Path for evaluating data is missing!")

        if not (in_te_path is not None and in_te_path.is_absolute()):
            raise KnownSimpleDemoError("Path for testing data is missing!")

    @final
    def _init_data_load(self, state_args: InitGeneralState, /) -> None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        in_tr_path = state_args.get_kwargs().input_train
        in_e_path = state_args.get_kwargs().input_eval
        if in_tr_path is not None and in_e_path is not None:
            create_dirs_rec(in_tr_path)
            self.__train = (datasets.MNIST(
                root=str(in_tr_path),
                train=True, download=True, transform=transform
            ),)
            self.train_data_set(create_subsets(1000, self.__train))
            create_dirs_rec(in_e_path)
            self.eval_data_set(create_subsets(1000, (datasets.MNIST(
                root=str(in_e_path),
                train=False, download=True, transform=transform
            ),)))
        in_te_path = state_args.get_kwargs().input_test
        if in_te_path is not None:
            create_dirs_rec(in_te_path)
            self.test_data_set(create_subsets(1000, (datasets.MNIST(
                root=str(in_te_path),
                train=False, download=True, transform=transform
            ),)))

    # only not final because of simple_net_demo_pre_training
    def _create_net(self, state_args: InitGeneralState,
                    net_state: NetGeneralState, /) -> NetModelInterface:
        net_load = state_args.get_kwargs().input_net
        if net_load is not None:
            if net_load.is_absolute() and net_load.exists() and net_load.is_file():
                self.retrain_set(net_state.get_kwargs().retrain)
                with net_load.open('rb') as net_handler:
                    buffer_load = net_interface_load_net(
                        type(self), net_handler.read()
                    )
                    if not isinstance(buffer_load, NetSavable):
                        raise KnownSimpleDemoError(
                            f"Expected net {NetSavable.__name__} "
                            + f" got {type(buffer_load).__name__}!"
                        )
                    current_net = buffer_load.ann_container
                    if not isinstance(current_net, SimpleDemoCon):
                        raise KnownSimpleDemoError(
                            f"Expected net {SimpleDemoCon.__name__} "
                            + f" got {type(current_net).__name__}!"
                        )
                    self.net_module_set(current_net)
                    self.savable_set(buffer_load)
                    return current_net
        if self.net_module is None:
            current_net = SimpleDemoCon()
            self.net_module_set(current_net)
            self.random_net_set(net_state.get_kwargs().random)

            self.savable_set(NetSavable[nn.Module, CurrentNetData, None, InitContainer](
                current_net,
                NetSavableArgs(
                    node_id=ENetInterfaceNames.SIMPLEDEMONET.value,
                    node_type=type(self),
                    ann_type=type(current_net),
                    to_save=net_state.get_kwargs().save
                ), InitContainer()
            ))
            return current_net
        raise KnownSimpleDemoError("This should never happen!")

    @final
    def check_init_state(self) -> InitGeneralState:
        state_args = self.arguments_con.initializer_wr.init_state
        if not isinstance(state_args, InitGeneralState):
            raise KnownSimpleDemoError(
                f"Init expected state type {InitGeneralState.__name__}"
                + f" got {self.arguments_con.initializer_wr.init_state_type}"
            )
        return state_args

    @final
    def check_net_state(self) -> NetGeneralState:
        net_state = self.arguments_con.net_state
        if not isinstance(net_state, NetGeneralState):
            raise KnownSimpleDemoError(
                f"Init expected net state type {NetGeneralState.__name__}"
                + f" got {type(net_state)}"
            )
        return net_state

    @final
    def prepare_data(self, sync_out: SyncStdoutInterface, /) -> None:
        state_args = self.check_init_state()
        net_state = self.check_net_state()
        print_init_net_states(net_state, state_args, sync_out)
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

    @final
    def init_net(self, internal_nets: Tuple[NodeANNDataElemInterface, ...],
                 sync_out: SyncStdoutInterface, /) -> bool:
        if internal_nets:
            raise KnownSimpleDemoError("This node didn't expect to have children!")
        state_args = self.check_init_state()
        if state_args.get_kwargs().subset_size is not None:
            logger_print_to_console(sync_out, "subset size are not supported, setting to None")
            state_args.disable_subset()
        net_state = self.check_net_state()
        # create net
        created_net = self._create_net(state_args, net_state)
        # wrapper init
        init_update = SimpleDemoUpdater(created_net, InitGeneralState)
        init_wrapper(created_net, init_update, self.arguments_con)
        return True
