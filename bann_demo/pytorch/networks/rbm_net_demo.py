# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import sys
from enum import Enum
from typing import Type, Tuple, Dict, Callable, List, Optional, final, Final
import numpy as np  # type: ignore
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms, datasets  # type: ignore

from bann.b_frameworks.pytorch.p_truth.p_truth_enum import PTruthId
from bann.b_data_functions.pytorch.shared_memory_interface import DataSetSharedMemoryA, \
    generate_shared_mem, TypeShapeCon, trim_shallow_copy, data_get_item, remap_shared_mem, \
    data_shallow_copy_shared_mem
from bann.b_data_functions.pytorch.subset_dataset import create_subsets
from bann.b_frameworks.pytorch.pytorch_lego_const import LegoContInit
from bann.b_container.functions.print_init_net_state import print_init_net_states
from bann.b_container.functions.pytorch.init_framework_fun import init_wrapper, InitNetArgs
from bann.b_data_functions.data_loader_percent_wrapper import RedirectWriteToLoggerPercent
from bann.b_frameworks.pytorch.truth_fun_lib import get_framework_truth_lib
from bann.b_container.states.general.g_init.init_general import InitGeneralState
from bann.b_container.states.general.interface.init_state import NetInitGlobalInterface, InitState
from bann.b_frameworks.pytorch.net_model_interface import NetModelInterface, CurrentNetData, \
    InitContainer
from bann.b_frameworks.pytorch.network_nodes.simple_net_abstract import SimpleNetCon, \
    SimpleAnnNet, get_simple_net_framework, DataSetTypes

from bann_demo.pytorch.networks.b_nets.rbm_con import create_rbm_net, RBMLCI
from bann_demo.pytorch.states.net_rbm import NetRBMState
from bann_demo.pytorch.networks.libs.rbm_lib import get_rbm_list
from bann_demo.pytorch.networks.errors.custom_erors import KnownRBMError

from bann_ex_con.pytorch.external_enum import ENetInterfaceNames

from pan.public.functions.load_net_function import net_interface_load_net
from pan.public.interfaces.pub_net_interface import NodeANNDataElemInterface, NetSavable, \
    NetSavableArgs
from pan.public.interfaces.net_connection import NetConnectionWr

from rewowr.public.functions.path_functions import create_dirs_rec
from rewowr.public.functions.syncout_dep_functions import logger_print_to_console
from rewowr.public.interfaces.logger_interface import SyncStdoutInterface


@final
class RBMDemoUpdater(NetInitGlobalInterface):

    def __init__(self, model: NetModelInterface, init_state_type: Type[InitState], /) -> None:
        super().__init__()
        self.__model = model
        self.__init_state_type = init_state_type

    def update_init(self, data_up: InitState, /) -> None:
        if not isinstance(data_up, InitGeneralState):
            raise KnownRBMError("Received wrong type for init updates!")
        self.__model.hyper_update((*data_up.get_kwargs().drop_rate,
                                   *data_up.get_kwargs().net_sizes))

    def update_init_type(self) -> Type[InitState]:
        return self.__init_state_type


@final
class RBMDemoCon(SimpleNetCon[Tuple[Optional[int], Optional[str], Optional[Tuple[int, int]]]]):

    def __init__(self, in_f: int, out_f: int, g_sampling: int, rbm: str, /) -> None:
        self.__g_sampling: Optional[int] = g_sampling
        self.__rbm: Optional[str] = rbm
        self.__in_out: Optional[Tuple[int, int]] = (in_f, out_f)
        super().__init__()

    @property
    def in_out(self) -> Optional[Tuple[int, int]]:
        return self.__in_out

    @in_out.setter
    def in_out(self, in_out: Tuple[int, int], /) -> None:
        self.__in_out = in_out

    @property
    def rbm(self) -> Optional[str]:
        return self.__rbm

    @rbm.setter
    def rbm(self, rbm: str, /) -> None:
        self.__rbm = rbm

    @property
    def g_sampling(self) -> Optional[int]:
        return self.__g_sampling

    @g_sampling.setter
    def g_sampling(self, g_sam: int, /) -> None:
        self.__g_sampling = g_sam

    def remove_before_save(self) -> Tuple[Optional[int], Optional[str], Optional[Tuple[int, int]]]:
        buf = (self.__g_sampling, self.__rbm, self.__in_out)
        self.__g_sampling = None
        self.__rbm = None
        self.__in_out = None
        return buf

    def reload_after_save(self, data: Tuple[
        Optional[int], Optional[str], Optional[Tuple[int, int]]
    ], /) -> None:
        self.__g_sampling, self.__rbm, self.__in_out = data

    @property
    def lego_init_cont(self) -> LegoContInit:
        raise KnownRBMError(f"{type(self).__name__} is not a lego net!")

    def _create_current_net(self) -> CurrentNetData:
        if self.__g_sampling is None or self.__in_out is None or self.__rbm is None:
            raise KnownRBMError("Input data was not set!")
        return create_rbm_net(RBMLCI(
            in_features=self.__in_out[0], out_features=self.__in_out[1],
            gibbs_sampling=self.__g_sampling, rbm_type=self.__rbm
        ))

    def _create_current_loaded_net(self, extra_args: InitContainer, /) -> CurrentNetData:
        if not isinstance(extra_args, RBMLCI):
            raise KnownRBMError(
                f"Expected args type {RBMLCI.__name__} got {type(extra_args).__name__}"
            )
        self.g_sampling = extra_args.gibbs_sampling
        self.in_out = (extra_args.in_features, extra_args.out_features)
        self.rbm = extra_args.rbm_type
        return self._create_current_net()

    def hyper_update(self, data_up: Tuple[float, ...], /) -> None:
        if data_up:
            raise KnownRBMError(f"Expected empty tuple got {len(data_up)} elements.")


@final
class RBMDemoNet(SimpleAnnNet[Tuple[Optional[int], Optional[str], Optional[Tuple[int, int]]]]):

    def __init__(self, args: InitNetArgs, /) -> None:
        self.__train: Optional[Tuple[Dataset, ...]] = None
        super().__init__(args)

    def re_read_data(self, data_type: DataSetTypes, /) -> Optional[Tuple[Dataset, ...]]:
        if not self.check_net_state().get_kwargs().resample:
            return None
        if data_type == DataSetTypes.TRAIN and self.__train is not None:
            return create_subsets(1000, self.__train)
        return None

    def get_truth_fun_id(self) -> str:
        truth_id = PTruthId.MSIM.value
        if not get_framework_truth_lib(
                get_simple_net_framework()
        ).truth_fun_check(truth_id):
            raise KnownRBMError("Could not find the needed truth function!")
        return truth_id

    @property
    def connection_in(self) -> NetConnectionWr:
        return NetConnectionWr("", "")

    @property
    def connection_out(self) -> Tuple[NetConnectionWr, ...]:
        return tuple()

    def check_net(self, internal_nets: Tuple[NodeANNDataElemInterface, ...],
                  sync_out: SyncStdoutInterface, /) -> None:
        _check_consistency()
        _ = self.init_net(internal_nets, sync_out)
        state_args = self.check_init_state()
        in_tr_path = state_args.get_kwargs().input_train
        in_e_path = state_args.get_kwargs().input_eval
        in_te_path = state_args.get_kwargs().input_test
        if not (in_tr_path is not None and in_tr_path.is_absolute()):
            raise KnownRBMError("Path for training data is missing!")

        if not (in_e_path is not None and in_e_path.is_absolute()):
            raise KnownRBMError("Path for evaluating data is missing!")

        if not (in_te_path is not None and in_te_path.is_absolute()):
            raise KnownRBMError("Path for testing data is missing!")

    def _init_data_load(self, state_args: InitGeneralState, /) -> None:
        rbm = self.check_net_state().get_kwargs().rbm
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        in_tr_path = state_args.get_kwargs().input_train
        in_e_path = state_args.get_kwargs().input_eval
        if in_tr_path is not None and in_e_path is not None:
            create_dirs_rec(in_tr_path)
            self.__train = (RBMMNISTDataSet(datasets.MNIST(
                root=str(in_tr_path),
                train=True, download=True, transform=transform
            ), rbm),)
            self.train_data_set(create_subsets(1000, self.__train))
            create_dirs_rec(in_e_path)
            self.eval_data_set((RBMMNISTDataSet(create_subsets(1000, (datasets.MNIST(
                root=str(in_e_path),
                train=False, download=True, transform=transform
            ),))[0], rbm),))
        in_te_path = state_args.get_kwargs().input_test
        if in_te_path is not None:
            create_dirs_rec(in_te_path)
            self.test_data_set((RBMMNISTDataSet(create_subsets(1000, (datasets.MNIST(
                root=str(in_te_path),
                train=False, download=True, transform=transform
            ),))[0], rbm),))

    def _create_net(self, state_args: InitGeneralState,
                    net_state: NetRBMState, /) -> NetModelInterface:
        net_load = state_args.get_kwargs().input_net
        if net_load is not None:
            if net_load.is_absolute() and net_load.exists() and net_load.is_file():
                self.retrain_set(net_state.get_kwargs().retrain)
                with net_load.open('rb') as net_handler:
                    buffer_load = net_interface_load_net(
                        type(self), net_handler.read()
                    )
                    if not isinstance(buffer_load, NetSavable):
                        raise KnownRBMError(
                            f"Expected net {NetSavable.__name__} "
                            + f" got {type(buffer_load).__name__}!"
                        )
                    current_net = buffer_load.ann_container
                    if not isinstance(current_net, RBMDemoCon):
                        raise KnownRBMError(
                            f"Expected net {RBMDemoCon.__name__} "
                            + f" got {type(current_net).__name__}!"
                        )
                    self.net_module_set(current_net)
                    self.savable_set(buffer_load)
                    return current_net
        if self.net_module is None:
            current_net = RBMDemoCon(
                28 * 28, net_state.get_kwargs().hidden_cnt,
                net_state.get_kwargs().g_sampling, net_state.get_kwargs().rbm,
            )
            self.net_module_set(current_net)
            self.random_net_set(net_state.get_kwargs().random)

            self.savable_set(NetSavable[
                nn.Module, CurrentNetData,
                Tuple[Optional[int], Optional[str], Optional[Tuple[int, int]]], InitContainer
            ](
                current_net,
                NetSavableArgs(
                    node_id=ENetInterfaceNames.RBMDEMONET.value,
                    node_type=type(self),
                    ann_type=type(current_net),
                    to_save=net_state.get_kwargs().save
                ), RBMLCI(
                    in_features=28 * 28,
                    out_features=net_state.get_kwargs().hidden_cnt,
                    gibbs_sampling=net_state.get_kwargs().g_sampling,
                    rbm_type=net_state.get_kwargs().rbm
                )
            ))
            return current_net
        raise KnownRBMError("This should never happen!")

    def check_init_state(self) -> InitGeneralState:
        state_args = self.arguments_con.initializer_wr.init_state
        if not isinstance(state_args, InitGeneralState):
            raise KnownRBMError(
                f"Init expected state type {InitGeneralState.__name__}"
                + f" got {self.arguments_con.initializer_wr.init_state_type}"
            )
        return state_args

    def check_net_state(self) -> NetRBMState:
        net_state = self.arguments_con.net_state
        if not isinstance(net_state, NetRBMState):
            raise KnownRBMError(
                f"Init expected net state type {NetRBMState.__name__}"
                + f" got {type(net_state)}"
            )
        return net_state

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

    def init_net(self, internal_nets: Tuple[NodeANNDataElemInterface, ...],
                 sync_out: SyncStdoutInterface, /) -> bool:
        if internal_nets:
            raise KnownRBMError("This node didn't expect to have children!")
        state_args = self.check_init_state()
        if state_args.get_kwargs().subset_size is not None:
            logger_print_to_console(sync_out, "subset size are not supported, setting to None")
            state_args.disable_subset()
        net_state = self.check_net_state()
        # create net
        created_net = self._create_net(state_args, net_state)
        # wrapper init
        init_update = RBMDemoUpdater(created_net, InitGeneralState)
        init_wrapper(created_net, init_update, self.arguments_con)
        return True


@final
class RBMLibDemo(Enum):
    BBRBM = "BinaryBinaryRBM"
    GBRBM = "GaussianBinaryRBM"


def _check_consistency() -> None:
    rbm_list = get_rbm_list()
    rbm_local = [
        rbm_t.value
        for rbm_t in RBMLibDemo.__members__.values()
    ]
    if len(rbm_local) != len(rbm_list):
        raise KnownRBMError("RBM Libraries (local and global) have different length!")
    for rbm_e in rbm_local:
        if rbm_e not in rbm_list:
            raise KnownRBMError(f"Could not find local {rbm_e} rbm")


def _min_max_norm(data_torch: List[torch.Tensor], /) -> List[torch.Tensor]:
    data_to_torch = torch.cat([elem.unsqueeze(0) for elem in data_torch])
    data_to_torch = \
        (data_to_torch - data_to_torch.min()) / (data_to_torch.max() - data_to_torch.min())
    return [data_to_torch[n_index] for n_index in range(data_to_torch.size(0))]


def _min_max_norm_bin(data_torch: List[torch.Tensor], /) -> List[torch.Tensor]:
    normed_list = _min_max_norm(data_torch)
    data_to_torch = torch.cat([elem.unsqueeze(0) for elem in normed_list])
    data_to_torch = data_to_torch.ge(data_to_torch.mean()).float()
    return [data_to_torch[n_index] for n_index in range(data_to_torch.size(0))]


_SWITCH: Final[Dict[str, Callable[[List[torch.Tensor]], List[torch.Tensor]]]] = {
    RBMLibDemo.GBRBM.value: _min_max_norm,
    RBMLibDemo.BBRBM.value: _min_max_norm_bin
}


def _error(_: List[torch.Tensor], /) -> List[torch.Tensor]:
    raise KnownRBMError(f"Could not find rbm")


@final
class RBMMNISTDataSet(DataSetSharedMemoryA[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, data_torch: Dataset, rbm: str, /) -> None:
        buffer = [
            data_t.view(-1) for l_da in range(len(data_torch)) for data_t in data_torch[l_da][0]
        ]
        self.__data_list: TypeShapeCon = TypeShapeCon(
            data=np.array(list(ne_v.tolist() for ne_v in _SWITCH.get(rbm, _error)(buffer)))
        )
        self.__copy_pre_trim: Optional[TypeShapeCon] = self.__data_list
        super().__init__(len(data_torch))

    def remap_shared_memory(self) -> None:
        remap_shared_mem(self.__data_list, self.subset)

    def _move_data_to_shared_memory(self) -> None:
        used_smm = self.used_smm
        if not (
                used_smm is None
                or self.__copy_pre_trim is None
                or self.__copy_pre_trim.data is None
        ):
            generate_shared_mem(self.__copy_pre_trim, used_smm)
            self.__data_list = data_shallow_copy_shared_mem(self.__copy_pre_trim)

    def _pre_send_empty(self) -> None:
        self.__copy_pre_trim = None

    def _trim_shallow_copy(self, indices: List[int], /) -> None:
        self.__copy_pre_trim = self.__data_list
        self.__data_list = trim_shallow_copy(self.__copy_pre_trim, indices)

    def _getitem(self, item: int, /) -> Tuple[torch.Tensor, torch.Tensor]:
        data_point = data_get_item(self.__data_list, item)
        data_tensor = torch.tensor(data_point).float()
        class_l = torch.tensor(data_point).float()
        return data_tensor, class_l
