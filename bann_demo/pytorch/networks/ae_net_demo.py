# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import sys
from copy import deepcopy
from enum import Enum
from typing import Type, Callable, Dict, Tuple, List, Optional, final, Final

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
from bann.b_frameworks.pytorch.network_nodes.simple_net_abstract import DataSetTypes
from bann.b_data_functions.data_loader_percent_wrapper import RedirectWriteToLoggerPercent
from bann.b_container.functions.print_init_net_state import print_init_net_states
from bann.b_container.functions.pytorch.init_framework_fun import init_wrapper, InitNetArgs
from bann.b_container.states.general.g_init.init_general import InitGeneralState
from bann.b_container.states.general.interface.init_state import NetInitGlobalInterface, InitState
from bann.b_frameworks.pytorch.net_model_interface import NetModelInterface, CurrentNetData, \
    InitContainer
from bann.b_frameworks.pytorch.network_nodes.simple_net_abstract import SimpleNetCon, \
    SimpleAnnNet, get_simple_net_framework
from bann.b_frameworks.pytorch.pytorch_lego_const import LegoContInit
from bann.b_frameworks.pytorch.truth_fun_lib import get_framework_truth_lib

from bann_demo.pytorch.networks.libs.ae_lib import get_ae_list
from bann_demo.pytorch.networks.b_nets.cnn_auto_encoder import CNNAutoEncoderLCI, \
    create_cnn_auto_encoder_net
from bann_demo.pytorch.networks.p_modules.p_conv_auto_encoder import PConvCl
from bann_demo.pytorch.networks.b_nets.auto_encoder import AutoEncoderLCI, create_auto_encoder_net
from bann_demo.pytorch.states.net_ae import NetAEState
from bann_demo.pytorch.networks.errors.custom_erors import KnownAutoEncoderError

from bann_ex_con.pytorch.external_enum import ENetInterfaceNames

from pan.public.functions.load_net_function import net_interface_load_net
from pan.public.interfaces.pub_net_interface import NodeANNDataElemInterface, NetSavable, \
    NetSavableArgs
from pan.public.interfaces.net_connection import NetConnectionWr

from rewowr.public.functions.path_functions import create_dirs_rec
from rewowr.public.functions.syncout_dep_functions import logger_print_to_console
from rewowr.public.interfaces.logger_interface import SyncStdoutInterface


@final
class AELibDemo(Enum):
    AE = "Autoencoder"
    CNNAE = "Conv autoencoder"


def _check_consistency() -> None:
    ae_list = get_ae_list()
    ae_local = [
        ae_t.value
        for ae_t in AELibDemo.__members__.values()
    ]
    if len(ae_list) != len(ae_local):
        raise KnownAutoEncoderError("AE Libraries (local and global) have different length!")
    for ae_e in ae_local:
        if ae_e not in ae_list:
            raise KnownAutoEncoderError(f"Could not find local {ae_e} rbm")


def _create_cnn_ae(layer_cnt: int, input_cnt: int, /) -> CNNAutoEncoderLCI:
    return CNNAutoEncoderLCI(
        net_width=(1, input_cnt, torch.tensor([28, 28])),
        layers_cnt=layer_cnt, kernel_cnt=3, cnn_dim=PConvCl.D2.value
    )


def _create_ae(layer_cnt: int, input_cnt: int, /) -> AutoEncoderLCI:
    return AutoEncoderLCI(input_cnt=input_cnt, layers_cnt=layer_cnt)


def _data_ae(data_ae: List[torch.Tensor], /) -> List[torch.Tensor]:
    return [elem.view(-1).float() for elem in data_ae]


def _data_cnn_ae(data_ae: List[torch.Tensor], /) -> List[torch.Tensor]:
    return [elem.unsqueeze(0).float() for elem in data_ae]


_SWITCH_TYPE: Final[Dict[Type[LegoContInit], Callable[[LegoContInit], CurrentNetData]]] = {
    AutoEncoderLCI: create_auto_encoder_net,
    CNNAutoEncoderLCI: create_cnn_auto_encoder_net
}
_SWITCH_CONT: Final[Dict[str, Callable[[int, int], LegoContInit]]] = {
    AELibDemo.AE.value: _create_ae,
    AELibDemo.CNNAE.value: _create_cnn_ae
}
_SWITCH_INIT_CONT: Final[Dict[str, Callable[[int, int], InitContainer]]] = {
    AELibDemo.AE.value: _create_ae,
    AELibDemo.CNNAE.value: _create_cnn_ae
}
_SWITCH_DATA: Final[Dict[str, Callable[[List[torch.Tensor]], List[torch.Tensor]]]] = {
    AELibDemo.AE.value: _data_ae,
    AELibDemo.CNNAE.value: _data_cnn_ae
}


def _error_lego(*_: int) -> LegoContInit:
    raise KnownAutoEncoderError(f"Could not find auto encoder")


def _error_lego_init(*_: int) -> InitContainer:
    raise KnownAutoEncoderError(f"Could not find auto encoder")


def _error_net_data(*_: LegoContInit) -> CurrentNetData:
    raise KnownAutoEncoderError(f"Could not find auto encoder")


def _error_tensor(*_: List[torch.Tensor]) -> List[torch.Tensor]:
    raise KnownAutoEncoderError(f"Could not find auto encoder")


@final
class AEDemoUpdater(NetInitGlobalInterface):

    def __init__(self, model: NetModelInterface, init_state_type: Type[InitState], /) -> None:
        super().__init__()
        self.__model = model
        self.__init_state_type = init_state_type

    def update_init(self, data_up: InitState, /) -> None:
        if not isinstance(data_up, InitGeneralState):
            raise KnownAutoEncoderError("Received wrong type for init updates!")
        self.__model.hyper_update((*data_up.get_kwargs().drop_rate,
                                   *data_up.get_kwargs().net_sizes))

    def update_init_type(self) -> Type[InitState]:
        return self.__init_state_type


@final
class AEDemoCon(SimpleNetCon[Optional[LegoContInit]]):

    def __init__(self, input_data: LegoContInit, /) -> None:
        self.__input_data: Optional[LegoContInit] = input_data
        super().__init__()

    @property
    def input_data(self) -> Optional[LegoContInit]:
        return self.__input_data

    @input_data.setter
    def input_data(self, data_cont: LegoContInit, /) -> None:
        self.__input_data = data_cont

    def reload_after_save(self, data: Optional[LegoContInit], /) -> None:
        self.__input_data = data

    def remove_before_save(self) -> Optional[LegoContInit]:
        buf = self.__input_data
        self.__input_data = None
        return buf

    @property
    def lego_init_cont(self) -> LegoContInit:
        raise KnownAutoEncoderError(f"{type(self).__name__} is not a lego net!")

    def _create_current_net(self) -> CurrentNetData:
        if self.__input_data is None:
            raise KnownAutoEncoderError("Input data was not set!")
        return _SWITCH_TYPE.get(type(self.__input_data), _error_net_data)(self.__input_data)

    def _create_current_loaded_net(self, extra_args: InitContainer, /) -> CurrentNetData:
        if not isinstance(extra_args, (CNNAutoEncoderLCI, AutoEncoderLCI)):
            raise KnownAutoEncoderError(
                f"Expected args type {AutoEncoderLCI.__name__} or {CNNAutoEncoderLCI.__name__}"
                + "got {type(extra_args).__name__}"
            )
        self.input_data = deepcopy(extra_args)
        return self._create_current_net()

    def hyper_update(self, data_up: Tuple[float, ...], /) -> None:
        if data_up:
            raise KnownAutoEncoderError(f"Expected empty tuple got {len(data_up)} elements.")


@final
class AEDemoNet(SimpleAnnNet[Optional[LegoContInit]]):

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
            raise KnownAutoEncoderError("Could not find the needed truth function!")
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
            raise KnownAutoEncoderError("Path for training data is missing!")

        if not (in_e_path is not None and in_e_path.is_absolute()):
            raise KnownAutoEncoderError("Path for evaluating data is missing!")

        if not (in_te_path is not None and in_te_path.is_absolute()):
            raise KnownAutoEncoderError("Path for testing data is missing!")

    def _init_data_load(self, state_args: InitGeneralState, /) -> None:
        auto_encoder = self.check_net_state().get_kwargs().auto_encoder
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        in_tr_path = state_args.get_kwargs().input_train
        in_e_path = state_args.get_kwargs().input_eval
        if in_tr_path is not None and in_e_path is not None:
            create_dirs_rec(in_tr_path)
            self.__train = (AEMNISTDataSet(datasets.MNIST(
                root=str(in_tr_path),
                train=True, download=True, transform=transform
            ), auto_encoder),)
            self.train_data_set(create_subsets(1000, self.__train))
            create_dirs_rec(in_e_path)
            self.eval_data_set((AEMNISTDataSet(create_subsets(1000, (datasets.MNIST(
                root=str(in_e_path),
                train=False, download=True, transform=transform
            ),))[0], auto_encoder),))
        in_te_path = state_args.get_kwargs().input_test
        if in_te_path is not None:
            create_dirs_rec(in_te_path)
            self.test_data_set((AEMNISTDataSet(create_subsets(1000, (datasets.MNIST(
                root=str(in_te_path),
                train=False, download=True, transform=transform
            ),))[0], auto_encoder),))

    def _create_net(self, state_args: InitGeneralState,
                    net_state: NetAEState, /) -> NetModelInterface:
        net_load = state_args.get_kwargs().input_net
        if net_load is not None:
            if net_load.is_absolute() and net_load.exists() and net_load.is_file():
                self.retrain_set(net_state.get_kwargs().retrain)
                with net_load.open('rb') as net_handler:
                    buffer_load = net_interface_load_net(
                        type(self), net_handler.read()
                    )
                    if not isinstance(buffer_load, NetSavable):
                        raise KnownAutoEncoderError(
                            f"Expected net {NetSavable.__name__} "
                            + f" got {type(buffer_load).__name__}!"
                        )
                    current_net = buffer_load.ann_container
                    if not isinstance(current_net, AEDemoCon):
                        raise KnownAutoEncoderError(
                            f"Expected net {AEDemoCon.__name__} "
                            + f" got {type(current_net).__name__}!"
                        )
                    self.net_module_set(current_net)
                    self.savable_set(buffer_load)
                    return current_net
        if self.net_module is None:
            current_net = AEDemoCon(
                _SWITCH_CONT.get(net_state.get_kwargs().auto_encoder, _error_lego)(
                    net_state.get_kwargs().layer_cnt, net_state.get_kwargs().input_cnt
                )
            )
            self.net_module_set(current_net)
            self.random_net_set(net_state.get_kwargs().random)
            self.savable_set(
                NetSavable[nn.Module, CurrentNetData, Optional[LegoContInit], InitContainer](
                    current_net,
                    NetSavableArgs(
                        node_id=ENetInterfaceNames.AEDEMONET.value,
                        node_type=type(self),
                        ann_type=type(current_net),
                        to_save=net_state.get_kwargs().save
                    ), _SWITCH_INIT_CONT.get(net_state.get_kwargs().auto_encoder, _error_lego_init)(
                        net_state.get_kwargs().layer_cnt, net_state.get_kwargs().input_cnt
                    )
                )
            )
            return current_net
        raise KnownAutoEncoderError("This should never happen!")

    def check_init_state(self) -> InitGeneralState:
        state_args = self.arguments_con.initializer_wr.init_state
        if not isinstance(state_args, InitGeneralState):
            raise KnownAutoEncoderError(
                f"Init expected state type {InitGeneralState.__name__}"
                + f" got {self.arguments_con.initializer_wr.init_state_type}"
            )
        return state_args

    def check_net_state(self) -> NetAEState:
        net_state = self.arguments_con.net_state
        if not isinstance(net_state, NetAEState):
            raise KnownAutoEncoderError(
                f"Init expected net state type {NetAEState.__name__}"
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
            raise KnownAutoEncoderError("This node didn't expect to have children!")
        state_args = self.check_init_state()
        if state_args.get_kwargs().subset_size is not None:
            logger_print_to_console(sync_out, "subset size are not supported, setting to None")
            state_args.disable_subset()
        net_state = self.check_net_state()
        # create net
        created_net = self._create_net(state_args, net_state)
        # wrapper init
        init_update = AEDemoUpdater(created_net, InitGeneralState)
        init_wrapper(created_net, init_update, self.arguments_con)
        return True


@final
class AEMNISTDataSet(DataSetSharedMemoryA[Tuple[torch.Tensor, torch.Tensor]]):

    def __init__(self, data_torch: Dataset, auto_encoder: str, /) -> None:
        buffer = [
            data_t for l_da in range(len(data_torch)) for data_t in data_torch[l_da][0]
        ]
        self.__data_list: TypeShapeCon = TypeShapeCon(
            data=np.array(list(ne_v.tolist() for ne_v in _SWITCH_DATA.get(
                auto_encoder, _error_tensor
            )(buffer)))
        )
        self.__copy_pre_trim: Optional[TypeShapeCon] = self.__data_list
        super().__init__(len(data_torch))

    def remap_shared_memory(self) -> None:
        remap_shared_mem(self.__data_list, self.subset)

    def _move_data_to_shared_memory(self) -> None:
        used_smm = self.used_smm
        if not (used_smm is None or self.__copy_pre_trim is None
                or self.__copy_pre_trim.data is None):
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
