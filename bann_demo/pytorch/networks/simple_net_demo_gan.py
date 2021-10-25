# -*- coding: utf-8 -*-
"""inspired by:
  - https://github.com/pytorch/examples/blob/master/mnist/main.py

LICENSE (BSD 3-Clause License): see extra_licenses/LICENSE_P_EXAMPLES

.. moduleauthor:: Artur Lissin
"""
from typing import final, Tuple

from torch import nn

from bann.b_container.states.general.g_init.init_general import InitGeneralState
from bann.b_container.states.general.net.net_general import NetGeneralState
from bann.b_frameworks.pytorch.net_model_interface import InitContainer, CurrentNetData, \
    NetModelInterface
from bann.b_frameworks.pytorch.network_nodes.simple_net_abstract import SimpleNetCon
from bann.b_frameworks.pytorch.pytorch_lego_const import LegoContInit

from bann_demo.pytorch.networks.b_nets.simple_demo import create_gan_demo_net
from bann_demo.pytorch.networks.errors.custom_erors import KnownSimpleDemoError
from bann_demo.pytorch.networks.simple_net_demo import SimpleDemoNet

from bann_ex_con.pytorch.external_enum import ENetInterfaceNames

from pan.public.functions.load_net_function import net_interface_load_net
from pan.public.interfaces.net_connection import NetConnectionWr
from pan.public.interfaces.pub_net_interface import NetSavable, NetSavableArgs


@final
class SimpleDemoGanCon(SimpleNetCon[None]):

    def remove_before_save(self) -> None:
        return None

    def reload_after_save(self, data: None, /) -> None:
        pass

    @property
    def lego_init_cont(self) -> LegoContInit:
        return LegoContInit()

    def _create_current_net(self) -> CurrentNetData:
        return create_gan_demo_net(LegoContInit())

    def _create_current_loaded_net(self, extra_args: InitContainer, /) -> CurrentNetData:
        return self._create_current_net()

    def hyper_update(self, data_up: Tuple[float, ...], /) -> None:
        if data_up:
            raise KnownSimpleDemoError(f"Expected empty tuple got {len(data_up)} elements.")


@final
class SimpleDemoGANNet(SimpleDemoNet):

    @property
    def connection_in(self) -> NetConnectionWr:
        return NetConnectionWr("", "")

    @property
    def connection_out(self) -> Tuple[NetConnectionWr, ...]:
        return tuple()

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
                    if not isinstance(current_net, SimpleDemoGanCon):
                        raise KnownSimpleDemoError(
                            f"Expected net {SimpleDemoGanCon.__name__} "
                            + f" got {type(current_net).__name__}!"
                        )
                    self.net_module_set(current_net)
                    self.savable_set(buffer_load)
                    return current_net
        if self.net_module is None:
            current_net = SimpleDemoGanCon()
            self.net_module_set(current_net)
            self.random_net_set(net_state.get_kwargs().random)

            self.savable_set(NetSavable[nn.Module, CurrentNetData, None, InitContainer](
                current_net,
                NetSavableArgs(
                    node_id=ENetInterfaceNames.SIMPLEDEMOGANNET.value,
                    node_type=type(self),
                    ann_type=type(current_net),
                    to_save=net_state.get_kwargs().save
                ), InitContainer()
            ))
            return current_net
        raise KnownSimpleDemoError("This should never happen!")
