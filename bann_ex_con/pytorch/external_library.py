# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from typing import Tuple, Final

from bann.b_container.states.general.net_param import NetLibName
from bann.b_pan_integration.framwork_key_lib import FrameworkKeyLib

from bann_demo.pytorch.networks.ae_net_demo import AEDemoNet
from bann_demo.pytorch.networks.complex_net_demo_net import \
    ComplexDemoNet
from bann_demo.pytorch.networks.rbm_net_demo import RBMDemoNet
from bann_demo.pytorch.networks.simple_net_demo import SimpleDemoNet
from bann_demo.pytorch.networks.simple_net_demo_pre_training import \
    SimpleDemoPreNet
from bann_demo.pytorch.connections.demo_connection import get_con_demo_name, get_demo_connection
from bann_demo.pytorch.create_node.check_simple_net import CheckSimpleNet
from bann_demo.pytorch.networks.simple_net_demo_gan import SimpleDemoGANNet

from bann_ex_con.external_states_enum import ENetLibName
from bann_ex_con.pytorch.external_enum import ENetInterfaceNames

from pan.public.interfaces.config_constants import NetDictLibraryType
from pan.public.interfaces.net_connection import NetConnectionDict, NetConnectionWr

_FRAMEWORK: Final[str] = FrameworkKeyLib.PYTORCH.value

_LocalConnectionLib: Final[NetConnectionDict] = NetConnectionDict(
    framework=_FRAMEWORK,
    con_dict={
        # TODO fill with a function from a list
        # demo
        NetConnectionWr.parse_dict_id(_FRAMEWORK, get_con_demo_name()):
            get_demo_connection(_FRAMEWORK),
        # e_networks
    }
)
_LocalNetInterfaceLib: Final[NetDictLibraryType] = NetDictLibraryType(
    framework=_FRAMEWORK,
    net_dict={
        # TODO fill with a function from a list
        # demo
        ENetInterfaceNames.SIMPLEDEMONET.value: CheckSimpleNet(
            ENetInterfaceNames.SIMPLEDEMONET.value,
            _FRAMEWORK, SimpleDemoNet, NetLibName.GENERAL.value
        ),
        ENetInterfaceNames.SIMPLEDEMOGANNET.value: CheckSimpleNet(
            ENetInterfaceNames.SIMPLEDEMOGANNET.value,
            _FRAMEWORK, SimpleDemoGANNet, NetLibName.GENERAL.value
        ),
        ENetInterfaceNames.SIMPLEDEMOPRENET.value: CheckSimpleNet(
            ENetInterfaceNames.SIMPLEDEMOPRENET.value,
            _FRAMEWORK, SimpleDemoPreNet, NetLibName.GENERAL.value
        ),
        ENetInterfaceNames.COMPLEXDEMONET.value: CheckSimpleNet(
            ENetInterfaceNames.COMPLEXDEMONET.value,
            _FRAMEWORK, ComplexDemoNet, ENetLibName.COMPLEX.value
        ),
        ENetInterfaceNames.RBMDEMONET.value: CheckSimpleNet(
            ENetInterfaceNames.RBMDEMONET.value,
            _FRAMEWORK, RBMDemoNet, ENetLibName.RBM.value
        ),
        ENetInterfaceNames.AEDEMONET.value: CheckSimpleNet(
            ENetInterfaceNames.AEDEMONET.value,
            _FRAMEWORK, AEDemoNet, ENetLibName.AE.value
        ),
        # e_networks
    }
)


def get_e_pytorch_connections() -> Tuple[str, NetConnectionDict]:
    erg = (_FRAMEWORK, _LocalConnectionLib)
    return erg


def get_e_pytorch_net_interfaces() -> Tuple[str, NetDictLibraryType]:
    erg = (_FRAMEWORK, _LocalNetInterfaceLib)
    return erg
