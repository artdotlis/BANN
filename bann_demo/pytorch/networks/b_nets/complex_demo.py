# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple, Any, List, final

import torch
from torch import nn
from torch.nn import functional as nn_fun

from bann.b_frameworks.pytorch.pytorch_lego_const import LegoContInit
from bann.b_frameworks.pytorch.net_model_interface import InitContainer
from bann.b_frameworks.pytorch.net_model_interface import CurrentNetData

from bann_demo.pytorch.pytorch_lego_library import get_pytorch_net_creators
from bann_demo.pytorch.networks.errors.custom_erors import KnownComplexDemoError

from pan.public.interfaces.pub_net_interface import NodeANNDataElemInterface


@final
@dataclass
class BNetComplexArgs:
    lego: nn.ModuleList
    fc2: nn.Module


@final
class BContainerComplexNet(nn.Module):

    def __init__(self, complete: BNetComplexArgs, /) -> None:
        super().__init__()
        self.lego: nn.ModuleList = complete.lego
        self.fc2: nn.Module = complete.fc2

    def forward(self, *input_args: Any) -> torch.Tensor:
        if not (input_args and isinstance(input_args[0], torch.Tensor)):
            raise KnownComplexDemoError(
                "Input data should not be empty and only contain tensor types"
            )
        input_data = input_args[0]
        lego_l: nn.ModuleList = self.lego
        list_x = [nn_fun.relu(child(input_data.clone())) for child in lego_l.children()]
        new_tensor = torch.cat(list_x, 1)
        input_data = self.fc2(new_tensor)
        return input_data


def create_current_complex_demo_net(internal_nets: Tuple[NodeANNDataElemInterface, ...], /) \
        -> CurrentNetData:
    children_list = [
        deepcopy(child.get_savable_data().ann_container.get_net_lego) for child in internal_nets
    ]
    puf_com = BContainerComplexNet(BNetComplexArgs(
        lego=nn.ModuleList(children_list),
        fc2=nn.Linear(
            sum((list(child.children())[-1].out_features for child in children_list)), 10
        )
    ))
    return CurrentNetData(
        fitness=float('inf'),
        com=puf_com,
        lego=puf_com
    )


@final
@dataclass
class BComplexDemoInitCon(InitContainer):
    children_ids: Tuple[str, ...]
    lego_init_con: Tuple[LegoContInit, ...]


def _calc_weight_num(child: nn.Module, /) -> int:
    child_nodes_l: List[nn.Module] = list(child.children())
    return child_nodes_l[-1].state_dict()['weight'].size(0)


def load_current_complex_demo(init_args: BComplexDemoInitCon, /) -> CurrentNetData:
    if not init_args.children_ids:
        raise KnownComplexDemoError("Expected at least one child!")
    if len(init_args.children_ids) != len(init_args.lego_init_con):
        raise KnownComplexDemoError("Mismatch in init container and children!")
    buffered_list = [
        get_pytorch_net_creators().get(child, None)
        for child in init_args.children_ids
    ]
    if None in buffered_list:
        raise KnownComplexDemoError("Some children ids could not be found!")
    children_list: List[nn.Module] = [
        el(init_args.lego_init_con[child_id]).lego
        for child_id, el in enumerate(buffered_list) if el is not None
    ]
    puf_com = BContainerComplexNet(BNetComplexArgs(
        lego=nn.ModuleList(children_list),
        fc2=nn.Linear(sum(_calc_weight_num(child) for child in children_list), 10)
    ))
    return CurrentNetData(
        fitness=float('inf'),
        com=puf_com,
        lego=puf_com
    )
