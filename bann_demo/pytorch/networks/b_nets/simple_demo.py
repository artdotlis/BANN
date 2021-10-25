# -*- coding: utf-8 -*-
"""
LICENSE (BSD 3-Clause License): see extra_licenses/LICENSE_P_EXAMPLES

.. moduleauthor:: Artur Lissin
"""
from dataclasses import dataclass
from typing import Optional, Iterable, Any, Tuple, final, Dict, List

import torch
from torch import nn
from torch.nn import functional as nn_fun

from bann.b_frameworks.pytorch.interfaces.truth_interface import TruthClassInterface
from bann.b_container.states.framework.pytorch.criterion_param import CriterionAlias
from bann.b_frameworks.pytorch.interfaces.gan_interface import GanInterface, GanInTarget
from bann.b_frameworks.pytorch.interfaces.glw_pretraining_interface import \
    GLWPNetInterface
from bann.b_frameworks.pytorch.net_model_interface import CurrentNetData
from bann.b_frameworks.pytorch.pytorch_lego_const import LegoContInit

from bann_demo.pytorch.networks.errors.custom_erors import KnownSimpleDemoError


@dataclass
class BNetLegoArgs:
    conv1: nn.Module
    pool: nn.Module
    conv2: nn.Module
    fc1: nn.Module


@final
@dataclass
class BNetComArgs(BNetLegoArgs):
    fc2: nn.Module


@dataclass
class BGANNetLegoArgs:
    conv1: nn.Module
    conv2: nn.Module
    conv3: nn.Module
    conv4: nn.Module
    fc1: nn.Module


@final
@dataclass
class BGANNetComArgs(BGANNetLegoArgs):
    fc2: nn.Module


@final
class BContainerDemoNet(nn.Module):

    def __init__(self, complete: Optional[BNetComArgs], lego: Optional[BNetLegoArgs], /) -> None:
        super().__init__()
        self.__complete = True
        self.__lego = True

        if complete is not None:
            self.__lego = False
            self.conv1: nn.Module = complete.conv1
            self.pool: nn.Module = complete.pool
            self.conv2: nn.Module = complete.conv2
            self.fc1: nn.Module = complete.fc1
            self.fc2: nn.Module = complete.fc2
        elif lego is not None:
            self.__complete = False
            self.conv1 = lego.conv1
            self.pool = lego.pool
            self.conv2 = lego.conv2
            self.fc1 = lego.fc1
        else:
            raise KnownSimpleDemoError("Complete net args and lego args can not be both None!")

    def forward(self, *input_args: Any) -> torch.Tensor:
        if not (input_args and isinstance(input_args[0], torch.Tensor)):
            raise KnownSimpleDemoError(
                "Input data should not be empty and only contain tensor types"
            )
        input_data = input_args[0]
        input_data = self.pool(nn_fun.relu(self.conv1(input_data)))
        input_data = self.pool(nn_fun.relu(self.conv2(input_data)))
        input_data = input_data.view(-1, 4 * 4 * 3)
        input_data = self.fc1(input_data)
        if self.__complete:
            input_data = self.fc2(nn_fun.relu(input_data))
        return input_data


@final
class _PreTrainCNNElem(nn.Module):

    def __init__(self, layer: nn.Module, pool: nn.Module,
                 last_layer: nn.Module, last_layer_cnt: int, /) -> None:
        super().__init__()
        self.conv1: nn.Module = layer
        self.pool: nn.Module = pool
        self.last_layer: nn.Module = last_layer
        self.__last_layer: int = last_layer_cnt

    def forward(self, *input_args: Any) -> Tuple[torch.Tensor, ...]:
        if not (input_args and isinstance(input_args[0], torch.Tensor)):
            raise KnownSimpleDemoError(
                "Input data should not be empty and only contain tensor types"
            )
        input_data = input_args[0]
        input_data = self.pool(nn_fun.relu(self.conv1(input_data)))
        input_data = input_data.view(-1, self.__last_layer)
        input_data = self.last_layer(input_data)
        return input_data,


@final
class _PreTrainANNElem(nn.Module):
    def __init__(self, hidden_layer: nn.Module, last_layer: nn.Module, /) -> None:
        super().__init__()
        self.hidden_layer: nn.Module = hidden_layer
        self.last_layer: nn.Module = last_layer

    def forward(self, *input_args: Any) -> Tuple[torch.Tensor, ...]:
        if not (input_args and isinstance(input_args[0], torch.Tensor)):
            raise KnownSimpleDemoError(
                "Input data should not be empty and only contain tensor types"
            )
        input_data = input_args[0]
        input_data = nn_fun.relu(self.hidden_layer(input_data))
        input_data = self.last_layer(input_data)
        return input_data,


@final
class BContainerPreDemoNet(GLWPNetInterface, nn.Module):

    def __init__(self, complete_net: BNetComArgs, /) -> None:
        super().__init__()
        self.conv1: nn.Module = complete_net.conv1
        self.pool: nn.Module = complete_net.pool
        self.conv2: nn.Module = complete_net.conv2
        self.fc1: nn.Module = complete_net.fc1
        self.fc2: nn.Module = complete_net.fc2

    def forward(self, *input_args: Any) -> torch.Tensor:
        if not (input_args and isinstance(input_args[0], torch.Tensor)):
            raise KnownSimpleDemoError(
                "Input data should not be empty and only contain tensor types"
            )
        input_data = input_args[0]
        input_data = self.pool(nn_fun.relu(self.conv1(input_data)))
        input_data = self.pool(nn_fun.relu(self.conv2(input_data)))
        input_data = input_data.view(-1, 4 * 4 * 3)
        input_data = nn_fun.relu(self.fc1(input_data))
        input_data = self.fc2(input_data)
        return input_data

    def prepare_input(self, layer: int, in_data: Tuple[torch.Tensor, ...], /) \
            -> Tuple[torch.Tensor, ...]:
        input_data = in_data[0]
        with torch.no_grad():
            if layer >= 1:
                input_data = self.pool(nn_fun.relu(self.conv1(input_data)))
            if layer >= 2:
                input_data = self.pool(nn_fun.relu(self.conv2(input_data)))
                input_data = input_data.view(-1, 4 * 4 * 3)
        if layer >= 3:
            raise KnownSimpleDemoError("Third layer is not defined!")
        res = (input_data, )
        return res

    def prepare_output(self, layer: int, in_data: Tuple[torch.Tensor, ...], /) -> torch.Tensor:
        return in_data[0]

    def prepare_target(self, layer: int, in_data: torch.Tensor, /) -> torch.Tensor:
        return in_data

    def get_stack(self) -> Iterable[nn.Module]:
        yield _PreTrainCNNElem(self.conv1, self.pool, nn.Linear(12 * 12 * 3, 10), 12 * 12 * 3)
        yield _PreTrainCNNElem(self.conv2, self.pool, nn.Linear(4 * 4 * 3, 10), 4 * 4 * 3)
        yield _PreTrainANNElem(self.fc1, self.fc2)

    @property
    def get_stack_first(self) -> nn.Module:
        return _PreTrainCNNElem(self.conv1, self.pool, nn.Linear(12 * 12 * 3, 10), 12 * 12 * 3)


def create_current_demo_net(_: LegoContInit, /) -> CurrentNetData:
    networks_temp = {
        'conv1': nn.Conv2d(1, 3, 5, 1),
        'pool': nn.MaxPool2d(2, 2),
        'conv2': nn.Conv2d(3, 3, 5, 1),
        'fc1': nn.Linear(4 * 4 * 3, 24),
        'fc2': nn.Linear(24, 10)
    }
    puf_com = BContainerDemoNet(BNetComArgs(**networks_temp), None)
    del networks_temp['fc2']
    puf_lego = BContainerDemoNet(None, BNetLegoArgs(**networks_temp))
    return CurrentNetData(
        fitness=float('inf'),
        com=puf_com,
        lego=puf_lego
    )


def create_pre_training_demo_net(_: LegoContInit, /) -> CurrentNetData:
    networks_temp = {
        'conv1': nn.Conv2d(1, 3, 5, 1),
        'pool': nn.MaxPool2d(2, 2),
        'conv2': nn.Conv2d(3, 3, 5, 1),
        'fc1': nn.Linear(4 * 4 * 3, 24),
        'fc2': nn.Linear(24, 10)
    }
    puf_com = BContainerPreDemoNet(BNetComArgs(**networks_temp))
    del networks_temp['fc2']
    puf_lego = BContainerDemoNet(None, BNetLegoArgs(**networks_temp))
    return CurrentNetData(
        fitness=float('inf'),
        com=puf_com,
        lego=puf_lego
    )


@final
class _DNet(nn.Module):

    def __init__(self, complete: Optional[BGANNetComArgs],
                 lego: Optional[BGANNetLegoArgs], /) -> None:
        super().__init__()
        self.__complete = True
        self.__lego = True

        if complete is not None:
            self.__lego = False
            self.conv1: nn.Module = complete.conv1
            self.conv2: nn.Module = complete.conv2
            self.conv3: nn.Module = complete.conv3
            self.conv4: nn.Module = complete.conv4
            self.fc1: nn.Module = complete.fc1
            self.fc2: nn.Module = complete.fc2
        elif lego is not None:
            self.__complete = False
            self.conv1 = lego.conv1
            self.conv2 = lego.conv2
            self.conv3 = lego.conv3
            self.conv4 = lego.conv4
            self.fc1 = lego.fc1
        else:
            raise KnownSimpleDemoError("Complete net args and lego args can not be both None!")

    def forward(self, *input_args: Any) -> torch.Tensor:
        if not (input_args and isinstance(input_args[0], torch.Tensor)):
            raise KnownSimpleDemoError(
                "Input data should not be empty and only contain tensor types"
            )
        input_data = input_args[0]
        input_data = nn_fun.relu(self.conv1(input_data))
        input_data = nn_fun.relu(self.conv2(input_data))
        input_data = nn_fun.relu(self.conv3(input_data))
        input_data = nn_fun.relu(self.conv4(input_data))
        input_data = input_data.view(-1, 3 * 12 * 12)
        input_data = self.fc1(input_data)
        if self.__complete:
            input_data = self.fc2(nn_fun.relu(input_data))
        return input_data


def _weights_init(module: nn.Module, /) -> None:
    # see example: https://github.com/pytorch/examples/blob/master/dcgan/main.py
    if isinstance(module, nn.Conv2d):
        module.weight.data.normal_(0.0, 0.02)


@final
class _GNet(nn.Module):

    def __init__(self, net: BGANNetComArgs, /) -> None:
        super().__init__()
        conv_4 = net.conv4
        if not isinstance(conv_4, nn.Conv2d):
            raise KnownSimpleDemoError("Extreme error")
        conv_3 = net.conv3
        if not isinstance(conv_3, nn.Conv2d):
            raise KnownSimpleDemoError("Extreme error")
        conv_2 = net.conv2
        if not isinstance(conv_2, nn.Conv2d):
            raise KnownSimpleDemoError("Extreme error")
        conv_1 = net.conv1
        if not isinstance(conv_1, nn.Conv2d):
            raise KnownSimpleDemoError("Extreme error")
        self.conv4 = nn.ConvTranspose2d(
            conv_4.out_channels, conv_4.in_channels,
            (conv_4.kernel_size[0], conv_4.kernel_size[1]), bias=False
        )
        self.conv3 = nn.ConvTranspose2d(
            conv_3.out_channels, conv_3.in_channels,
            (conv_3.kernel_size[0], conv_3.kernel_size[1]), bias=False
        )
        self.conv2 = nn.ConvTranspose2d(
            conv_2.out_channels, conv_2.in_channels,
            (conv_2.kernel_size[0], conv_2.kernel_size[1]), bias=False
        )
        self.conv1 = nn.ConvTranspose2d(
            conv_1.out_channels, conv_1.in_channels,
            (conv_1.kernel_size[0], conv_1.kernel_size[1]), bias=False
        )
        self.apply(_weights_init)

    def forward(self, *input_args: Any) -> torch.Tensor:
        input_data = input_args[0]
        if not isinstance(input_data, torch.Tensor):
            raise KnownSimpleDemoError(
                "Input data should not be empty and only contain tensor types"
            )
        input_data = nn_fun.relu(self.conv4(input_data))
        input_data = nn_fun.relu(self.conv3(input_data))
        input_data = nn_fun.relu(self.conv2(input_data))
        input_data = torch.tanh(self.conv1(input_data))
        return input_data


@final
class BContainerDemoGANNet(GanInterface[torch.Tensor, torch.Tensor], nn.Module):

    def __init__(self, complete: Optional[BGANNetComArgs],
                 lego: Optional[BGANNetLegoArgs], /) -> None:
        super().__init__()
        self.__complete = True
        self.__lego = True
        self.d_net = _DNet(complete, lego)
        self.__batch_factor: float = 0.95
        if complete is not None:
            self.__lego = False
            self.g_net = _GNet(complete)
        elif lego is not None:
            self.__complete = False
        else:
            raise KnownSimpleDemoError("Complete net args and lego args can not be both None!")

    def fix_target_d(self, targets: torch.Tensor, device: torch.device, /) -> torch.Tensor:
        return torch.ones(targets.size(0), dtype=torch.long, device=device)

    def create_input_target(self, input_t: Tuple[torch.Tensor, ...], device: torch.device,
                            generator: bool, /) -> GanInTarget:
        if self.__lego:
            raise KnownSimpleDemoError("lego-nets don't implement a generator")
        if len(input_t) != 1:
            raise KnownSimpleDemoError(f"generator expected tuple of length 1, got {len(input_t)}")
        batch_s = int(self.__batch_factor * input_t[0].size(0))
        if generator:
            batch_s = int(input_t[0].size(0))
        if batch_s < 1:
            batch_s = 1
        res_net = self.g_net(torch.randn((batch_s, 3, 12, 12), requires_grad=True, device=device))
        if generator:
            target = torch.ones(res_net.size(0), dtype=torch.long, device=device)
        else:
            target = torch.zeros(res_net.size(0), dtype=torch.long, device=device)
        return GanInTarget(input=(res_net, ), target=target)

    def forward_gan(self, input_t: Tuple[torch.Tensor, ...],
                    device: torch.device, /) -> torch.Tensor:
        if self.__lego:
            raise KnownSimpleDemoError("lego-nets don't implement a generator")
        return self.d_net(*input_t)

    @property
    def generator(self) -> nn.Module:
        if self.__lego:
            raise KnownSimpleDemoError("lego-nets don't implement a generator")
        return self.g_net

    @property
    def discriminator(self) -> nn.Module:
        return self.d_net

    @property
    def layer_modules(self) -> Dict[str, nn.Module]:
        if self.__lego:
            raise KnownSimpleDemoError("lego-nets don't support layered optimisation")
        return {'d_net': self.d_net, 'g_net': self.g_net}

    @property
    def layer_names(self) -> List[str]:
        if self.__lego:
            raise KnownSimpleDemoError("lego-nets don't support layered optimisation")
        return ['d_net', 'g_net']

    def forward(self, *input_args: Any) -> torch.Tensor:
        return self.d_net(*input_args)

    def criterion(self, output_d: torch.Tensor, target_d: torch.Tensor, device: torch.device,
                  criterion: CriterionAlias, /) -> torch.Tensor:
        return criterion(output_d, target_d)

    def truth(self, output_d: torch.Tensor, target_d: torch.Tensor, device: torch.device,
              truth_fun: TruthClassInterface, /) -> float:
        return truth_fun.calc_truth(truth_fun.cr_truth_container(output_d, target_d, device))


def create_gan_demo_net(_: LegoContInit, /) -> CurrentNetData:
    networks_temp = {
        'conv1': nn.Conv2d(1, 3, 5, 1),
        'conv2': nn.Conv2d(3, 3, 5, 1),
        'conv3': nn.Conv2d(3, 3, 5, 1),
        'conv4': nn.Conv2d(3, 3, 5, 1),
        'fc1': nn.Linear(3 * 12 * 12, 24),
        'fc2': nn.Linear(24, 2)
    }
    puf_com = BContainerDemoGANNet(BGANNetComArgs(**networks_temp), None)
    del networks_temp['fc2']
    puf_lego = BContainerDemoGANNet(None, BGANNetLegoArgs(**networks_temp))
    return CurrentNetData(
        fitness=float('inf'),
        com=puf_com,
        lego=puf_lego
    )
