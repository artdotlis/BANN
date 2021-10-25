# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from enum import Enum
from typing import final


@final
class ENetInterfaceNames(Enum):
    # demo
    SIMPLEDEMONET = 'SimpleDemoNet'
    SIMPLEDEMOGANNET = 'SimpleDemoGANNet'
    SIMPLEDEMOPRENET = 'SimpleDemoPreNet'
    COMPLEXDEMONET = 'ComplexDemoNet'
    RBMDEMONET = 'RBMDemoNet'
    AEDEMONET = 'AEDemoNet'
    # e_networks


@final
class EPConnectionNames(Enum):
    # demo
    DEMO = 'DemoConnection'
    # e_networks
