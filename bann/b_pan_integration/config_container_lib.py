# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
from bann.b_pan_integration.net_connection_lib import net_connection_config
from bann.b_pan_integration.net_interface_lib import net_interface_config
from bann.b_pan_integration.net_plotter import net_plotter
from pan.public.functions.check_config_container import ConfigContainer


def create_config_container() -> ConfigContainer:
    return ConfigContainer(
        net_container=net_interface_config(),
        connection_container=net_connection_config(),
        plotter=net_plotter()
    )
