# -*- coding: utf-8 -*-
""".. moduleauthor:: Artur Lissin"""
import argparse
import sys
from pathlib import Path

import torch

from bann.b_pan_integration.framwork_key_lib import FrameworkKeyLib
from bann.b_pan_integration.net_interface_lib import net_interface_config

from pan.public.functions.load_net_function import net_interface_load_net


def _load_save_net(in_file: Path, out_folder: Path, class_name: str, framework: str, /) -> None:
    interface_lib = net_interface_config()
    if framework not in interface_lib:
        print(f"Could not find framework: {framework}\n{','.join(interface_lib)}")
        sys.exit()
    class_lib = interface_lib[framework].net_dict
    if class_name not in class_lib:
        print(f"Could not find framework: {class_name}\n{','.join(class_lib)}")
        sys.exit()
    if not (out_folder.exists() and out_folder.is_dir()):
        out_folder.mkdir()
    with in_file.open('rb') as net_h:
        buffer_load = net_interface_load_net(class_lib[class_name].net_type, net_h.read())
        if framework == FrameworkKeyLib.PYTORCH.value:
            torch.save(
                buffer_load.ann_container.get_net_com.state_dict(),
                str(out_folder.joinpath("state_dict.pt"))
            )
        else:
            print(f"Does not support a saving option for the given framework: {framework}")


def load_net_save_state() -> None:
    arg_parser = argparse.ArgumentParser(
        prog='bann_lnss',
        description="A script for loading a network and saving its state_dict"
    )
    arg_parser.add_argument(
        '-o', '--out',
        action='store',
        type=str,
        required=True,
        help='the output folder',
        dest='out_folder',
        metavar='str'
    )
    arg_parser.add_argument(
        '-i', '--input',
        action='store',
        type=str,
        required=True,
        help='the input file',
        dest='in_file',
        metavar='str'
    )
    arg_parser.add_argument(
        '-c', '--class',
        action='store',
        type=str,
        required=True,
        help='the class name of the saved network',
        dest='class_name',
        metavar='str'
    )
    arg_parser.add_argument(
        '-f', '--framework',
        action='store',
        type=str,
        required=True,
        help='the framework name of the saved network',
        dest='framework_name',
        metavar='str'
    )
    args = arg_parser.parse_args()
    _load_save_net(Path(args.in_file), Path(args.out_folder), args.class_name, args.framework_name)


if __name__ == '__main__':
    load_net_save_state()
